"""
LLM-based extractor for DEF 14A proxy statements via Ollama.

Two calls per filing:
  1. Comp table   → CEO name / title / salary / total + all NEO rows
  2. Governance   → board_size, pct_independent, auditor, say_on_pay_pct,
                    ceo_chairman_combined, fiscal_year, CEO career summary

Strategy: ask literal questions and request pipe-delimited rows rather than
demanding JSON schema compliance. Instruction-tuned models answer questions
reliably but resist filling structured schemas. We parse the answers.

Recommended models (via `ollama pull`):
  llama3.2:3b   — 2 GB,  ~27s/filing, good accuracy
  qwen2.5:7b    — 4.7 GB, ~60s/filing, better accuracy on ambiguous tables
  gemma3:4b     — 3.3 GB, ~35s/filing, strong instruction following
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import requests

from .parser import (
    html_to_lines,
    extract_director_section,
    extract_compensation_section,
)

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2:3b"

_DIR_MAX_CHARS  = 10_000   # governance context: director bio section
_DOC_PREFIX     = 20_000   # governance context: fallback (top of document)
_COMP_MAX_CHARS =  8_000   # compensation context: comp table section

# ── Prompt 1: compensation table ───────────────────────────────────────────
_COMP_PROMPT = """\
From this compensation table, list each named executive officer. \
One row per person per year reported.

Format — exactly six pipe-separated columns, first column is ALWAYS the person's full name:
Name | Title | CEO | Year | Salary | Total

Rules:
- CEO column: yes if this person is the CEO or Principal Executive Officer, else no
- Use 0 for any cell that shows a dash or is blank
- Never put "yes" or "no" in the Name column

Example output:
Jane Doe | Chief Executive Officer | yes | 2023 | 850000 | 3200000
John Smith | Chief Financial Officer | no | 2023 | 520000 | 1100000

COMPENSATION TABLE:
{text}"""

# ── Prompt 2: governance Q&A ───────────────────────────────────────────────
_GOV_PROMPT = """\
Read this proxy statement section and answer each question with a short direct answer.

Q1: What fiscal year does the compensation cover? (year number only)
Q2: Is the CEO also the Board Chair? (yes or no)
Q3: How many directors total are on the board? (number only, or: unknown)
Q4: What fraction of directors are independent? (decimal 0.0-1.0, or: unknown)
Q5: Who is the external auditor? (firm name only, or: unknown)
Q6: Most recent say-on-pay approval %? (number only, or: unknown)
Q7: Describe the CEO career in 1-2 sentences.

PROXY STATEMENT SECTION:
{text}"""


def _chat(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 180) -> str:
    """Send a single chat message to a local Ollama model."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":   model,
                "stream":  False,
                "options": {"temperature": 0.0, "num_predict": -1},
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content") or ""
    except Exception:
        return ""


# ── Compensation table parser ──────────────────────────────────────────────

def _dollars(s: str) -> Optional[int]:
    s = s.strip().lstrip("$").replace(",", "").replace("—", "").replace("-", "").strip()
    if not s or s.lower() in ("null", "unknown", "n/a", ""):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


_STANDALONE_NAME = re.compile(
    r"^(?:\d+\.\s+)?([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-\.]+){1,4}"
    r"(?:,?\s+(?:Jr\.?|Sr\.?|II|III|IV))?)$"
)
_HEADER_ROW = re.compile(r"Title.*CEO.*Year|Year.*Salary.*Total", re.IGNORECASE)
_CEO_TITLE   = re.compile(r"\bchief\s+executive|(?<!\S)ceo(?!\S)", re.IGNORECASE)


def _parse_comp_table(raw: str) -> list[dict]:
    """
    Parse the model's pipe-delimited compensation table response.

    Handles two formats the model produces:
      Flat:       Name | Title | CEO | Year | Salary | Total
      Multi-line: "N. Name" on its own line, then data rows below
    """
    rows: list[dict] = []
    current_name: Optional[str] = None
    current_is_ceo: Optional[bool] = None

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        if "|" not in line:
            m = _STANDALONE_NAME.match(line)
            if m:
                current_name = m.group(1).strip()
                current_is_ceo = None
            continue

        if re.match(r"^[\|\-\s]+$", line) or _HEADER_ROW.search(line):
            continue

        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]

        year_idx = next(
            (i for i, p in enumerate(parts) if re.fullmatch(r"20[12]\d", p)),
            None,
        )
        if year_idx is None:
            continue

        year = int(parts[year_idx])

        if current_name and year_idx <= 2:
            name  = current_name
            title = parts[0] if parts[0].lower() not in ("yes", "no") else (
                parts[1] if len(parts) > 1 else ""
            )
        else:
            name  = parts[0]
            title = parts[1] if year_idx >= 2 else ""
            if not name or len(name) < 3:
                name = current_name or ""

        if not name:
            continue

        pre_year     = " ".join(parts[:year_idx])
        is_ceo_flag  = bool(re.search(r"\byes\b", pre_year, re.IGNORECASE))
        is_ceo_title = bool(_CEO_TITLE.search(title))
        is_ceo       = is_ceo_flag or is_ceo_title

        if current_name and current_name == name:
            if is_ceo:
                current_is_ceo = True
            elif current_is_ceo is True:
                is_ceo = True

        numerics = [_dollars(p) for p in parts[year_idx + 1:]]
        numerics = [n for n in numerics if n is not None]
        salary = numerics[0] if numerics else None
        total  = numerics[-1] if len(numerics) >= 2 else None

        if salary is not None and total is not None and salary > total:
            total = None

        name = re.sub(r"^\d+\.\s+", "", name).strip()

        if name.lower() in ("yes", "no", "ceo", "n/a", "na", "unknown"):
            if current_name:
                name = current_name
            else:
                continue

        if re.search(r"\bpresident\b|\bchief\b|\bofficer\b|\bvice\b", name, re.IGNORECASE):
            if current_name:
                name = current_name
            else:
                continue

        rows.append({
            "name":       name,
            "title":      title,
            "is_ceo":     is_ceo,
            "year":       year,
            "salary":     salary,
            "total_comp": total,
        })

    return rows


# ── Governance Q&A parser ──────────────────────────────────────────────────

def _extract_qa(raw: str) -> dict[int, str]:
    answers: dict[int, str] = {}
    for line in raw.splitlines():
        m = re.match(r"Q(\d+)[:\.]?\s*(.+)", line.strip())
        if m:
            answers[int(m.group(1))] = m.group(2).strip()
    return answers


def _parse_int(s: str) -> Optional[int]:
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None


def _parse_year(s: str) -> Optional[int]:
    m = re.search(r"20[12]\d", s)
    return int(m.group()) if m else None


def _parse_float(s: str) -> Optional[float]:
    s = s.replace("%", "").strip()
    m = re.search(r"\d+\.?\d*", s)
    if not m:
        return None
    v = float(m.group())
    return round(v / 100 if v > 1.0 else v, 4)


def _parse_bool(s: str) -> bool:
    return bool(re.search(r"\byes\b", s, re.IGNORECASE))


_AUDIT_FIRMS = re.compile(
    r"(Deloitte\s*&?\s*Touche|Ernst\s*&?\s*Young|KPMG|PricewaterhouseCoopers|PwC"
    r"|Grant\s+Thornton|BDO|Crowe|RSM|Moss\s+Adams|Dixon\s+Hughes|Plante\s+Moran"
    r"|Forvis|CliftonLarsonAllen|Wipfli|[A-Z][A-Za-z\s&,]+(?:LLP|LLC|PC|CPAs?))",
    re.IGNORECASE,
)


def _find_auditor(full_text: str) -> Optional[str]:
    """Scan document text for auditor firm name near engagement-related language."""
    for pat in [
        r"appointment\s+of\s+((?:[A-Z][\w&,\s]+?)(?:LLP|LLC|PC|CPAs?|CPA)\.?)",
        r"retained\s+((?:[A-Z][\w&,\s]+?)(?:LLP|LLC|PC|CPAs?|CPA)\.?)\s+as",
        r"engaged\s+((?:[A-Z][\w&,\s]+?)(?:LLP|LLC|PC|CPAs?|CPA)\.?)\s+",
        r"((?:[A-Z][\w&,\s]+?)(?:LLP|LLC|PC|CPAs?|CPA)\.?)\s+(?:has\s+)?served\s+as\s+(?:our\s+)?(?:independent|external)",
        r"((?:[A-Z][\w&,\s]+?)(?:LLP|LLC|PC|CPAs?|CPA)\.?)\s+audited",
    ]:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip(",").strip()
            if 4 < len(candidate) < 60 and not re.search(
                r"\b(the|our|that|with|all)\b", candidate, re.IGNORECASE
            ):
                return candidate
    return None


def _parse_governance(raw: str) -> dict:
    qa  = _extract_qa(raw)
    gov: dict = {}

    gov["fiscal_year"]          = _parse_year(qa.get(1, ""))
    gov["ceo_chairman_combined"] = _parse_bool(qa.get(2, "")) if qa.get(2) else False
    gov["board_size"]            = _parse_int(qa.get(3, "")) if qa.get(3) and "unknown" not in qa.get(3, "").lower() else None
    gov["pct_independent"]       = _parse_float(qa.get(4, "")) if qa.get(4) and "unknown" not in qa.get(4, "").lower() else None
    gov["auditor"]               = qa.get(5) if qa.get(5) and "unknown" not in qa.get(5, "").lower() else None
    gov["say_on_pay_pct"]        = _parse_float(qa.get(6, "")) if qa.get(6) and "unknown" not in qa.get(6, "").lower() else None
    gov["ceo_career_summary"]    = qa.get(7)

    return gov


# ── Bio-section validator ──────────────────────────────────────────────────

_BIO_SIGNAL = re.compile(
    r"(^|\n)(Mr\.|Ms\.|Dr\.|"
    r"has served|served as|is a |was a |joined |founded |"
    r"has been |previously served|currently serves|began his|began her|"
    r"Extensive |Significant |More than \d|Over \d)",
    re.IGNORECASE,
)


def _is_bio_section(text: str) -> bool:
    return bool(_BIO_SIGNAL.search(text))


# ── Main entry point ───────────────────────────────────────────────────────

def extract(path: Path, model: str = DEFAULT_MODEL) -> Optional[dict]:
    """
    Extract CEO, governance, and NEO compensation from a DEF 14A filing.

    Parameters
    ----------
    path : Path
        Path to a downloaded DEF 14A HTML file (primary.htm or similar).
    model : str
        Ollama model name. Default: ``llama3.2:3b``.
        Use ``qwen2.5:7b`` for better accuracy on ambiguous documents.

    Returns
    -------
    dict or None
        Structured extraction result, or None if the file is empty or
        no compensation section could be found.

    Output schema::

        {
          "fiscal_year": 2023,
          "ceo": {
            "name": "Jane Doe",
            "title": "President & Chief Executive Officer",
            "is_chairman": false,
            "career_summary": "Ms. Doe joined Acme in 2015 ..."
          },
          "governance": {
            "board_size": 9,
            "pct_independent": 0.78,
            "ceo_chairman_combined": false,
            "say_on_pay_pct": 0.94,
            "auditor": "KPMG LLP"
          },
          "compensation": [
            {
              "name": "Jane Doe",
              "title": "President & CEO",
              "is_ceo": true,
              "year": 2023,
              "salary": 850000,
              "total_comp": 3200000
            },
            ...
          ]
        }
    """
    lines = html_to_lines(path)
    if not lines:
        return None

    full_text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))

    # Call 1: compensation table
    comp_section = extract_compensation_section(lines)
    if not comp_section:
        return None

    comp_raw  = _chat(_COMP_PROMPT.format(text=comp_section[:_COMP_MAX_CHARS]), model=model)
    comp_rows = _parse_comp_table(comp_raw)
    if not comp_rows:
        return None

    ceo_row  = next((r for r in comp_rows if r["is_ceo"]), comp_rows[0])
    ceo_name = ceo_row["name"]

    # Call 2: governance Q&A
    dir_section = extract_director_section(lines)
    if dir_section and len(dir_section) > 200 and _is_bio_section(dir_section):
        gov_ctx = dir_section[:_DIR_MAX_CHARS]
    else:
        gov_ctx = full_text[:_DOC_PREFIX]

    gov_raw = _chat(_GOV_PROMPT.format(text=gov_ctx), model=model)
    gov     = _parse_governance(gov_raw)

    comp_years = [r["year"] for r in comp_rows if r.get("year")]
    fy_qa      = gov.pop("fiscal_year", None)
    fy_comp    = max(comp_years) if comp_years else None
    fiscal_year = fy_comp or fy_qa

    auditor = gov.get("auditor")
    if not auditor:
        auditor = _find_auditor(full_text)

    return {
        "fiscal_year": fiscal_year,
        "ceo": {
            "name":           ceo_name,
            "title":          ceo_row.get("title"),
            "is_chairman":    gov.get("ceo_chairman_combined", False),
            "career_summary": gov.get("ceo_career_summary"),
        },
        "governance": {
            "board_size":            gov.get("board_size"),
            "pct_independent":       gov.get("pct_independent"),
            "ceo_chairman_combined": gov.get("ceo_chairman_combined", False),
            "say_on_pay_pct":        gov.get("say_on_pay_pct"),
            "auditor":               auditor,
        },
        "compensation": comp_rows,
    }
