"""
Rule-based section extractor for DEF 14A proxy statements.

Converts raw HTML to clean text and locates two sections:
  1. Summary Compensation Table  → fed to the comp-table LLM prompt
  2. Director/nominee bio section → fed to the governance LLM prompt

Deliberately avoids the LLM for anything that can be done reliably with
pattern matching, keeping LLM calls minimal and focused.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


_UNICODE_NOISE = re.compile(r"[​‌‍﻿­]+")  # zero-width spaces / soft-hyphen in iXBRL filings

def html_to_lines(path: Path) -> list[str]:
    """Parse a DEF 14A HTML file and return cleaned, non-empty lines."""
    try:
        soup = BeautifulSoup(path.read_bytes(), "lxml")
    except Exception:
        return []
    text = soup.get_text(separator="\n", strip=True)
    text = _UNICODE_NOISE.sub("", text)
    return [l.strip() for l in text.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Section patterns
# ---------------------------------------------------------------------------

_DIRECTOR_START = re.compile(
    r"(information about (the )?nominees|directors? and executive officers|"
    r"election of directors|nominees? for (the )?board|"
    r"information about directors?)",
    re.IGNORECASE,
)

_DIRECTOR_END = re.compile(
    r"^(board diversity|executive compensation|compensation of directors|"
    r"director compensation|compensation discussion|proposal\s+2|ratification|"
    r"security ownership|voting securities|certain relationships|"
    r"shareholder approval|stockholder approval|audit committee report|"
    r"related.party transactions?)$",
    re.IGNORECASE,
)

_COMP_START = re.compile(
    r"(summary compensation table|executive compensation\b)",
    re.IGNORECASE,
)

_COMP_END = re.compile(
    r"(grants of plan.based awards|outstanding equity awards|"
    r"option exercises|pension benefits|nonqualified deferred compensation(?! earnings)|"
    r"potential payments|pay versus performance|ceo pay ratio|"
    r"director compensation)",
    re.IGNORECASE,
)

_NAME_WORD = r"(?:[A-Z][a-z][A-Za-z\-]*|[A-Z]\.|II?I?V?|IV|VI?I?I?)"
_NAME_LINE = re.compile(rf"^{_NAME_WORD}(?:[,]? {_NAME_WORD}){{1,5}}[,]?\.?$")

_BIO_STARTERS = re.compile(
    r"^(has |was |is |served |joined |founded |currently |previously |"
    r"began |brings |The Nominating |Mr\.|Ms\.|Dr\.)",
    re.IGNORECASE,
)


def _extract_section(
    lines: list[str],
    start_pat: re.Pattern,
    end_pat: re.Pattern,
    max_lines: int = 600,
    use_last: bool = False,
) -> str:
    matches = [i for i, l in enumerate(lines) if start_pat.search(l)]
    if not matches:
        return ""
    start_idx = matches[-1] if use_last else matches[0]
    result = []
    for line in lines[start_idx: start_idx + max_lines]:
        if result and end_pat.search(line):
            break
        result.append(line)
    return "\n".join(result)


def _is_bio_name(line: str) -> bool:
    if not _NAME_LINE.match(line):
        return False
    lower = line.lower().rstrip(".")
    skip = {
        "board of directors", "executive officers", "audit committee",
        "compensation committee", "nominating committee", "corporate governance",
        "director since", "summary compensation", "united states", "federal reserve",
        "annual meeting", "common stock", "new york", "total compensation",
    }
    return lower not in skip and len(line) <= 65


def _has_bio_text(lines: list[str], name_idx: int) -> bool:
    name_parts = lines[name_idx].rstrip(".").split()
    first_name = name_parts[0] if name_parts else ""
    for j in range(name_idx + 1, min(name_idx + 5, len(lines))):
        l = lines[j].strip()
        if not l:
            continue
        if len(l) < 30:
            return False
        if _BIO_STARTERS.match(l):
            return True
        if first_name and l.startswith(first_name):
            return True
        return False
    return False


def extract_director_section(lines: list[str]) -> str:
    """
    Extract the director/nominee bio section.

    Finds the first standalone name line followed by actual bio prose —
    this skips the TOC summary table where names appear without paragraphs.
    """
    start_idx = None
    for i, line in enumerate(lines):
        if _is_bio_name(line) and _has_bio_text(lines, i):
            start_idx = max(0, i - 3)
            break

    if start_idx is None:
        return _extract_section(lines, _DIRECTOR_START, _DIRECTOR_END,
                                max_lines=800, use_last=True)

    result = []
    for line in lines[start_idx: start_idx + 800]:
        if result and _DIRECTOR_END.search(line):
            break
        result.append(line)
    return "\n".join(result)


_COMP_DATA_YEAR   = re.compile(r"\b20[12]\d\b")
_COMP_DATA_DOLLAR = re.compile(r"\$?\s*[\d,]{5,}")


def _has_comp_data(text: str, window: int = 800) -> bool:
    """True if text contains a year (20XX) and a dollar amount within `window` chars."""
    w = text[:window]
    return bool(_COMP_DATA_YEAR.search(w)) and bool(_COMP_DATA_DOLLAR.search(w))


def _extract_from_idx(lines: list[str], start_idx: int) -> str:
    result = []
    for line in lines[max(0, start_idx - 1): start_idx + 300]:
        if result and _COMP_END.search(line):
            break
        result.append(line)
    return "\n".join(result)


_PAGE_REF = re.compile(r"^(Page\s*\d{1,3}|pg\.?\s*\d{1,3})$", re.IGNORECASE)


def _is_toc_line(lines: list[str], idx: int) -> bool:
    """True if line idx looks like a TOC entry (bare page number before or after)."""
    _page_num = re.compile(r"^\d{1,3}$")
    before = lines[idx - 1].strip() if idx - 1 >= 0 else ""
    after  = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
    return (
        bool(_page_num.match(before)) or bool(_page_num.match(after)) or
        bool(_PAGE_REF.match(before)) or bool(_PAGE_REF.match(after))
    )


def extract_compensation_section(lines: list[str]) -> str:
    """
    Extract the Summary Compensation Table section.

    Strategy:
      1. Find all short (< 40 char) occurrences of "Summary Compensation Table",
         excluding Pay-vs-Performance column labels and inline cross-references.
      2. Filter TOC entries: bare page number (digits or "Page N") on the line
         immediately before OR after, including non-breaking-space variants.
      3. From remaining candidates, pick the FIRST one whose extracted text
         actually contains compensation data (year + dollar amount).
      4. If no candidate passes the data check, fall through to the broader
         _COMP_START pattern rather than returning a TOC entry.
      5. Broader pattern tried twice: first occurrence first (section header),
         then last occurrence (avoids say-on-pay narrative at end).
      6. Last resort: scan for Salary + Total column-header cluster.
    """
    summary_pat = re.compile(r"summary compensation table", re.IGNORECASE)
    pvp_suffix  = re.compile(r"summary compensation table\s+(total|amount)", re.IGNORECASE)
    inline_ref  = re.compile(
        r"summary compensation table[,.]?\s*(above|below|herein|set forth|included)",
        re.IGNORECASE,
    )
    short_matches = [
        i for i, l in enumerate(lines)
        if summary_pat.search(l) and len(l) < 40
        and not pvp_suffix.search(l)
        and not inline_ref.search(l)
    ]

    if short_matches:
        candidates = [i for i in short_matches if not _is_toc_line(lines, i)]
        if not candidates:
            candidates = short_matches

        for idx in candidates:
            text = _extract_from_idx(lines, idx)
            if _has_comp_data(text):
                return text
        # No candidate had real data — fall through to broader patterns below
        # rather than returning a TOC entry or cross-reference snippet.

    # Broader pattern: first occurrence is the actual section header;
    # last occurrence risks landing in a say-on-pay narrative near the end.
    broader_first = _extract_section(lines, _COMP_START, _COMP_END,
                                     max_lines=300, use_last=False)
    if broader_first and _has_comp_data(broader_first):
        return broader_first

    broader_last = _extract_section(lines, _COMP_START, _COMP_END,
                                    max_lines=300, use_last=True)
    if broader_last and _has_comp_data(broader_last):
        return broader_last

    # Last resort: scan for Salary + Total column-header cluster
    salary_pat = re.compile(r"\bSalary\b", re.IGNORECASE)
    total_pat  = re.compile(r"\bTotal\b",  re.IGNORECASE)
    for i, line in enumerate(lines):
        if salary_pat.search(line):
            if total_pat.search("\n".join(lines[i: i + 5])):
                text = _extract_from_idx(lines, i)
                if _has_comp_data(text):
                    return text

    return broader_last or ""
