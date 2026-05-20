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
    r"option exercises|pension benefits|nonqualified deferred compensation plan|"
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


def extract_compensation_section(lines: list[str]) -> str:
    """
    Extract the Summary Compensation Table section.

    Challenges solved:
    - TOC entries: the heading "Summary Compensation Table" appears in the
      table of contents (followed by a page number) before the actual section.
      We filter these out by detecting when the next line is a bare 1-3 digit
      page number.
    - Compensation Actually Paid (CAP) references: some filings embed the
      heading as a column label in the Pay vs Performance reconciliation table
      before the real section. Taking the *last* surviving candidate avoids
      these earlier false matches.
    """
    summary_pat = re.compile(r"summary compensation table", re.IGNORECASE)
    # "Summary Compensation Table Total/Amount" are Pay-vs-Performance column
    # labels, not section headers — exclude them to avoid picking the PvP table.
    pvp_suffix = re.compile(r"summary compensation table\s+(total|amount)", re.IGNORECASE)
    short_matches = [i for i, l in enumerate(lines)
                     if summary_pat.search(l) and len(l) < 40
                     and not pvp_suffix.search(l)]

    if short_matches:
        # Filter TOC entries: next line is a bare page number (1-3 digits)
        candidates = [
            i for i in short_matches
            if not (i + 1 < len(lines)
                    and re.match(r"^\d{1,3}$", lines[i + 1].strip()))
        ]
        # Prefer the last candidate — skips CAP table column-header references
        # that appear before the actual section in some filings
        start_idx = candidates[-1] if candidates else short_matches[-1]

        result = []
        for line in lines[max(0, start_idx - 1): start_idx + 300]:
            if result and _COMP_END.search(line):
                break
            result.append(line)
        return "\n".join(result)

    # Fallback: broader "executive compensation" pattern, last occurrence
    return _extract_section(lines, _COMP_START, _COMP_END,
                            max_lines=300, use_last=True)
