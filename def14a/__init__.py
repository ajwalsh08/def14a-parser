"""def14a-parser: extract CEO compensation and governance data from SEC DEF 14A filings."""
from .extractor import extract, DEFAULT_MODEL
from .parser import html_to_lines, extract_compensation_section, extract_director_section

__all__ = [
    "extract",
    "DEFAULT_MODEL",
    "html_to_lines",
    "extract_compensation_section",
    "extract_director_section",
]
