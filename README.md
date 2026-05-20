# def14a-parser

Extract CEO compensation and governance data from SEC DEF 14A proxy statements using a **local LLM via [Ollama](https://ollama.com)** — no API keys, no cloud costs, runs on a laptop.

```json
{
  "fiscal_year": 2023,
  "ceo": {
    "name": "Jane Doe",
    "title": "President & Chief Executive Officer",
    "is_chairman": false,
    "career_summary": "Ms. Doe joined Acme Bank in 2015 as CFO before becoming CEO in 2019."
  },
  "governance": {
    "board_size": 9,
    "pct_independent": 0.78,
    "ceo_chairman_combined": false,
    "say_on_pay_pct": 0.94,
    "auditor": "KPMG LLP"
  },
  "compensation": [
    {"name": "Jane Doe",  "title": "President & CEO", "is_ceo": true,  "year": 2023, "salary": 850000, "total_comp": 3200000},
    {"name": "John Smith","title": "Chief Financial Officer", "is_ceo": false, "year": 2023, "salary": 520000, "total_comp": 1100000}
  ]
}
```

## Why another DEF 14A parser?

Existing tools take one of three approaches:

| Approach | Example | Limitation |
|---|---|---|
| Rule-based HTML parsing | edgartools, ceopay | Breaks on inconsistent formatting across thousands of filers |
| Vision/ML table classifier | Execcomp-AI | Requires GPU, heavy dependency stack |
| Commercial API | sec-api | Paid, not reproducible |

This library uses a **Q&A prompt approach**: instead of demanding that the LLM output a rigid JSON schema (which instruction-tuned models resist), we ask literal factual questions and parse the answers ourselves. This is significantly more reliable across the wide variation in proxy statement formatting found in the wild — especially for smaller regional filers who don't follow large-company templates.

Two calls per filing:
1. **Comp table call** — "list each NEO as a pipe-delimited row: Name | Title | CEO | Year | Salary | Total"
2. **Governance Q&A** — seven numbered questions answered one per line (board size, % independent, auditor, say-on-pay %, CEO/chair combined, fiscal year, CEO bio)

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- At least one model pulled (see [Model recommendations](#model-recommendations))

## Installation

```bash
pip install def14a-parser
```

Or from source:

```bash
git clone https://github.com/ajwalsh08/def14a-parser
cd def14a-parser
pip install -e .
```

## Quick start

### Get a filing

DEF 14A filings are free from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar). Download the primary HTML document from any filing index page. Example using the SEC's EDGAR full-text search:

```bash
# Find a company's most recent proxy
# e.g. https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000037808&type=DEF+14A

# Download the primary document (respect SEC rate limits: max 10 req/s)
curl -A "Your Name your@email.com" \
  "https://www.sec.gov/Archives/edgar/data/37808/.../primary.htm" \
  -o primary.htm
```

### Python API

```python
import def14a

result = def14a.extract("primary.htm")
# or with a specific model:
result = def14a.extract("primary.htm", model="qwen2.5:7b")

print(result["ceo"]["name"])
print(result["governance"]["auditor"])
for row in result["compensation"]:
    print(f"{row['name']}: ${row['total_comp']:,}")
```

### CLI

```bash
# Single file — result to stdout
def14a run primary.htm

# Single file — write to JSON
def14a run primary.htm --output result.json

# Batch — each file gets a .json sidecar
def14a run filings/*/primary.htm --model qwen2.5:7b
```

## Output schema

| Field | Type | Description |
|---|---|---|
| `fiscal_year` | int | Year the compensation covers (from comp table, more reliable than Q&A) |
| `ceo.name` | str | CEO full name |
| `ceo.title` | str | CEO title as stated in the filing |
| `ceo.is_chairman` | bool | True if CEO is also Board Chair |
| `ceo.career_summary` | str | 1-2 sentence bio from the LLM |
| `governance.board_size` | int \| null | Total directors on board |
| `governance.pct_independent` | float \| null | Fraction of independent directors (0.0–1.0) |
| `governance.ceo_chairman_combined` | bool | CEO and Chair are the same person |
| `governance.say_on_pay_pct` | float \| null | Most recent advisory vote approval (0.0–1.0) |
| `governance.auditor` | str \| null | External auditor firm name |
| `compensation` | list[dict] | All named executive officers (NEOs) |
| `compensation[].name` | str | NEO full name |
| `compensation[].title` | str | NEO title |
| `compensation[].is_ceo` | bool | True for the principal executive officer |
| `compensation[].year` | int | Fiscal year for this row |
| `compensation[].salary` | int \| null | Base salary ($) |
| `compensation[].total_comp` | int \| null | Total compensation ($) |

## Model recommendations

Pull models with `ollama pull <name>` before use.

| Model | Size | Speed | Accuracy | Notes |
|---|---|---|---|---|
| `llama3.2:3b` | 2 GB | ~27s/filing | Good | Default; fastest option |
| `qwen2.5:7b` | 4.7 GB | ~60s/filing | Better | Recommended for production runs |
| `gemma3:4b` | 3.3 GB | ~35s/filing | Good | Strong instruction following |

For a corpus of 1,000+ filings: run `llama3.2:3b` first (fast, ~7.5 hrs), then `qwen2.5:7b --force` on the handful that produce bad CEO names (typically < 3%).

## How section extraction works

DEF 14A filings contain a table of contents that references "Summary Compensation Table" (with a page number) long before the actual section. Naive matching grabs the TOC entry and returns 3 lines of content instead of 300.

This library disambiguates using two heuristics:

1. **Page-number filter** — a TOC entry is immediately followed by a bare 1–3 digit page number. We skip any match with that pattern.
2. **Last-candidate preference** — some filings embed "Summary Compensation Table" as a column header in the Pay vs Performance reconciliation table (which appears *before* the actual comp section in unusual filing orderings). Taking the *last* surviving candidate avoids these false matches.

Together these recover ~56% of filings that naive parsers skip entirely.

## Parser-only mode (no LLM)

If you only need the raw section text:

```python
import def14a

lines        = def14a.html_to_lines("primary.htm")
comp_section = def14a.extract_compensation_section(lines)
dir_section  = def14a.extract_director_section(lines)
```

## Known limitations

- **Wrapper files** — EDGAR sometimes stores a brief index file as the primary document. These have no compensation content and return `None`. Download the actual exhibit document instead.
- **Non-standard headings** — a small minority of filers (mostly very small companies) label their comp section differently (e.g. "Named Executive Officer Compensation" without "Summary Compensation Table"). The fallback `executive compensation` pattern catches most of these.
- **Governance accuracy** — board size and auditor extraction depend on how clearly the filing is structured. Auditor is cross-checked with a regex scan of the full document text as a fallback.
- **Multi-company filings** — holding companies that file for multiple subsidiaries may return data for only one entity.

## Related projects

- [edgartools](https://github.com/dgunning/edgartools) — general-purpose SEC EDGAR library; DEF 14A support is broad but compensation extraction is shallow
- [ceopay](https://github.com/talsan/ceopay) — Random Forest table classifier + rule-based parsing; compensation only, no governance
- [Execcomp-AI](https://github.com/pierpierpy/Execcomp-AI) — VLM pipeline covering 2005–2022; requires GPU

## Contributing

Bug reports and PRs welcome. The most valuable contributions:
- Additional test cases (especially tricky real-world filings)
- Auditor name normalization entries
- Prompt improvements for edge cases

## License

MIT
