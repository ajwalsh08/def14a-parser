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

DEF 14A filings are free from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar). Download the primary HTML document from any filing index page.

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
| `ceo.career_summary` | str \| null | 1-2 sentence bio from the proxy statement |
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
| `llama3.2:3b` | 2 GB | ~27s/filing | Good | Fastest option |
| `qwen2.5:7b` | 4.7 GB | ~50s/filing | Best | Recommended for production runs |
| `gemma3:4b` | 3.3 GB | ~35s/filing | Good | Strong instruction following |

For a corpus of 1,000+ filings: run `llama3.2:3b` first (fast), then `qwen2.5:7b --force` on filings that produce bad CEO names (typically < 3%).

## Expected results

These numbers come from a production run against 1,347 DEF 14A filings from 236 US regional bank holding companies (SIC 6020/6022), fiscal years 2017–2026, using `qwen2.5:7b`.

**Extraction rate**

| Outcome | Count | % |
|---|---|---|
| Successfully extracted | 1,206 | 89.5% |
| Irreducible failures | 141 | 10.5% |

The 141 failures fall into three categories that no parser can recover:
- **Soliciting materials** (14a-12 filings) — brief pre-proxy documents with no comp table
- **Image-only tables** — scanned PDFs where the comp table is a JPEG; BeautifulSoup returns empty text
- **Word-per-line HTML** — unusual layouts where table cells are individual `<p>` tags; column alignment is unrecoverable without the visual structure

**Company coverage**

Of 236 companies with DEF 14A filings, 229 (97%) have at least one successful extraction. The 7 with zero extractions are all image-heavy filers across every year.

**Data quality of successful extractions**

| Check | Result |
|---|---|
| `ceo.name` populated | 100% |
| Salary data present | 98% |
| `is_ceo=True` row confirmed in comp table | 91% |
| CEO career summary populated | ~95% |

**The 9% without a confirmed `is_ceo=True` row** are structurally unusual filings — ESOP-contribution tables (where all participants are listed as "Employee"), mutual thrift conversions, and holding companies where the comp table covers subsidiary officers only. The `ceo.name` field is still populated for these (using the first-listed executive as a fallback), but treat it as lower-confidence.

**Known quality caveats**

- **Career summary accuracy (~62%)** — The governance Q&A asks "describe the CEO career" without anchoring to the CEO's name. In ~38% of filings the model pulls a bio paragraph for the board chair or another prominent director instead of the CEO. This does not affect compensation data. If career summary accuracy matters for your use case, re-run the governance call with a name-anchored prompt: "What is [CEO name]'s career background?"

- **Top-earner ≠ CEO (~10%)** — In holding companies with separately-compensated subsidiary CEOs, the highest-paid executive may not be the parent company CEO. This is structurally correct, not an extraction error.

- **`total_comp` null when salary > total** — Some filings report multi-year data where a prior-year total looks smaller than the current-year salary (e.g., a year with no equity awards). The parser sets `total_comp = None` in these cases rather than returning an implausible value.

## How section extraction works

DEF 14A filings contain several traps for a naive parser. The heading "Summary Compensation Table" appears in three places before the real section:

1. **Table of contents** — the heading followed immediately by a bare page number ("42")
2. **Pay-vs-Performance column label** — "Summary Compensation Table Total" as a column header in the PvP reconciliation (SEC rule added 2023)
3. **Narrative cross-references** — "as shown in the Summary Compensation Table above" mid-sentence

The section extractor works through three tiers:

**Tier 1 — Exact heading match (~85% of filings)**
Find all short (<40 char) standalone occurrences of "Summary Compensation Table". Exclude PvP column labels, inline directional references ("above"/"below"), and sentence-final occurrences (trailing period with nothing after, which BeautifulSoup produces when a period falls at an HTML tag boundary). Skip TOC entries (lines immediately preceded or followed by a bare page number or "Page N"). Take the first remaining candidate whose extracted text contains both a fiscal year and a dollar amount — the data check distinguishes the real section from any remaining footnote cross-references.

**Tier 2 — Broader "Executive Compensation" heading (~10% of filings)**
Some filers omit a standalone "Summary Compensation Table" heading entirely, or it lives inside a merged table cell that BeautifulSoup cannot extract. The broader pattern matches "Executive Compensation" section headers, which almost always precede the comp table. The first occurrence is tried before the last — the first is typically the actual section header, the last often appears in a say-on-pay advisory-vote narrative at the document end.

**Tier 3 — Column-header cluster scan (last resort)**
A handful of filers have no recognizable section header at all but follow the standard SEC column layout. Scanning for adjacent "Salary" and "Total" lines within 5 rows locates the table directly.

## Parser-only mode (no LLM)

If you only need the raw section text:

```python
import def14a

lines        = def14a.html_to_lines("primary.htm")
comp_section = def14a.extract_compensation_section(lines)
dir_section  = def14a.extract_director_section(lines)
```

## Known limitations

- **Wrapper/index files** — EDGAR sometimes stores a brief index file as the primary document. These have no compensation content and return `None`. Download the actual exhibit document instead.
- **Non-standard headings** — a small minority of filers label their comp section differently (e.g. "Named Executive Officer Compensation"). Tier 2 and Tier 3 catch most of these.
- **Multi-subsidiary holding companies** — holding companies that file for multiple subsidiaries may return data for only the parent entity.
- **Governance field coverage varies** — `board_size` is populated in ~34% of filings (the Q&A answer is often narrative rather than a bare number). `auditor` is cross-checked with a full-document regex scan as a fallback and populates in ~80% of filings.

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
