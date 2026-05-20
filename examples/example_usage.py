"""
Basic usage examples for def14a-parser.

Assumes Ollama is running locally with at least one compatible model pulled:
    ollama pull llama3.2:3b   # fast, 2 GB
    ollama pull qwen2.5:7b    # more accurate, 4.7 GB

To get a DEF 14A filing to test with, download one from EDGAR:
    curl -A "your-name your@email.com" \
        "https://www.sec.gov/Archives/edgar/data/37808/000003780825000016/0000037808-25-000016-index.htm" \
        -o filing_index.htm
    # Then download the primary document linked from that index.
"""
from pathlib import Path
import json
import def14a


# ── Single file ────────────────────────────────────────────────────────────

def example_single_file():
    path = Path("primary.htm")  # replace with your DEF 14A file path

    result = def14a.extract(path)

    if result is None:
        print("No compensation data found (file may be a wrapper or have no comp table)")
        return

    print(json.dumps(result, indent=2))


# ── With a specific model ──────────────────────────────────────────────────

def example_with_model():
    path = Path("primary.htm")

    # Use a larger model for better accuracy
    result = def14a.extract(path, model="qwen2.5:7b")

    if result:
        ceo = result["ceo"]
        print(f"CEO: {ceo['name']} ({ceo['title']})")
        print(f"Fiscal year: {result['fiscal_year']}")

        gov = result["governance"]
        print(f"Board size: {gov['board_size']}, "
              f"pct independent: {gov['pct_independent']:.0%}")
        print(f"Auditor: {gov['auditor']}")
        print(f"Say-on-pay approval: {gov['say_on_pay_pct']:.0%}")

        print("\nNamed executive officers:")
        for row in result["compensation"]:
            tag = " ← CEO" if row["is_ceo"] else ""
            print(f"  {row['name']}: salary=${row['salary']:,}, "
                  f"total=${row['total_comp']:,}{tag}")


# ── Using the parser layer directly (no LLM) ──────────────────────────────

def example_parser_only():
    """Extract just the compensation table text without calling any LLM."""
    path = Path("primary.htm")

    lines = def14a.html_to_lines(path)
    comp_section = def14a.extract_compensation_section(lines)
    dir_section  = def14a.extract_director_section(lines)

    print(f"Compensation section: {len(comp_section)} chars")
    print(comp_section[:500])

    print(f"\nDirector bio section: {len(dir_section)} chars")
    print(dir_section[:500])


if __name__ == "__main__":
    example_with_model()
