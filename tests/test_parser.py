"""
Unit tests for the rule-based parser (no LLM required).
"""
import pytest
from def14a.parser import extract_compensation_section, extract_director_section


def _lines(text: str) -> list[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]


# ── extract_compensation_section ───────────────────────────────────────────

class TestExtractCompSection:

    def test_plain_table(self):
        lines = _lines("""
            Some preamble text.
            Summary Compensation Table
            Name and Principal Position
            Year
            Salary
            John Smith | CEO | yes | 2023 | 500000 | 1200000
            Jane Doe | CFO | no | 2023 | 400000 | 800000
        """)
        result = extract_compensation_section(lines)
        assert "Summary Compensation Table" in result
        assert "John Smith" in result

    def test_skips_toc_entry(self):
        """TOC entry (followed by bare page number) should be skipped in favor of
        the actual section heading that appears later."""
        lines = _lines("""
            Table of Contents
            Summary Compensation Table
            42
            Other content here.
            Summary Compensation Table
            Name and Principal Position
            Year
            Salary
            John Smith | CEO | yes | 2023 | 500000 | 1200000
        """)
        result = extract_compensation_section(lines)
        # Should find the second occurrence (actual table), not the TOC entry
        assert "John Smith" in result

    def test_no_summary_falls_back_to_exec_comp(self):
        """When there is no 'Summary Compensation Table' heading, falls back to
        'Executive Compensation' section heading."""
        lines = _lines("""
            EXECUTIVE COMPENSATION
            The following table shows compensation for our named executive officers.
            Name | Year | Salary
            John Smith | 2023 | 500000
            Grants of Plan-Based Awards
            More content here.
        """)
        result = extract_compensation_section(lines)
        assert "EXECUTIVE COMPENSATION" in result
        assert "John Smith" in result

    def test_stops_at_end_pattern(self):
        lines = _lines("""
            Summary Compensation Table
            John Smith | CEO | yes | 2023 | 500000 | 1200000
            Grants of Plan-Based Awards
            This content should not appear.
        """)
        result = extract_compensation_section(lines)
        assert "This content should not appear" not in result

    def test_empty_file(self):
        assert extract_compensation_section([]) == ""


# ── _parse_comp_table (via extractor module) ───────────────────────────────

class TestParseCompTable:

    def test_basic_flat_row(self):
        from def14a.extractor import _parse_comp_table
        raw = "Jane Doe | Chief Executive Officer | yes | 2023 | 850000 | 3200000"
        rows = _parse_comp_table(raw)
        assert len(rows) == 1
        assert rows[0]["name"] == "Jane Doe"
        assert rows[0]["is_ceo"] is True
        assert rows[0]["year"] == 2023
        assert rows[0]["salary"] == 850000
        assert rows[0]["total_comp"] == 3200000

    def test_salary_exceeds_total_nulled(self):
        from def14a.extractor import _parse_comp_table
        # When salary > total, total is a sub-component — set to None
        raw = "John Smith | CFO | no | 2023 | 500000 | 200000"
        rows = _parse_comp_table(raw)
        assert rows[0]["total_comp"] is None

    def test_ceo_flag_from_title(self):
        from def14a.extractor import _parse_comp_table
        raw = "Jane Doe | Chief Executive Officer | no | 2023 | 850000 | 3200000"
        rows = _parse_comp_table(raw)
        # Title contains "chief executive" → is_ceo=True even if flag says "no"
        assert rows[0]["is_ceo"] is True

    def test_yes_no_not_treated_as_name(self):
        from def14a.extractor import _parse_comp_table
        raw = "yes | Chief Executive Officer | yes | 2023 | 850000 | 3200000"
        rows = _parse_comp_table(raw)
        # "yes" as first column should be rejected (no current_name to fall back to)
        assert len(rows) == 0

    def test_multiple_years(self):
        from def14a.extractor import _parse_comp_table
        raw = (
            "Jane Doe | CEO | yes | 2023 | 850000 | 3200000\n"
            "Jane Doe | CEO | yes | 2022 | 800000 | 2900000\n"
            "John Smith | CFO | no | 2023 | 500000 | 1100000\n"
        )
        rows = _parse_comp_table(raw)
        assert len(rows) == 3
        ceo_rows = [r for r in rows if r["is_ceo"]]
        assert len(ceo_rows) == 2

    def test_dollar_sign_and_commas(self):
        from def14a.extractor import _parse_comp_table
        raw = "Jane Doe | CEO | yes | 2023 | $850,000 | $3,200,000"
        rows = _parse_comp_table(raw)
        assert rows[0]["salary"] == 850000
        assert rows[0]["total_comp"] == 3200000
