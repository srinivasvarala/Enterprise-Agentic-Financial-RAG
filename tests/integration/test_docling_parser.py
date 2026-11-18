"""Thin adapter tests: the Docling parser against the committed fixtures (real Docling models).

Needs no docker services, but loads Docling's layout/table models (downloaded to the Hugging
Face cache on first use), so it lives under the `integration` marker: `make test-int`.
"""

from functools import cache
from pathlib import Path

import pytest

from finrag.core.domain import HeadingBlock, ParagraphBlock, ParsedDocument, TableBlock
from finrag.core.errors import ParseError
from finrag.ingestion.docling_parser import DoclingParser

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent.parent / "fixtures"


@cache
def parsed_fixture() -> ParsedDocument:
    return DoclingParser().parse((FIXTURES / "example_10k_extract.pdf").read_bytes())


def headings() -> list[HeadingBlock]:
    return [b for b in parsed_fixture().blocks if isinstance(b, HeadingBlock)]


def test_part_and_item_headings_get_nested_levels() -> None:
    levels = {h.text: h.level for h in headings()}

    assert levels["PART I"] == 1
    assert levels["Item 1A. Risk Factors"] == 2
    assert levels["Item 7. Management's Discussion and Analysis"] == 2
    assert levels["Risks Related to Our Business"] == 3  # sub-heading under an Item


def test_headings_carry_pages_and_bounding_boxes() -> None:
    by_text = {h.text: h for h in headings()}

    assert by_text["PART I"].page == 1
    assert by_text["PART II"].page == 2
    assert by_text["Item 8. Financial Statements and Supplementary Data"].page == 3
    box = by_text["PART I"].bbox
    assert box is not None
    assert box.x0 < box.x1
    assert box.y0 < box.y1


def test_tables_have_header_body_caption_and_page() -> None:
    tables = [b for b in parsed_fixture().blocks if isinstance(b, TableBlock)]

    assert [t.page for t in tables] == [2, 3]
    first = tables[0]
    assert first.header is not None
    assert "Segment" in first.header
    assert set(first.header.splitlines()[1]) <= set("|-: ")  # the separator line
    body = first.markdown.splitlines()
    assert len(body) == 4
    assert "Total net sales" in body[-1]
    assert "27,600" in body[-1]
    assert first.caption == "Table 1: Net sales by segment (in millions of dollars)"
    assert first.bbox is not None


def test_running_footer_and_page_numbers_are_dropped() -> None:
    text = "\n".join(b.text for b in parsed_fixture().blocks if isinstance(b, ParagraphBlock))

    assert "Form 10-K" not in text
    assert "Page 2" not in text
    assert "Our results of operations" in text


def test_table_caption_is_not_also_emitted_as_a_paragraph() -> None:
    paragraphs = [b.text for b in parsed_fixture().blocks if isinstance(b, ParagraphBlock)]

    assert not any(p.startswith("Table 1:") for p in paragraphs)


def test_corrupt_pdf_raises_parse_error_with_a_readable_message() -> None:
    with pytest.raises(ParseError, match="PDF"):
        DoclingParser().parse((FIXTURES / "corrupt.pdf").read_bytes())
