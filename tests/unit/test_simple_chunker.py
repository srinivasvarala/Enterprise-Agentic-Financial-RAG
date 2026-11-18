"""The simple paragraph chunker and text parser as pure functions (no DB, no API)."""

from finrag.core.domain import ChunkKind
from finrag.ingestion.chunking import chunk_document
from finrag.ingestion.parsed import (
    BoundingBox,
    HeadingBlock,
    ParagraphBlock,
    ParsedDocument,
    TableBlock,
)
from finrag.ingestion.text_parser import TextParser


def doc(*blocks: HeadingBlock | ParagraphBlock | TableBlock) -> ParsedDocument:
    return ParsedDocument(blocks=blocks)


def test_paragraphs_are_packed_up_to_the_token_limit() -> None:
    parsed = doc(*(ParagraphBlock(text="w " * 40, page=1) for _ in range(5)))

    chunks = chunk_document(parsed, max_tokens=100)

    assert [c.token_count for c in chunks] == [80, 80, 40]
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_an_oversized_paragraph_is_split_on_words() -> None:
    chunks = chunk_document(doc(ParagraphBlock(text="w " * 250, page=1)), max_tokens=100)

    assert [c.token_count for c in chunks] == [100, 100, 50]


def test_section_path_follows_the_heading_stack_and_chunks_never_cross_headings() -> None:
    parsed = doc(
        HeadingBlock("Part II", level=1, page=1),
        HeadingBlock("Item 7. MD&A", level=2, page=1),
        ParagraphBlock("Liquidity is fine.", page=1),
        HeadingBlock("Item 8. Statements", level=2, page=2),
        ParagraphBlock("Audited.", page=2),
        HeadingBlock("Part III", level=1, page=3),
        ParagraphBlock("Directors.", page=3),
    )

    chunks = chunk_document(parsed, max_tokens=800)

    assert [(c.section_path, c.text) for c in chunks] == [
        (("Part II", "Item 7. MD&A"), "Liquidity is fine."),
        (("Part II", "Item 8. Statements"), "Audited."),
        (("Part III",), "Directors."),
    ]
    assert chunks[0].index_text == "Part II \u203a Item 7. MD&A\nLiquidity is fine."


def test_page_range_spans_the_blocks_in_a_chunk() -> None:
    parsed = doc(ParagraphBlock("a", page=3), ParagraphBlock("b", page=4))

    (chunk,) = chunk_document(parsed, max_tokens=800)

    assert (chunk.page_start, chunk.page_end) == (3, 4)


def test_a_table_becomes_its_own_table_chunk_with_caption_and_header() -> None:
    box = BoundingBox(0, 0, 10, 10)
    parsed = doc(
        HeadingBlock("Results", level=1, page=1),
        ParagraphBlock("Before.", page=1),
        TableBlock(
            markdown="| Revenue | 10 |",
            header="| Item | USD |\n| --- | --- |",
            caption="Table 1",
            page=2,
            bbox=box,
        ),
        ParagraphBlock("After.", page=2),
    )

    chunks = chunk_document(parsed, max_tokens=800)

    assert [c.kind for c in chunks] == [ChunkKind.TEXT, ChunkKind.TABLE, ChunkKind.TEXT]
    table = chunks[1]
    assert table.text == "Table 1\n| Item | USD |\n| --- | --- |\n| Revenue | 10 |"
    assert table.section_path == ("Results",)
    assert (table.page_start, table.page_end) == (2, 2)
    assert table.bounding_boxes == ((2, box),)


def test_text_parser_splits_paragraphs_on_blank_lines() -> None:
    parsed = TextParser().parse(b"\xef\xbb\xbfOne\r\nstill one\r\n\r\n\r\nTwo\n")

    assert [(type(b).__name__, b.page) for b in parsed.blocks] == [("ParagraphBlock", 1)] * 2
    assert [getattr(b, "text", None) for b in parsed.blocks] == ["One\nstill one", "Two"]
