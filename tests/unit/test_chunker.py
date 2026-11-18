"""The structure-aware chunker and text parser as pure functions over parsed documents."""

import itertools

import pytest

from finrag.core.domain import ChunkKind
from finrag.ingestion.chunking import ChunkingConfig, chunk_document, count_tokens
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


def cfg(target: int = 50, max_: int = 80, overlap: int = 0, table_max: int = 60) -> ChunkingConfig:
    return ChunkingConfig(
        target_tokens=target, max_tokens=max_, overlap_tokens=overlap, table_max_tokens=table_max
    )


def sentences(n: int, words: int = 10, tag: str = "s") -> str:
    """n sentences of `words` words each; every sentence is unique and ends with a period."""
    t = tag.upper()  # sentences start with a capital, as the sentence splitter expects
    return " ".join(
        " ".join(f"{t}{i}w{k}" for k in range(words - 1)) + f" {t}{i}end." for i in range(n)
    )


def test_paragraphs_pack_to_the_target_and_close_on_a_paragraph_boundary() -> None:
    parsed = doc(*(ParagraphBlock(text=sentences(2, tag=f"p{i}"), page=1) for i in range(5)))

    chunks = chunk_document(parsed, cfg(target=50, max_=80))

    assert [c.token_count for c in chunks] == [60, 40]  # 3 paragraphs reach the target, then 2
    assert [c.ordinal for c in chunks] == [0, 1]
    assert chunks[0].text.count("\n\n") == 2


def test_a_chunk_never_exceeds_the_max() -> None:
    parsed = doc(ParagraphBlock(text=sentences(30), page=1))

    chunks = chunk_document(parsed, cfg(target=50, max_=80))

    assert all(c.token_count <= 80 for c in chunks)
    assert sum(c.token_count for c in chunks) == 300


def test_oversized_paragraphs_split_on_sentence_boundaries() -> None:
    parsed = doc(ParagraphBlock(text=sentences(30), page=1))

    chunks = chunk_document(parsed, cfg(target=50, max_=80))

    assert len(chunks) > 1
    assert all(c.text.endswith("end.") for c in chunks)


def test_a_single_oversized_sentence_is_split_on_words() -> None:
    parsed = doc(ParagraphBlock(text="w " * 250, page=1))

    chunks = chunk_document(parsed, cfg(target=50, max_=100))

    assert [c.token_count for c in chunks] == [100, 100, 50]


def test_overlap_repeats_trailing_sentences_within_a_section() -> None:
    parsed = doc(HeadingBlock("Risk", level=1, page=1), ParagraphBlock(text=sentences(20), page=1))

    chunks = chunk_document(parsed, cfg(target=50, max_=60, overlap=20))

    assert len(chunks) >= 3
    for prev, nxt in itertools.pairwise(chunks):
        tail = prev.text.split(". ")[-2:]  # last two sentences of the previous chunk
        assert nxt.text.startswith(tail[0])
    assert all(c.token_count <= 60 for c in chunks)


def test_no_overlap_across_headings_or_tables() -> None:
    parsed = doc(
        HeadingBlock("A", level=1, page=1),
        ParagraphBlock(text=sentences(8, tag="a"), page=1),
        HeadingBlock("B", level=1, page=1),
        ParagraphBlock(text=sentences(8, tag="b"), page=1),
    )

    chunks = chunk_document(parsed, cfg(target=50, max_=60, overlap=20))

    b_chunks = [c for c in chunks if c.section_path == ("B",)]
    assert b_chunks
    assert not any("A0w0" in c.text for c in b_chunks)
    assert b_chunks[0].text.startswith("B0w0")


def test_overlap_zero_means_chunks_are_disjoint() -> None:
    parsed = doc(ParagraphBlock(text=sentences(20), page=1))

    chunks = chunk_document(parsed, cfg(target=50, max_=60, overlap=0))

    assert sum(c.token_count for c in chunks) == 200


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

    chunks = chunk_document(parsed, cfg())

    assert [(c.section_path, c.text) for c in chunks] == [
        (("Part II", "Item 7. MD&A"), "Liquidity is fine."),
        (("Part II", "Item 8. Statements"), "Audited."),
        (("Part III",), "Directors."),
    ]
    assert chunks[0].index_text == "Part II \u203a Item 7. MD&A\nLiquidity is fine."


def test_a_new_top_level_section_starts_a_new_chunk_even_when_the_last_is_small() -> None:
    parsed = doc(
        HeadingBlock("One", level=1, page=1),
        ParagraphBlock("tiny", page=1),
        HeadingBlock("Two", level=1, page=1),
        ParagraphBlock("also tiny", page=1),
    )

    chunks = chunk_document(parsed, cfg())

    assert [c.section_path for c in chunks] == [("One",), ("Two",)]


def test_skipped_heading_levels_and_text_before_any_heading() -> None:
    parsed = doc(
        ParagraphBlock("Preamble.", page=1),
        HeadingBlock("Top", level=1, page=1),
        HeadingBlock("Deep", level=3, page=1),
        ParagraphBlock("Body.", page=1),
        HeadingBlock("Mid", level=2, page=1),
        ParagraphBlock("More.", page=1),
    )

    chunks = chunk_document(parsed, cfg())

    assert [c.section_path for c in chunks] == [(), ("Top", "Deep"), ("Top", "Mid")]
    assert chunks[0].index_text == "Preamble."


def test_page_range_spans_the_blocks_in_a_chunk() -> None:
    parsed = doc(ParagraphBlock("a", page=3), ParagraphBlock("b", page=4))

    (chunk,) = chunk_document(parsed, cfg())

    assert (chunk.page_start, chunk.page_end) == (3, 4)


def test_split_paragraph_chunks_carry_their_own_page_range() -> None:
    parsed = doc(
        ParagraphBlock(sentences(6, tag="a"), page=1),
        ParagraphBlock(sentences(6, tag="b"), page=2),
    )

    chunks = chunk_document(parsed, cfg(target=50, max_=60))

    assert (chunks[0].page_start, chunks[0].page_end) == (1, 1)
    assert (chunks[-1].page_start, chunks[-1].page_end) == (2, 2)


def test_text_chunks_collect_bounding_boxes_once_per_block() -> None:
    box = BoundingBox(0, 0, 5, 5)
    parsed = doc(ParagraphBlock(sentences(12), page=1, bbox=box))

    chunks = chunk_document(parsed, cfg(target=50, max_=60))

    assert len(chunks) > 1
    assert all(c.bounding_boxes == ((1, box),) for c in chunks)


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

    chunks = chunk_document(parsed, cfg())

    assert [c.kind for c in chunks] == [ChunkKind.TEXT, ChunkKind.TABLE, ChunkKind.TEXT]
    table = chunks[1]
    assert table.text == "Table 1\n| Item | USD |\n| --- | --- |\n| Revenue | 10 |"
    assert table.section_path == ("Results",)
    assert (table.page_start, table.page_end) == (2, 2)
    assert table.bounding_boxes == ((2, box),)
    assert table.index_text.startswith("Results\nTable 1")


def test_an_oversized_table_splits_into_row_groups_repeating_caption_and_header() -> None:
    header = "| Item | USD |\n| --- | --- |"
    rows = [f"| Line {i} | {i * 100} |" for i in range(40)]
    parsed = doc(
        TableBlock(markdown="\n".join(rows), header=header, caption="Table 2", page=5),
    )

    chunks = chunk_document(parsed, cfg(table_max=60))

    assert len(chunks) > 1
    assert all(c.kind is ChunkKind.TABLE for c in chunks)
    for c in chunks:
        assert c.text.startswith(f"Table 2\n{header}\n")
        assert c.token_count <= 60
        assert (c.page_start, c.page_end) == (5, 5)
    body_rows = [line for c in chunks for line in c.text.splitlines()[3:]]
    assert body_rows == rows  # every row exactly once, in order


def test_a_table_row_is_never_split_even_if_it_alone_exceeds_the_limit() -> None:
    long_row = "| " + " ".join(f"c{i}" for i in range(100)) + " |"
    parsed = doc(TableBlock(markdown=f"{long_row}\n| a | b |", header="| H |\n| --- |", page=1))

    chunks = chunk_document(parsed, cfg(table_max=30))

    assert [c.text.splitlines()[-1] for c in chunks] == [long_row, "| a | b |"]


def test_count_tokens_is_whitespace_words() -> None:
    assert count_tokens("a  b\nc") == 3


def test_config_rejects_inconsistent_limits() -> None:
    with pytest.raises(ValueError, match="overlap"):
        cfg(target=50, max_=60, overlap=60)
    with pytest.raises(ValueError, match="target"):
        cfg(target=90, max_=80)


def test_text_parser_splits_paragraphs_on_blank_lines() -> None:
    parsed = TextParser().parse(b"\xef\xbb\xbfOne\r\nstill one\r\n\r\n\r\nTwo\n")

    assert [(type(b).__name__, b.page) for b in parsed.blocks] == [("ParagraphBlock", 1)] * 2
    assert [getattr(b, "text", None) for b in parsed.blocks] == ["One\nstill one", "Two"]
