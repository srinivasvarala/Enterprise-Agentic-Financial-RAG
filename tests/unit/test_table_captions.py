"""Table caption promotion: layout models disagree across platforms about whether
"Table 1: ..." is a caption, a heading or a paragraph; the parser normalises it."""

from finrag.core.domain import Block, HeadingBlock, ParagraphBlock, TableBlock
from finrag.ingestion.docling_parser import promote_table_captions

CAPTION = "Table 1: Net sales by segment (in millions of dollars)"


def _table(page: int = 2, caption: str | None = None) -> TableBlock:
    return TableBlock(
        markdown="| Americas | 12,450 |",
        page=page,
        header="| Segment | 2024 |\n|---|---|",
        caption=caption,
    )


def _promote(*blocks: Block) -> tuple[Block, ...]:
    return promote_table_captions(list(blocks))


def test_heading_before_table_becomes_its_caption_and_leaves_the_section_path() -> None:
    result = _promote(
        HeadingBlock(text="Results of Operations", level=3, page=2),
        HeadingBlock(text=CAPTION, level=3, page=2),
        _table(),
    )

    assert [type(b).__name__ for b in result] == ["HeadingBlock", "TableBlock"]
    assert result[0].text == "Results of Operations"  # type: ignore[union-attr]
    assert result[1].caption == CAPTION  # type: ignore[union-attr]


def test_paragraph_before_table_becomes_its_caption() -> None:
    result = _promote(ParagraphBlock(text=CAPTION, page=2), _table())

    assert len(result) == 1
    assert result[0].caption == CAPTION  # type: ignore[union-attr]


def test_existing_caption_is_kept_and_the_preceding_block_is_untouched() -> None:
    result = _promote(ParagraphBlock(text=CAPTION, page=2), _table(caption="Docling caption"))

    assert len(result) == 2
    assert result[1].caption == "Docling caption"  # type: ignore[union-attr]


def test_caption_on_a_different_page_is_not_attached() -> None:
    result = _promote(HeadingBlock(text=CAPTION, level=3, page=1), _table(page=2))

    assert len(result) == 2
    assert result[1].caption is None  # type: ignore[union-attr]


def test_ordinary_text_before_a_table_is_not_taken_as_a_caption() -> None:
    result = _promote(ParagraphBlock(text="Net sales rose in every segment.", page=2), _table())

    assert len(result) == 2
    assert result[1].caption is None  # type: ignore[union-attr]
