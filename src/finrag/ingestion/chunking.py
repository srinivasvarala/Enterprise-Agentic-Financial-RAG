"""Deliberately simple paragraph-based chunker.

A pure function over a ParsedDocument. The structure-aware chunker (overlap, sentence splits,
table row groups) replaces it in a later ticket; the ChunkDraft output shape is the contract.
Token counts are approximate: whitespace-separated words.
"""

from dataclasses import dataclass

from finrag.core.domain import ChunkKind
from finrag.ingestion.parsed import (
    BoundingBox,
    HeadingBlock,
    ParagraphBlock,
    ParsedDocument,
    TableBlock,
)

SECTION_SEPARATOR = " \u203a "  # " > " as a single-guillemet, e.g. "Part II > Item 7"


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    ordinal: int
    kind: ChunkKind
    text: str
    section_path: tuple[str, ...]
    page_start: int
    page_end: int
    token_count: int
    bounding_boxes: tuple[tuple[int, BoundingBox], ...] = ()  # (page, bbox)

    @property
    def index_text(self) -> str:
        """What the embedder and BM25 encoder see: section path, then chunk text."""
        if not self.section_path:
            return self.text
        return f"{SECTION_SEPARATOR.join(self.section_path)}\n{self.text}"


def approx_tokens(text: str) -> int:
    return len(text.split())


def chunk_document(parsed: ParsedDocument, max_tokens: int) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    stack: list[tuple[int, str]] = []  # (heading level, text)
    pending: list[tuple[str, int, BoundingBox | None]] = []  # (text, page, bbox)
    pending_tokens = 0

    def path() -> tuple[str, ...]:
        return tuple(text for _, text in stack)

    def flush() -> None:
        nonlocal pending, pending_tokens
        if not pending:
            return
        text = "\n\n".join(p[0] for p in pending)
        pages = [p[1] for p in pending]
        boxes = tuple((p[1], p[2]) for p in pending if p[2] is not None)
        chunks.append(
            ChunkDraft(
                ordinal=len(chunks),
                kind=ChunkKind.TEXT,
                text=text,
                section_path=path(),
                page_start=min(pages),
                page_end=max(pages),
                token_count=approx_tokens(text),
                bounding_boxes=boxes,
            )
        )
        pending, pending_tokens = [], 0

    for block in parsed.blocks:
        if isinstance(block, HeadingBlock):
            flush()  # a chunk never crosses a heading, so its section path stays truthful
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text))
        elif isinstance(block, ParagraphBlock):
            for piece in _split_words(block.text, max_tokens):
                tokens = approx_tokens(piece)
                if pending and pending_tokens + tokens > max_tokens:
                    flush()
                pending.append((piece, block.page, block.bbox))
                pending_tokens += tokens
        elif isinstance(block, TableBlock):
            flush()
            text = _table_text(block)
            chunks.append(
                ChunkDraft(
                    ordinal=len(chunks),
                    kind=ChunkKind.TABLE,
                    text=text,
                    section_path=path(),
                    page_start=block.page,
                    page_end=block.page,
                    token_count=approx_tokens(text),
                    bounding_boxes=((block.page, block.bbox),) if block.bbox else (),
                )
            )
    flush()
    return chunks


def _split_words(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


def _table_text(block: TableBlock) -> str:
    parts = [block.caption, block.header, block.markdown]
    return "\n".join(p for p in parts if p)
