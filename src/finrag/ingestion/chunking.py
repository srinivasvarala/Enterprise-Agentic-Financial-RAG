"""Structure-aware chunker: a pure function from a ParsedDocument to ChunkDrafts.

Rules:
- Section path comes from the heading stack. Every heading closes the current chunk, so a chunk
  never crosses a section boundary and its path is always truthful; a new top-level section
  therefore always starts a new chunk.
- Text chunks fill towards a soft target and never exceed the hard max. They close on a
  paragraph boundary once the target is reached; otherwise they split on sentence boundaries
  (and, for a single oversized sentence, on words).
- Consecutive text chunks of the same section share a small sentence-aligned overlap. Overlap
  never crosses a heading or a table.
- A table is its own Table Chunk: caption, header, body rows. An oversized table splits into row
  groups that each repeat the caption and header; a row is never split.
Token counts are approximate (see `count_tokens`), the single place a real tokenizer plugs in.
"""

import itertools
import re
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


def count_tokens(text: str) -> int:
    """Approximate token count (whitespace words). Swap in a real tokenizer here only."""
    return len(text.split())


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 500  # soft: close a text chunk at the next paragraph boundary
    max_tokens: int = 800  # hard ceiling for a text chunk
    overlap_tokens: int = 50  # trailing sentences repeated into the next chunk of a section
    table_max_tokens: int = 800  # above this a table splits into row groups

    def __post_init__(self) -> None:
        if not 0 < self.target_tokens <= self.max_tokens:
            raise ValueError("target_tokens must be positive and at most max_tokens")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens must be non-negative and below max_tokens")
        if self.table_max_tokens <= 0:
            raise ValueError("table_max_tokens must be positive")


_SENTENCE_BREAK = re.compile(r'(?<=[.!?])\s+(?=["(\[A-Z0-9])')


@dataclass(frozen=True, slots=True)
class _Unit:
    """A sentence (or word run) of a paragraph, the atom text chunks are packed from."""

    text: str
    tokens: int
    page: int
    bbox: BoundingBox | None
    paragraph: int


def chunk_document(parsed: ParsedDocument, config: ChunkingConfig) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    stack: list[tuple[int, str]] = []  # (heading level, text)
    units: list[_Unit] = []
    paragraph_no = 0

    def path() -> tuple[str, ...]:
        return tuple(text for _, text in stack)

    def flush_text() -> None:
        nonlocal units
        _pack_text(units, config, path(), chunks)
        units = []

    for block in parsed.blocks:
        if isinstance(block, HeadingBlock):
            flush_text()
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text))
        elif isinstance(block, ParagraphBlock):
            paragraph_no += 1
            units.extend(_units(block, paragraph_no, config.max_tokens))
        elif isinstance(block, TableBlock):
            flush_text()
            for text in _table_texts(block, config.table_max_tokens):
                chunks.append(
                    ChunkDraft(
                        ordinal=len(chunks),
                        kind=ChunkKind.TABLE,
                        text=text,
                        section_path=path(),
                        page_start=block.page,
                        page_end=block.page,
                        token_count=count_tokens(text),
                        bounding_boxes=((block.page, block.bbox),) if block.bbox else (),
                    )
                )
    flush_text()
    return chunks


def _units(block: ParagraphBlock, paragraph: int, max_tokens: int) -> list[_Unit]:
    out: list[_Unit] = []
    for sentence in _SENTENCE_BREAK.split(block.text.strip()):
        words = sentence.split()
        if not words:
            continue
        if len(words) <= max_tokens:
            pieces = [sentence.strip()]
        else:
            pieces = [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]
        out.extend(
            _Unit(piece, count_tokens(piece), block.page, block.bbox, paragraph) for piece in pieces
        )
    return out


def _pack_text(
    units: list[_Unit], config: ChunkingConfig, section: tuple[str, ...], out: list[ChunkDraft]
) -> None:
    """Greedily pack one section's units into chunks, appending ChunkDrafts to `out`."""
    current: list[_Unit] = []
    fresh = 0  # units in `current` not carried over as overlap
    total = 0

    def emit() -> None:
        text = _join(current)
        pages = [u.page for u in current]
        boxes = dict.fromkeys((u.page, u.bbox) for u in current if u.bbox is not None)
        out.append(
            ChunkDraft(
                ordinal=len(out),
                kind=ChunkKind.TEXT,
                text=text,
                section_path=section,
                page_start=min(pages),
                page_end=max(pages),
                token_count=count_tokens(text),
                bounding_boxes=tuple(boxes),
            )
        )

    for unit in units:
        if fresh:
            over_max = total + unit.tokens > config.max_tokens
            at_target = total >= config.target_tokens and unit.paragraph != current[-1].paragraph
            if over_max or at_target:
                emit()
                current = _overlap_tail(current, config.overlap_tokens)
                total = sum(u.tokens for u in current)
                fresh = 0
                if total + unit.tokens > config.max_tokens:
                    current, total = [], 0
        current.append(unit)
        total += unit.tokens
        fresh += 1
    if fresh:
        emit()


def _overlap_tail(chunk: list[_Unit], overlap_tokens: int) -> list[_Unit]:
    """Trailing whole sentences worth at most `overlap_tokens`, never the entire chunk."""
    tail: list[_Unit] = []
    tokens = 0
    for unit in reversed(chunk[1:]):
        if tokens + unit.tokens > overlap_tokens:
            break
        tail.append(unit)
        tokens += unit.tokens
    tail.reverse()
    return tail


def _join(units: list[_Unit]) -> str:
    parts = [units[0].text]
    for prev, unit in itertools.pairwise(units):
        parts.append("\n\n" if unit.paragraph != prev.paragraph else " ")
        parts.append(unit.text)
    return "".join(parts)


def _table_texts(block: TableBlock, max_tokens: int) -> list[str]:
    prefix = "\n".join(p for p in (block.caption, block.header) if p)
    rows = [row for row in block.markdown.splitlines() if row.strip()]

    def render(body: list[str]) -> str:
        return "\n".join(p for p in (prefix, "\n".join(body)) if p)

    whole = render(rows)
    if count_tokens(whole) <= max_tokens or len(rows) <= 1:
        return [whole]
    groups: list[list[str]] = []
    group: list[str] = []
    used = count_tokens(prefix)
    for row in rows:
        tokens = count_tokens(row)
        if group and used + tokens > max_tokens:
            groups.append(group)
            group, used = [], count_tokens(prefix)
        group.append(row)
        used += tokens
    groups.append(group)
    return [render(g) for g in groups]
