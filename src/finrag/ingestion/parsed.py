"""The parsed-document shape: parser output, chunker input.

Parsers own all format knowledge; the chunker never sees a file. This module is pure data and
imports nothing from the rest of finrag, so any parser (plain text, Docling, ...) can emit it.
Page numbers are 1-based. A bounding box is in the parser's page coordinate space.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, slots=True)
class HeadingBlock:
    text: str
    level: int  # 1 = top-level section
    page: int
    bbox: BoundingBox | None = None


@dataclass(frozen=True, slots=True)
class ParagraphBlock:
    text: str
    page: int
    bbox: BoundingBox | None = None


@dataclass(frozen=True, slots=True)
class TableBlock:
    markdown: str  # body rows only, Markdown
    page: int
    header: str | None = None  # Markdown header row(s) incl. separator, kept separately
    caption: str | None = None
    bbox: BoundingBox | None = None


Block = HeadingBlock | ParagraphBlock | TableBlock


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Blocks in reading order."""

    blocks: tuple[Block, ...]
