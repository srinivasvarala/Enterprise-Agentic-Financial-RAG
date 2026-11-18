"""Domain types shared across modules."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class ChunkKind(StrEnum):
    TEXT = "text"
    TABLE = "table"


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is calling: the Tenant and Collection every request is scoped to.

    M1 resolves every request to the seeded defaults; M6 swaps in API-key auth.
    """

    tenant_id: UUID
    collection_id: UUID


@dataclass(frozen=True, slots=True)
class IngestionJob:
    tenant_id: UUID
    document_id: UUID
    document_version_id: UUID


@dataclass(frozen=True, slots=True)
class SparseVector:
    """BM25 term weights: parallel arrays of vocabulary indices and weights."""

    indices: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """One indexed Chunk. `id` is deterministic: uuid5(document version id, ordinal)."""

    id: UUID
    dense: tuple[float, ...]
    sparse: SparseVector
    tenant_id: UUID
    collection_id: UUID
    document_id: UUID
    document_version_id: UUID
    chunk_id: UUID
    kind: ChunkKind
    page_start: int
    page_end: int


# -- Parsed document: parser output, chunker input -----------------------------------------------
# Parsers own all format knowledge; the chunker never sees a file. Page numbers are 1-based.
# A bounding box is in the parser's page coordinate space.


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
