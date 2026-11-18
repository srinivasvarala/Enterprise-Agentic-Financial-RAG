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


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One fused search result: a Chunk id and its fusion score (higher is better)."""

    chunk_id: UUID
    score: float


# -- Answering: generator output, retrieved Evidence and the validated Answer (ADR-0008) ----------


class RefusalReason(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNVERIFIABLE = "unverifiable"
    OUT_OF_POLICY = "out_of_policy"


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    """A Chunk retrieved for a question, with what a Citation must resolve to."""

    chunk_id: UUID
    document_id: UUID
    text: str
    section_path: tuple[str, ...]
    page_start: int
    page_end: int


@dataclass(frozen=True, slots=True)
class DraftCitation:
    """What the generator claims: Citation Marker `marker` is supported by `quoted_span`."""

    marker: int
    chunk_id: UUID
    quoted_span: str


@dataclass(frozen=True, slots=True)
class DraftAnswer:
    """The generator's structured output, not yet verified."""

    text: str
    citations: tuple[DraftCitation, ...]


@dataclass(frozen=True, slots=True)
class Citation:
    """A verified link from a Citation Marker to the Chunk's document, pages and section."""

    marker: int
    chunk_id: UUID
    document_id: UUID
    section_path: tuple[str, ...]
    page_start: int
    page_end: int
    quoted_span: str


@dataclass(frozen=True, slots=True)
class Answer:
    """A validated Answer, or a Refusal when `refusal_reason` is set."""

    text: str
    citations: tuple[Citation, ...]
    refusal_reason: RefusalReason | None = None
