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
