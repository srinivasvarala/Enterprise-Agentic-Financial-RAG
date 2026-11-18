"""Protocols for every external provider. Concrete adapters live in their own module."""

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from finrag.core.domain import IngestionJob, ParsedDocument, SparseVector, VectorPoint


class BlobStore(Protocol):
    """Stores original documents unchanged."""

    def put(self, key: str, data: bytes, content_type: str) -> None: ...

    def get(self, key: str) -> bytes: ...


class JobQueue(Protocol):
    """Enqueues background work. Execution happens in a separate worker process."""

    def enqueue_ingestion(self, job: IngestionJob) -> None: ...


class Parser(Protocol):
    """Turns one document format into the parser-independent ParsedDocument.

    Raises finrag.core.errors.ParseError when the document cannot be understood.
    """

    def parse(self, data: bytes) -> ParsedDocument: ...


class Embedder(Protocol):
    """Dense document embeddings. Raises TransientError for retryable provider failures."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...


class SparseEncoder(Protocol):
    """BM25 sparse vectors for documents."""

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...


class VectorStore(Protocol):
    """Hybrid (dense + bm25) index. Every read and write requires a tenant id.

    Implementations must raise TenantRequiredError when tenant_id is None/missing and must
    never touch points of another tenant.
    """

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection and its indexes if missing. Idempotent."""

    def upsert(self, tenant_id: UUID, points: Sequence[VectorPoint]) -> None:
        """Insert or replace points by id. Every point's tenant_id must equal tenant_id."""

    def delete_version(self, tenant_id: UUID, document_version_id: UUID) -> None:
        """Remove all points of one Document Version."""

    def get_version_points(self, tenant_id: UUID, document_version_id: UUID) -> list[VectorPoint]:
        """All points of one Document Version, in no particular order."""
