"""Protocols for every external provider. Concrete adapters live in their own module."""

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from finrag.core.domain import (
    DraftAnswer,
    EvidenceChunk,
    IngestionJob,
    ParsedDocument,
    Principal,
    SearchHit,
    SparseVector,
    VectorPoint,
)


class BlobStore(Protocol):
    """Stores original documents unchanged."""

    def put(self, key: str, data: bytes, content_type: str) -> None: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None:
        """Remove one object. Absent keys are fine."""


class JobQueue(Protocol):
    """Enqueues background work. Execution happens in a separate worker process."""

    def enqueue_ingestion(self, job: IngestionJob) -> bool:
        """Enqueue the job; False if one for the same Document Version is already waiting."""
        ...


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

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed a question; providers treat queries differently from documents."""
        ...


class SparseEncoder(Protocol):
    """BM25 sparse vectors for documents."""

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def encode_query(self, text: str) -> SparseVector: ...


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

    def search(
        self,
        tenant_id: UUID,
        collection_id: UUID,
        dense: tuple[float, ...],
        sparse: SparseVector,
        *,
        dense_k: int,
        sparse_k: int,
        rrf_k: int,
        limit: int,
    ) -> list[SearchHit]:
        """Dense and BM25 candidates fused by reciprocal rank fusion, best first.

        Only points of this Tenant and Collection are searched.
        """
        ...


class ChunkRepository(Protocol):
    """Loads Chunks from the system of record. Ids that no longer exist are simply absent."""

    def get_many(
        self, tenant_id: UUID, collection_id: UUID, chunk_ids: Sequence[UUID]
    ) -> list[EvidenceChunk]: ...


class Reranker(Protocol):
    """Scores each chunk 0 to 1 for how well it answers the question, in input order."""

    def score(self, question: str, chunks: Sequence[EvidenceChunk]) -> list[float]: ...


class EvidenceRetriever(Protocol):
    """Finds the Evidence for a question within the Principal's Collection."""

    def retrieve(self, principal: Principal, question: str) -> tuple[EvidenceChunk, ...]: ...


class AnswerGenerator(Protocol):
    """Drafts an answer from the Evidence. The draft is unverified.

    Raises finrag.core.errors.ModelOutputError when the model's reply is not usable and
    TransientError for retryable provider failures.
    """

    def generate(self, question: str, evidence: Sequence[EvidenceChunk]) -> DraftAnswer: ...
