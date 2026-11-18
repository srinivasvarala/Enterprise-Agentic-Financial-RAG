"""In-memory fakes for the Protocols in finrag.core.interfaces."""

from collections.abc import Callable, Sequence
from uuid import UUID

from finrag.core.domain import EvidenceChunk, IngestionJob, VectorPoint
from finrag.indexing.in_memory import HashEmbedder, InMemoryVectorStore


class InMemoryBlobStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = data

    def get(self, key: str) -> bytes:
        return self.objects[key]


class InMemoryJobQueue:
    def __init__(self) -> None:
        self.jobs: list[IngestionJob] = []

    def enqueue_ingestion(self, job: IngestionJob) -> bool:
        self.jobs.append(job)
        return True


class FlakyEmbedder(HashEmbedder):
    """Raises `error` for the first `failures` calls, then behaves normally."""

    def __init__(self, failures: int = 0, error: Exception | None = None) -> None:
        super().__init__()
        self.failures = failures
        self.error = error or RuntimeError("embedder failure")
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return super().embed_documents(texts)


class SpyVectorStore(InMemoryVectorStore):
    """Calls `before_upsert` first, so a test can observe the world at write time."""

    def __init__(self) -> None:
        super().__init__()
        self.before_upsert: Callable[[], None] | None = None

    def upsert(self, tenant_id: UUID, points: Sequence[VectorPoint]) -> None:
        if self.before_upsert is not None:
            self.before_upsert()
        super().upsert(tenant_id, points)


class InMemoryChunkRepository:
    """Evidence chunks keyed by (tenant, collection); unknown ids are simply absent."""

    def __init__(self) -> None:
        self._chunks: dict[tuple[UUID, UUID, UUID], EvidenceChunk] = {}

    def add(self, tenant_id: UUID, collection_id: UUID, chunk: EvidenceChunk) -> None:
        self._chunks[(tenant_id, collection_id, chunk.chunk_id)] = chunk

    def remove(self, tenant_id: UUID, collection_id: UUID, chunk_id: UUID) -> None:
        del self._chunks[(tenant_id, collection_id, chunk_id)]

    def get_many(
        self, tenant_id: UUID, collection_id: UUID, chunk_ids: Sequence[UUID]
    ) -> list[EvidenceChunk]:
        found = (self._chunks.get((tenant_id, collection_id, i)) for i in chunk_ids)
        return [c for c in found if c is not None]


class FixedReranker:
    """Scores chunks from a table, so a test controls the rerank order exactly."""

    def __init__(self) -> None:
        self.scores: dict[UUID, float] = {}

    def score(self, question: str, chunks: Sequence[EvidenceChunk]) -> list[float]:
        return [self.scores.get(c.chunk_id, 0.0) for c in chunks]
