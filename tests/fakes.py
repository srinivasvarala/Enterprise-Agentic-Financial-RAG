"""In-memory fakes for the Protocols in finrag.core.interfaces."""

from collections.abc import Callable, Sequence
from uuid import UUID

from finrag.core.domain import IngestionJob, VectorPoint
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

    def enqueue_ingestion(self, job: IngestionJob) -> None:
        self.jobs.append(job)


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
