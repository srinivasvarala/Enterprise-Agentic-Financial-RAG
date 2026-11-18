"""In-memory fakes for the Protocols in finrag.core.interfaces."""

from finrag.core.domain import IngestionJob


class InMemoryBlobStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = data

    def get(self, key: str) -> bytes:
        return self.objects[key]

    def exists(self, key: str) -> bool:
        return key in self.objects


class InMemoryJobQueue:
    def __init__(self) -> None:
        self.jobs: list[IngestionJob] = []

    def enqueue_ingestion(self, job: IngestionJob) -> None:
        self.jobs.append(job)
