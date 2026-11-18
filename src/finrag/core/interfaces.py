"""Protocols for every external provider. Concrete adapters live in their own module."""

from typing import Protocol

from finrag.core.domain import IngestionJob


class BlobStore(Protocol):
    """Stores original files unchanged."""

    def put(self, key: str, data: bytes, content_type: str) -> None: ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


class JobQueue(Protocol):
    """Enqueues background work. Execution happens in a separate worker process."""

    def enqueue_ingestion(self, job: IngestionJob) -> None: ...
