"""Test harness for Seam 1 unit tests."""

from dataclasses import dataclass
from uuid import uuid4

from fakes import FlakyEmbedder, InMemoryBlobStore, InMemoryJobQueue, SpyVectorStore
from fastapi import FastAPI
from fastapi.testclient import TestClient

from finrag.api.deps import current_principal
from finrag.core.config import Settings
from finrag.core.container import Container
from finrag.core.domain import IngestionJob, Principal
from finrag.ingestion.pipeline import IngestionPipeline
from finrag.storage.models import Collection, Tenant


@dataclass
class Env:
    app: FastAPI
    client: TestClient
    blobs: InMemoryBlobStore
    jobs: InMemoryJobQueue
    settings: Settings
    container: Container
    vectors: SpyVectorStore
    embedder: FlakyEmbedder
    sleeps: list[float]

    def pipeline(self) -> IngestionPipeline:
        c = self.container
        return IngestionPipeline(
            c.session_factory,
            c.blob_store,
            c.parsers,
            c.embedder,
            c.sparse_encoder,
            c.vector_store,
            c.settings,
            sleep=self.sleeps.append,  # record backoff instead of sleeping
        )

    def run_jobs(self) -> list[IngestionJob]:
        """Run every queued ingestion job synchronously, as the worker would."""
        pipeline = self.pipeline()
        jobs, self.jobs.jobs = self.jobs.jobs, []
        for n, job in enumerate(jobs):
            pipeline.run(job, job_id=f"test-job-{n}")
        return jobs

    def other_tenant(self) -> Principal:
        """Create a second Tenant + Collection and return its principal."""
        tenant_id, collection_id = uuid4(), uuid4()
        with self.container.session_factory() as session:
            session.add(Tenant(id=tenant_id, name="Other"))
            session.flush()
            session.add(Collection(id=collection_id, tenant_id=tenant_id, name="Other"))
            session.commit()
        return Principal(tenant_id=tenant_id, collection_id=collection_id)

    def act_as(self, principal: Principal) -> None:
        self.app.dependency_overrides[current_principal] = lambda: principal
