"""The eval environment: its own Tenant, Collection, vector-store collection and blob bucket.

Eval documents live apart from the dev stack's, so results never mix with dev documents and the
default Tenant never sees them. The compose `eval-index` service sets these values through the
`FINRAG_*` settings; `require_eval_settings` refuses to run against anything else. Later tickets
(fingerprint, retrieval and answer runs) import the constants from here.
"""

from uuid import UUID

from sqlalchemy.orm import sessionmaker

from finrag.core.config import Settings
from finrag.core.container import Container, build_queueless_container
from finrag.core.domain import DocumentStatus, IngestionJob, Principal
from finrag.evaluation.index import IndexBuildError
from finrag.ingestion.pipeline import IngestionPipeline
from finrag.ingestion.service import DocumentService
from finrag.storage.models import Collection, DocumentVersion, Tenant

EVAL_TENANT_ID = UUID("00000000-0000-4000-8000-0000000000e1")
EVAL_COLLECTION_ID = UUID("00000000-0000-4000-8000-0000000000e2")
EVAL_QDRANT_COLLECTION = "finrag_eval_chunks"
EVAL_S3_BUCKET = "finrag-eval-documents"


class EvalEnvironmentError(Exception):
    """The process is not configured for the eval environment."""


def eval_principal() -> Principal:
    return Principal(tenant_id=EVAL_TENANT_ID, collection_id=EVAL_COLLECTION_ID)


def require_eval_settings(settings: Settings) -> None:
    """Refuse to touch the dev stack's Tenant, Qdrant collection or bucket."""
    wrong = [
        name
        for name, actual, expected in (
            ("FINRAG_DEFAULT_TENANT_ID", settings.default_tenant_id, EVAL_TENANT_ID),
            ("FINRAG_DEFAULT_COLLECTION_ID", settings.default_collection_id, EVAL_COLLECTION_ID),
            ("FINRAG_QDRANT_COLLECTION", settings.qdrant_collection, EVAL_QDRANT_COLLECTION),
            ("FINRAG_S3_BUCKET", settings.s3_bucket, EVAL_S3_BUCKET),
        )
        if actual != expected
    ]
    if wrong:
        raise EvalEnvironmentError(
            f"not the eval environment ({', '.join(wrong)} differ); run it through "
            "`make eval-index`, which uses the compose `eval` profile."
        )


def seed_eval_tenant(session_factory: sessionmaker) -> None:  # type: ignore[type-arg]
    """Idempotent: create the eval Tenant and Collection if missing."""
    with session_factory() as session:
        if session.get(Tenant, EVAL_TENANT_ID) is None:
            session.add(Tenant(id=EVAL_TENANT_ID, name="Eval"))
            session.flush()
        if session.get(Collection, EVAL_COLLECTION_ID) is None:
            session.add(Collection(id=EVAL_COLLECTION_ID, tenant_id=EVAL_TENANT_ID, name="Eval"))
        session.commit()


class _CapturedJobs:
    """A JobQueue that keeps jobs for the caller to run in-process: no worker, no Procrastinate."""

    def __init__(self) -> None:
        self.jobs: list[IngestionJob] = []

    def enqueue_ingestion(self, job: IngestionJob) -> bool:
        self.jobs.append(job)
        return True


class PipelineFilingIndexer:
    """Ingests PDF Filings into the eval Tenant and Collection via the real IngestionPipeline."""

    def __init__(self, container: Container, pipeline: IngestionPipeline) -> None:
        self._container = container
        self._pipeline = pipeline
        self._queue = _CapturedJobs()
        self._documents = DocumentService(
            container.session_factory, container.blob_store, self._queue, container.settings
        )
        self._principal = eval_principal()

    def index(self, filename: str, pdf: bytes) -> bool:
        view, created = self._documents.upload(self._principal, filename, "application/pdf", pdf)
        if created:
            jobs, self._queue.jobs = self._queue.jobs, []
        elif view.status == DocumentStatus.READY:
            return False
        else:  # a earlier run failed or was interrupted part-way: run it again
            jobs = [self._job_for(view.id)]
        for job in jobs:
            self._pipeline.run(job)
        done = self._documents.get_document(self._principal, view.id)
        if done is None or done.status != DocumentStatus.READY:
            reason = done.failure_reason if done is not None else "the document vanished"
            raise IndexBuildError(f"{filename} did not become searchable: {reason}")
        return True

    def _job_for(self, document_id: UUID) -> IngestionJob:
        with self._container.session_factory() as session:
            version = (
                session.query(DocumentVersion)
                .filter(DocumentVersion.document_id == document_id)
                .order_by(DocumentVersion.created_at.desc())
                .first()
            )
            assert version is not None  # upload creates the version with the document
            return IngestionJob(
                tenant_id=self._principal.tenant_id,
                document_id=document_id,
                document_version_id=version.id,
            )


def open_pipeline_indexer(settings: Settings) -> PipelineFilingIndexer:
    """Wire the real adapters from Settings, seed the eval Tenant, and check the vector index."""
    require_eval_settings(settings)
    container = build_queueless_container(settings)
    seed_eval_tenant(container.session_factory)
    pipeline = container.ingestion_pipeline()
    pipeline.startup_check()
    return PipelineFilingIndexer(container, pipeline)
