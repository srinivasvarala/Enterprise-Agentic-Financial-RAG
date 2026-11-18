"""Reconciler: re-defers ingestion jobs for Documents that a crash or a lost enqueue orphaned.

Two failure modes leave a Document non-terminal forever: the worker dies (OOM, kill) mid-stage,
leaving `parsing`/`chunking`/`embedding`; or the API commits the Document but the enqueue that
follows fails, leaving `queued` with no job. Every status change bumps `updated_at`, so a
non-terminal Document untouched for `stuck_document_seconds` is presumed orphaned.

Re-deferring is safe because the pipeline is idempotent (deterministic ids, replace-not-append).
Duplicate jobs are prevented by the queueing lock `ProcrastinateJobQueue` puts on every ingestion
job (one waiting job per Document Version): a Document that already has a job waiting is skipped.
The one residual case is a *running* job on a live worker that exceeds the threshold: a second job
then waits behind it (worker concurrency is 1) and re-runs the same work idempotently.

Runs at worker startup and on demand: `make reconcile` / `python -m finrag.jobs.reconcile`.
"""

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from finrag.core.config import Settings, get_settings
from finrag.core.domain import DocumentStatus, IngestionJob
from finrag.core.interfaces import JobQueue
from finrag.core.logging import configure_logging
from finrag.storage.models import Document, DocumentVersion

log = logging.getLogger(__name__)

_IN_FLIGHT = (
    DocumentStatus.QUEUED,
    DocumentStatus.PARSING,
    DocumentStatus.CHUNKING,
    DocumentStatus.EMBEDDING,
)


def reconcile_stuck_documents(
    session_factory: sessionmaker,  # type: ignore[type-arg]
    queue: JobQueue,
    stuck_after_seconds: int,
    now: datetime | None = None,
) -> list[IngestionJob]:
    """Re-defer the newest version of every stuck Document. Returns the jobs actually enqueued.

    Spans all Tenants by design (it is a system process), but every job carries the tenant id of
    the row it was derived from.
    """
    cutoff = (now or datetime.now(UTC)) - timedelta(seconds=stuck_after_seconds)
    with session_factory() as session:
        rows = session.execute(
            select(Document.tenant_id, Document.id, DocumentVersion.id)
            .join(
                DocumentVersion,
                (DocumentVersion.document_id == Document.id)
                & (DocumentVersion.tenant_id == Document.tenant_id),
            )
            .where(
                Document.status.in_([s.value for s in _IN_FLIGHT]),
                Document.updated_at < cutoff,
            )
            .order_by(Document.id, DocumentVersion.created_at.desc())
        ).all()

    newest: dict[object, IngestionJob] = {}
    for tenant_id, document_id, version_id in rows:
        newest.setdefault(document_id, IngestionJob(tenant_id, document_id, version_id))

    enqueued: list[IngestionJob] = []
    for job in newest.values():
        extra = {
            "tenant_id": str(job.tenant_id),
            "document_id": str(job.document_id),
            "document_version_id": str(job.document_version_id),
        }
        if queue.enqueue_ingestion(job):
            enqueued.append(job)
            log.warning("stuck document re-deferred", extra=extra)
        else:
            log.info("stuck document already has a queued job", extra=extra)
    return enqueued


def run_reconciliation(settings: Settings) -> list[IngestionJob]:
    """Reconcile against the real database and Procrastinate queue."""
    from finrag.jobs.queue import ProcrastinateJobQueue, build_app
    from finrag.storage.db import build_engine, build_session_factory

    queue = ProcrastinateJobQueue(build_app(settings))
    try:
        return reconcile_stuck_documents(
            build_session_factory(build_engine(settings)), queue, settings.stuck_document_seconds
        )
    finally:
        queue.close()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    jobs = run_reconciliation(settings)
    log.info("reconciliation finished", extra={"redeferred": len(jobs)})


if __name__ == "__main__":
    main()
