"""Procrastinate job queue: app, tasks and the JobQueue adapter."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from uuid import UUID

import procrastinate
from procrastinate import JobContext, SyncPsycopgConnector
from procrastinate.connector import BaseConnector

from finrag.core.config import Settings
from finrag.core.domain import IngestionJob
from finrag.core.logging import configure_logging
from finrag.ingestion.pipeline import IngestionPipeline
from finrag.storage.db import psycopg_dsn

INGEST_TASK = "finrag.ingest_document_version"


def build_app(
    settings: Settings,
    *,
    connector: BaseConnector | None = None,
    pipeline_provider: Callable[[], IngestionPipeline] | None = None,
) -> procrastinate.App:
    """The Procrastinate app. Tests inject an in-memory connector and a ready-made pipeline."""
    app = procrastinate.App(
        connector=connector or SyncPsycopgConnector(conninfo=psycopg_dsn(settings))
    )
    provider = pipeline_provider or _lazy_pipeline(settings)

    @app.task(name=INGEST_TASK, pass_context=True)
    def ingest_document_version(
        context: JobContext, tenant_id: str, document_id: str, document_version_id: str
    ) -> None:
        provider().run(
            IngestionJob(
                tenant_id=UUID(tenant_id),
                document_id=UUID(document_id),
                document_version_id=UUID(document_version_id),
            ),
            job_id=str(context.job.id),
        )

    return app


def _lazy_pipeline(settings: Settings) -> Callable[[], IngestionPipeline]:
    """Builds the worker's pipeline on first use and reuses it."""
    cache: list[IngestionPipeline] = []

    def provide() -> IngestionPipeline:
        if not cache:
            from finrag.core.container import build_ingestion_pipeline

            configure_logging(settings.log_level)
            cache.append(build_ingestion_pipeline(settings))
        return cache[0]

    return provide


class ProcrastinateJobQueue:
    def __init__(self, app: procrastinate.App) -> None:
        self._app = app
        self._open: AbstractContextManager[procrastinate.App] = app.open()
        self._open.__enter__()

    def enqueue_ingestion(self, job: IngestionJob) -> None:
        self._app.configure_task(INGEST_TASK).defer(
            tenant_id=str(job.tenant_id),
            document_id=str(job.document_id),
            document_version_id=str(job.document_version_id),
        )

    def close(self) -> None:
        self._open.__exit__(None, None, None)
