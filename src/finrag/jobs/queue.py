"""Procrastinate job queue: app, tasks and the JobQueue adapter."""

from contextlib import AbstractContextManager

import procrastinate
from procrastinate import SyncPsycopgConnector

from finrag.core.config import Settings
from finrag.core.domain import IngestionJob
from finrag.storage.db import psycopg_dsn

INGEST_TASK = "finrag.ingest_document_version"


def build_app(settings: Settings) -> procrastinate.App:
    app = procrastinate.App(connector=SyncPsycopgConnector(conninfo=psycopg_dsn(settings)))

    @app.task(name=INGEST_TASK)
    def ingest_document_version(tenant_id: str, document_id: str, document_version_id: str) -> None:
        # Implemented by a later M1 ticket; the worker does not run it yet.
        raise NotImplementedError("ingestion is not implemented yet")

    return app


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
