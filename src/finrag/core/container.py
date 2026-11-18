"""Composition root: the only place concrete adapters are wired from Settings."""

from dataclasses import dataclass

from sqlalchemy.orm import sessionmaker

from finrag.core.config import Settings
from finrag.core.interfaces import BlobStore, JobQueue


@dataclass
class Container:
    settings: Settings
    session_factory: sessionmaker  # type: ignore[type-arg]
    blob_store: BlobStore
    job_queue: JobQueue


def build_container(settings: Settings) -> Container:
    from finrag.jobs.queue import ProcrastinateJobQueue, build_app
    from finrag.storage.db import build_engine, build_session_factory
    from finrag.storage.s3_blob_store import S3BlobStore

    blob_store = S3BlobStore.from_settings(settings)
    blob_store.ensure_bucket()
    return Container(
        settings=settings,
        session_factory=build_session_factory(build_engine(settings)),
        blob_store=blob_store,
        job_queue=ProcrastinateJobQueue(build_app(settings)),
    )
