"""Composition root: the only place concrete adapters are wired from Settings."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from sqlalchemy.orm import sessionmaker

from finrag.core.config import Settings
from finrag.core.domain import IngestionJob
from finrag.core.interfaces import (
    BlobStore,
    Embedder,
    JobQueue,
    Parser,
    SparseEncoder,
    VectorStore,
)
from finrag.ingestion.pipeline import IngestionPipeline

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    session_factory: sessionmaker  # type: ignore[type-arg]
    blob_store: BlobStore
    job_queue: JobQueue
    parsers: Mapping[str, Parser]
    embedder: Embedder
    sparse_encoder: SparseEncoder
    vector_store: VectorStore

    def ingestion_pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(
            self.session_factory,
            self.blob_store,
            self.parsers,
            self.embedder,
            self.sparse_encoder,
            self.vector_store,
            self.settings,
        )


def build_parsers() -> Mapping[str, Parser]:
    from finrag.ingestion.text_parser import TextParser

    return {"text/plain": TextParser()}  # PDF parser arrives with the Docling ticket


def build_indexing_adapters() -> tuple[Embedder, SparseEncoder, VectorStore]:
    """The one place to swap in the Voyage / FastEmbed BM25 / Qdrant adapters (ticket #4)."""
    from finrag.indexing.in_memory import HashEmbedder, HashSparseEncoder, InMemoryVectorStore

    log.warning("using in-memory index adapters: indexed points are not persisted")
    return HashEmbedder(), HashSparseEncoder(), InMemoryVectorStore()


def build_container(settings: Settings) -> Container:
    """Container for the API process: enqueues ingestion jobs onto Procrastinate."""
    from finrag.jobs.queue import ProcrastinateJobQueue, build_app

    return _build(settings, ProcrastinateJobQueue(build_app(settings)))


def build_ingestion_pipeline(settings: Settings) -> IngestionPipeline:
    """Pipeline for the worker process, which consumes the queue rather than feeding it."""
    return _build(settings, _NoQueue()).ingestion_pipeline()


class _NoQueue:
    def enqueue_ingestion(self, job: IngestionJob) -> None:
        raise RuntimeError("this container has no job queue")


def _build(settings: Settings, job_queue: JobQueue) -> Container:
    from finrag.storage.db import build_engine, build_session_factory
    from finrag.storage.s3_blob_store import S3BlobStore

    blob_store = S3BlobStore.from_settings(settings)
    blob_store.ensure_bucket()
    embedder, sparse_encoder, vector_store = build_indexing_adapters()
    return Container(
        settings=settings,
        session_factory=build_session_factory(build_engine(settings)),
        blob_store=blob_store,
        job_queue=job_queue,
        parsers=build_parsers(),
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        vector_store=vector_store,
    )
