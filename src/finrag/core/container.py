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
from finrag.ingestion.service import DocumentService

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

    def document_service(self) -> DocumentService:
        return DocumentService(self.session_factory, self.blob_store, self.job_queue, self.settings)

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


def build_parsers(settings: Settings) -> Mapping[str, Parser]:
    from finrag.ingestion.docling_parser import DoclingParser  # Docling itself loads lazily
    from finrag.ingestion.text_parser import TextParser

    return {
        "text/plain": TextParser(),
        "application/pdf": DoclingParser(ocr_enabled=settings.pdf_ocr_enabled),
    }


def build_indexing_adapters(settings: Settings) -> tuple[Embedder, SparseEncoder, VectorStore]:
    """Wire the Embedder, SparseEncoder and VectorStore chosen by Settings (ADR-0001, ADR-0005)."""
    from finrag.indexing.in_memory import HashEmbedder, HashSparseEncoder, InMemoryVectorStore

    embedder: Embedder
    if settings.embedder_backend == "voyage":
        from finrag.indexing.voyage import VoyageEmbedder

        embedder = VoyageEmbedder.from_settings(settings)
    else:
        embedder = HashEmbedder()

    sparse: SparseEncoder
    if settings.sparse_backend == "fastembed":
        from finrag.indexing.bm25 import FastEmbedBm25Encoder

        sparse = FastEmbedBm25Encoder.from_settings(settings)
    else:
        sparse = HashSparseEncoder()

    store: VectorStore
    if settings.vector_store_backend == "qdrant":
        from finrag.indexing.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore.from_url(
            settings.qdrant_url, settings.qdrant_collection, settings.qdrant_timeout_seconds
        )
    else:
        log.warning("using the in-memory vector store: indexed points are not persisted")
        store = InMemoryVectorStore()
    return embedder, sparse, store


def build_container(settings: Settings) -> Container:
    """Container for the API process: enqueues ingestion jobs onto Procrastinate."""
    from finrag.jobs.queue import ProcrastinateJobQueue, build_app

    return _build(settings, ProcrastinateJobQueue(build_app(settings)))


def build_ingestion_pipeline(settings: Settings) -> IngestionPipeline:
    """Pipeline for the worker process, which consumes the queue rather than feeding it."""
    return _build(settings, _NoQueue()).ingestion_pipeline()


class _NoQueue:
    def enqueue_ingestion(self, job: IngestionJob) -> bool:
        raise RuntimeError("this container has no job queue")


def _build(settings: Settings, job_queue: JobQueue) -> Container:
    from finrag.storage.db import build_engine, build_session_factory
    from finrag.storage.s3_blob_store import S3BlobStore

    blob_store = S3BlobStore.from_settings(settings)
    blob_store.ensure_bucket()
    embedder, sparse_encoder, vector_store = build_indexing_adapters(settings)
    return Container(
        settings=settings,
        session_factory=build_session_factory(build_engine(settings)),
        blob_store=blob_store,
        job_queue=job_queue,
        parsers=build_parsers(settings),
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        vector_store=vector_store,
    )
