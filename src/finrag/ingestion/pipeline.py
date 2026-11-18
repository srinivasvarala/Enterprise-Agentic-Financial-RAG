"""The ingestion job: one Document Version from queued to ready (or failed).

Stages: parsing -> chunking -> embedding -> ready. Write order (ADR-0001): Chunks are persisted
in Postgres first, then upserted to the vector store, and only then does the Document become
ready, so `ready` always means "searchable". Re-running replaces the version's Chunks and points
(ids are deterministic), so retries and manual re-runs never duplicate.
"""

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, TypeVar
from uuid import UUID, uuid5

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from finrag.core.config import Settings
from finrag.core.domain import DocumentStatus, IngestionJob, SparseVector, VectorPoint
from finrag.core.errors import ParseError, TransientError
from finrag.core.interfaces import BlobStore, Embedder, Parser, SparseEncoder, VectorStore
from finrag.ingestion.chunking import ChunkDraft, chunk_document
from finrag.ingestion.parsed import ParsedDocument
from finrag.storage.models import Chunk, Document, DocumentVersion

log = logging.getLogger(__name__)
T = TypeVar("T")


def chunk_id_for(document_version_id: UUID, ordinal: int) -> UUID:
    """Deterministic Chunk id, reused as the index point id."""
    return uuid5(document_version_id, str(ordinal))


class IngestionPipeline:
    def __init__(
        self,
        session_factory: sessionmaker,  # type: ignore[type-arg]
        blob_store: BlobStore,
        parsers: Mapping[str, Parser],
        embedder: Embedder,
        sparse_encoder: SparseEncoder,
        vector_store: VectorStore,
        settings: Settings,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._session_factory = session_factory
        self._blobs = blob_store
        self._parsers = parsers
        self._embedder = embedder
        self._sparse = sparse_encoder
        self._vectors = vector_store
        self._settings = settings
        self._sleep = sleep

    def run(self, job: IngestionJob, job_id: str | None = None) -> None:
        """Ingest one Document Version. Terminal outcome is recorded on the Document."""
        ctx = {
            "tenant_id": str(job.tenant_id),
            "document_id": str(job.document_id),
            "document_version_id": str(job.document_version_id),
            "job_id": job_id,
        }
        stage = DocumentStatus.PARSING
        try:
            blob_key, media_type, collection_id = self._load(job)
            self._set_status(job, DocumentStatus.PARSING, ctx)
            parsed = self._parse(blob_key, media_type, ctx)

            stage = DocumentStatus.CHUNKING
            self._set_status(job, stage, ctx)
            drafts = chunk_document(parsed, self._settings.chunk_max_tokens)
            if not drafts:
                raise ParseError("The document contains no extractable text.")
            self._persist_chunks(job, collection_id, drafts)
            log.info("chunks persisted", extra={**ctx, "chunk_count": len(drafts)})

            stage = DocumentStatus.EMBEDDING
            self._set_status(job, stage, ctx)
            self._index(job, collection_id, drafts, ctx)

            self._set_status(job, DocumentStatus.READY, ctx)
        except ParseError as exc:
            self._fail(job, str(exc), ctx)
        except TransientError as exc:
            attempts = self._settings.ingest_max_attempts
            self._fail(
                job,
                f"A provider kept failing during {stage.value} after {attempts} attempts: {exc}",
                ctx,
            )
        except Exception:
            log.exception("ingestion crashed", extra=ctx)
            self._fail(job, f"Unexpected internal error during {stage.value}.", ctx)

    # -- stages ------------------------------------------------------------------------------

    def _parse(self, blob_key: str, media_type: str, ctx: dict[str, Any]) -> ParsedDocument:
        parser = self._parsers.get(media_type)
        if parser is None:
            raise ParseError(f"No parser is available for media type {media_type}.")
        data = self._retrying(lambda: self._blobs.get(blob_key), ctx, "blob get")
        return parser.parse(data)

    def _persist_chunks(
        self, job: IngestionJob, collection_id: UUID, drafts: Sequence[ChunkDraft]
    ) -> None:
        with self._session_factory() as session:
            session.execute(
                delete(Chunk).where(
                    Chunk.tenant_id == job.tenant_id,
                    Chunk.document_version_id == job.document_version_id,
                )
            )
            session.add_all(
                Chunk(
                    id=chunk_id_for(job.document_version_id, d.ordinal),
                    tenant_id=job.tenant_id,
                    collection_id=collection_id,
                    document_id=job.document_id,
                    document_version_id=job.document_version_id,
                    ordinal=d.ordinal,
                    kind=d.kind.value,
                    text=d.text,
                    section_path=list(d.section_path),
                    page_start=d.page_start,
                    page_end=d.page_end,
                    token_count=d.token_count,
                    bounding_boxes=[
                        {"page": page, "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1}
                        for page, b in d.bounding_boxes
                    ]
                    or None,
                )
                for d in drafts
            )
            document = self._document(session, job)
            document.chunk_count = len(drafts)
            session.commit()

    def _index(
        self,
        job: IngestionJob,
        collection_id: UUID,
        drafts: Sequence[ChunkDraft],
        ctx: dict[str, Any],
    ) -> None:
        texts = [d.index_text for d in drafts]
        dense: list[tuple[float, ...]] = []
        sparse: list[SparseVector] = []
        step = self._settings.embedding_batch_size
        for i in range(0, len(texts), step):
            batch = texts[i : i + step]
            dense += self._retrying(partial(self._embedder.embed_documents, batch), ctx, "embed")
            sparse += self._retrying(partial(self._sparse.encode_documents, batch), ctx, "bm25")
        points = [
            VectorPoint(
                id=chunk_id_for(job.document_version_id, d.ordinal),
                dense=dense[i],
                sparse=sparse[i],
                tenant_id=job.tenant_id,
                collection_id=collection_id,
                document_id=job.document_id,
                document_version_id=job.document_version_id,
                chunk_id=chunk_id_for(job.document_version_id, d.ordinal),
                kind=d.kind,
                page_start=d.page_start,
                page_end=d.page_end,
            )
            for i, d in enumerate(drafts)
        ]

        def write() -> None:
            self._vectors.ensure_collection(self._embedder.dimension)
            # Replace, not append: drops points of ordinals a previous run had but this one lacks.
            self._vectors.delete_version(job.tenant_id, job.document_version_id)
            self._vectors.upsert(job.tenant_id, points)

        self._retrying(write, ctx, "vector upsert")

    # -- helpers -----------------------------------------------------------------------------

    def _retrying(self, fn: Callable[[], T], ctx: dict[str, Any], what: str) -> T:
        attempts = self._settings.ingest_max_attempts
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except TransientError as exc:
                if attempt == attempts:
                    raise
                delay = self._settings.ingest_retry_base_seconds * 2 ** (attempt - 1)
                log.warning(
                    "transient error, retrying",
                    extra={
                        **ctx,
                        "step": what,
                        "attempt": attempt,
                        "delay_s": delay,
                        "error": str(exc),
                    },
                )
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _load(self, job: IngestionJob) -> tuple[str, str, UUID]:
        with self._session_factory() as session:
            document = self._document(session, job)
            version = session.scalars(
                select(DocumentVersion).where(
                    DocumentVersion.id == job.document_version_id,
                    DocumentVersion.tenant_id == job.tenant_id,
                    DocumentVersion.document_id == job.document_id,
                )
            ).one()
            return version.blob_key, document.media_type, document.collection_id

    @staticmethod
    def _document(session: Session, job: IngestionJob) -> Document:
        return session.scalars(
            select(Document).where(
                Document.id == job.document_id, Document.tenant_id == job.tenant_id
            )
        ).one()

    def _set_status(
        self,
        job: IngestionJob,
        status: DocumentStatus,
        ctx: dict[str, Any],
        reason: str | None = None,
    ) -> None:
        with self._session_factory() as session:
            document = self._document(session, job)
            document.status = status.value
            document.failure_reason = reason
            session.commit()
        log.info("status changed", extra={**ctx, "status": status.value, "failure_reason": reason})

    def _fail(self, job: IngestionJob, reason: str, ctx: dict[str, Any]) -> None:
        self._set_status(job, DocumentStatus.FAILED, ctx, reason)
