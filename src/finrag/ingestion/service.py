"""Upload and tracking of Documents."""

import hashlib
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from finrag.core.config import Settings
from finrag.core.domain import ChunkKind, DocumentStatus, IngestionJob, Principal
from finrag.core.interfaces import BlobStore, JobQueue
from finrag.storage.models import Chunk, Document, DocumentVersion

SUPPORTED_MEDIA_TYPES = frozenset({"application/pdf", "text/plain"})


class UnsupportedMediaTypeError(Exception):
    pass


class DocumentTooLargeError(Exception):
    pass


class EmptyDocumentError(Exception):
    pass


class DocumentView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    collection_id: UUID
    filename: str
    media_type: str
    size_bytes: int
    status: DocumentStatus
    failure_reason: str | None
    chunk_count: int
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        # Timestamps are stored in UTC; some drivers (SQLite) return them naive.
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value


class ChunkView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    collection_id: UUID
    document_id: UUID
    document_version_id: UUID
    ordinal: int
    kind: ChunkKind
    text: str
    section_path: list[str]
    page_start: int
    page_end: int
    token_count: int


class ChunkPage(BaseModel):
    items: list[ChunkView]
    total: int
    limit: int
    offset: int


class DocumentService:
    def __init__(
        self,
        session_factory: sessionmaker,  # type: ignore[type-arg]
        blob_store: BlobStore,
        job_queue: JobQueue,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._blobs = blob_store
        self._jobs = job_queue
        self._settings = settings

    @property
    def max_upload_bytes(self) -> int:
        return self._settings.max_upload_bytes

    def upload(
        self, principal: Principal, filename: str, media_type: str, data: bytes
    ) -> tuple[DocumentView, bool]:
        """Store a Document. Returns (document, created); created is False for duplicates."""
        media_type = media_type.split(";")[0].strip().lower()
        if media_type not in SUPPORTED_MEDIA_TYPES:
            raise UnsupportedMediaTypeError(media_type)
        if len(data) > self._settings.max_upload_bytes:
            raise DocumentTooLargeError
        if not data:
            raise EmptyDocumentError

        sha256 = hashlib.sha256(data).hexdigest()
        with self._session_factory() as session:
            existing = self._find_by_hash(session, principal, sha256)
            if existing is not None:
                return DocumentView.model_validate(existing), False

            blob_key = f"{principal.tenant_id}/{sha256}"
            self._blobs.put(blob_key, data, media_type)
            document = Document(
                id=uuid4(),
                tenant_id=principal.tenant_id,
                collection_id=principal.collection_id,
                filename=filename,
                media_type=media_type,
                size_bytes=len(data),
            )
            version = DocumentVersion(
                id=uuid4(),
                tenant_id=principal.tenant_id,
                document_id=document.id,
                content_sha256=sha256,
                blob_key=blob_key,
                size_bytes=len(data),
            )
            session.add(document)
            session.flush()
            session.add(version)
            try:
                session.commit()
            except IntegrityError:  # concurrent identical upload won the race
                session.rollback()
                existing = self._find_by_hash(session, principal, sha256)
                if existing is None:
                    raise
                return DocumentView.model_validate(existing), False
            view = DocumentView.model_validate(document)
            # Enqueued after commit: the worker must be able to see the rows.
            self._jobs.enqueue_ingestion(
                IngestionJob(
                    tenant_id=principal.tenant_id,
                    document_id=document.id,
                    document_version_id=version.id,
                )
            )
            return view, True

    def list_documents(self, principal: Principal) -> list[DocumentView]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(Document)
                .where(Document.tenant_id == principal.tenant_id)
                .order_by(Document.created_at.desc(), Document.id)
            )
            return [DocumentView.model_validate(row) for row in rows]

    def get_document(self, principal: Principal, document_id: UUID) -> DocumentView | None:
        with self._session_factory() as session:
            row = session.scalars(
                select(Document).where(
                    Document.id == document_id, Document.tenant_id == principal.tenant_id
                )
            ).one_or_none()
            return None if row is None else DocumentView.model_validate(row)

    def list_chunks(
        self, principal: Principal, document_id: UUID, limit: int, offset: int
    ) -> ChunkPage | None:
        """A page of Chunks in ordinal order; None if the Document is absent or not the tenant's."""
        with self._session_factory() as session:
            owned = session.scalars(
                select(Document.id).where(
                    Document.id == document_id, Document.tenant_id == principal.tenant_id
                )
            ).one_or_none()
            if owned is None:
                return None
            where = (Chunk.tenant_id == principal.tenant_id, Chunk.document_id == document_id)
            total = session.scalar(select(func.count()).select_from(Chunk).where(*where)) or 0
            rows = session.scalars(
                select(Chunk).where(*where).order_by(Chunk.ordinal).limit(limit).offset(offset)
            )
            return ChunkPage(
                items=[ChunkView.model_validate(r) for r in rows],
                total=total,
                limit=limit,
                offset=offset,
            )

    @staticmethod
    def _find_by_hash(session: Session, principal: Principal, sha256: str) -> Document | None:
        return session.scalars(
            select(Document)
            .join(DocumentVersion, DocumentVersion.document_id == Document.id)
            .where(
                DocumentVersion.tenant_id == principal.tenant_id,
                DocumentVersion.content_sha256 == sha256,
            )
        ).first()
