"""Postgres-backed ChunkRepository: the system of record for Chunk text and Citation data."""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from finrag.core.domain import EvidenceChunk
from finrag.storage.models import Chunk


class SqlChunkRepository:
    def __init__(self, session_factory: sessionmaker) -> None:  # type: ignore[type-arg]
        self._session_factory = session_factory

    def get_many(
        self, tenant_id: UUID, collection_id: UUID, chunk_ids: Sequence[UUID]
    ) -> list[EvidenceChunk]:
        """Chunks of this Tenant's Collection only; other ids are simply absent."""
        if not chunk_ids:
            return []
        query = select(Chunk).where(
            Chunk.tenant_id == tenant_id,
            Chunk.collection_id == collection_id,
            Chunk.id.in_(chunk_ids),
        )
        with self._session_factory() as session:
            return [
                EvidenceChunk(
                    chunk_id=c.id,
                    document_id=c.document_id,
                    text=c.text,
                    section_path=tuple(c.section_path),
                    page_start=c.page_start,
                    page_end=c.page_end,
                )
                for c in session.scalars(query)
            ]
