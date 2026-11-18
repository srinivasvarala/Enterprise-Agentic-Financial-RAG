"""The Postgres ChunkRepository against the real database, including cross-tenant reads."""

from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from finrag.core.container import Container
from finrag.storage.chunk_repository import SqlChunkRepository
from finrag.storage.models import Chunk, DocumentVersion

pytestmark = pytest.mark.integration


def ingest_document(client: TestClient, container: Container) -> UUID:
    body = "\n\n".join(f"Paragraph {i} {uuid4()} " + "word " * 400 for i in range(3))
    doc_id = UUID(
        client.post("/documents", files={"file": ("f.txt", body.encode(), "text/plain")}).json()[
            "id"
        ]
    )
    with container.session_factory() as session:
        version = session.scalars(
            select(DocumentVersion).where(DocumentVersion.document_id == doc_id)
        ).one()
    from finrag.core.domain import IngestionJob

    container.ingestion_pipeline().run(
        IngestionJob(version.tenant_id, version.document_id, version.id), job_id="integration"
    )
    return doc_id


def test_chunks_load_with_the_text_pages_and_section_path_a_citation_needs(
    client: TestClient, container: Container
) -> None:
    doc_id = ingest_document(client, container)
    repo = SqlChunkRepository(container.session_factory)
    tenant = container.settings.default_tenant_id
    collection = container.settings.default_collection_id
    with container.session_factory() as session:
        stored = list(session.scalars(select(Chunk).where(Chunk.document_id == doc_id)))

    loaded = repo.get_many(tenant, collection, [c.id for c in stored])

    assert {c.chunk_id for c in loaded} == {c.id for c in stored}
    first = next(c for c in loaded if c.chunk_id == stored[0].id)
    assert first.text == stored[0].text
    assert first.document_id == doc_id
    assert first.section_path == tuple(stored[0].section_path)
    assert (first.page_start, first.page_end) == (stored[0].page_start, stored[0].page_end)


def test_another_tenants_or_collections_chunks_are_not_found(
    client: TestClient, container: Container
) -> None:
    doc_id = ingest_document(client, container)
    repo = SqlChunkRepository(container.session_factory)
    tenant = container.settings.default_tenant_id
    collection = container.settings.default_collection_id
    with container.session_factory() as session:
        ids = [c.id for c in session.scalars(select(Chunk).where(Chunk.document_id == doc_id))]

    assert repo.get_many(uuid4(), collection, ids) == []
    assert repo.get_many(tenant, uuid4(), ids) == []
    assert repo.get_many(tenant, collection, [uuid4()]) == []
