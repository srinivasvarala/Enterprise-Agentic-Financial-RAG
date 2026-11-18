"""The Qdrant VectorStore against the real Qdrant from Compose (`make up`), with fake models."""

from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import select

from finrag.core.config import Settings
from finrag.core.container import Container
from finrag.core.domain import ChunkKind, IngestionJob, SparseVector, VectorPoint
from finrag.core.errors import TenantRequiredError
from finrag.storage.models import Chunk, DocumentVersion

pytestmark = pytest.mark.integration


def ingest(client: TestClient, container: Container) -> tuple[IngestionJob, str]:
    body = "\n\n".join(f"Paragraph {i} {uuid4()} " + "word " * 400 for i in range(4))
    doc_id = client.post(
        "/documents", files={"file": ("f.txt", body.encode(), "text/plain")}
    ).json()["id"]
    with container.session_factory() as session:
        version = session.scalars(
            select(DocumentVersion).where(DocumentVersion.document_id == UUID(doc_id))
        ).one()
    job = IngestionJob(version.tenant_id, version.document_id, version.id)
    container.ingestion_pipeline().run(job, job_id="integration")
    return job, doc_id


def raw_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def test_ingestion_indexes_chunks_with_both_vectors_and_the_right_payload(
    client: TestClient, container: Container, settings: Settings
) -> None:
    job, doc_id = ingest(client, container)
    assert client.get(f"/documents/{doc_id}").json()["status"] == "ready"

    with container.session_factory() as session:
        chunks = {
            c.id: c
            for c in session.scalars(select(Chunk).where(Chunk.document_id == job.document_id))
        }
    points = container.vector_store.get_version_points(job.tenant_id, job.document_version_id)

    assert len(points) == len(chunks) > 1
    for p in points:
        chunk = chunks[p.id]
        assert len(p.dense) == container.embedder.dimension
        assert p.sparse.indices and len(p.sparse.indices) == len(p.sparse.values)
        assert (p.tenant_id, p.collection_id) == (job.tenant_id, chunk.collection_id)
        assert (p.document_id, p.document_version_id) == (job.document_id, job.document_version_id)
        assert p.chunk_id == chunk.id
        assert p.kind == ChunkKind(chunk.kind)
        assert (p.page_start, p.page_end) == (chunk.page_start, chunk.page_end)


def test_collection_has_named_cosine_dense_and_idf_bm25_and_payload_indexes(
    client: TestClient, container: Container, settings: Settings
) -> None:
    ingest(client, container)  # ensures the collection exists
    info = raw_client(settings).get_collection(settings.qdrant_collection)

    dense = info.config.params.vectors["dense"]  # type: ignore[index]
    assert dense.distance.value == "Cosine"
    assert dense.size == container.embedder.dimension
    sparse = info.config.params.sparse_vectors["bm25"]  # type: ignore[index]
    assert sparse.modifier is not None and sparse.modifier.value == "idf"
    assert set(info.payload_schema) >= {
        "tenant_id",
        "collection_id",
        "document_id",
        "document_version_id",
        "chunk_id",
        "kind",
        "page_start",
        "page_end",
    }
    tenant_index = info.payload_schema["tenant_id"].params
    assert tenant_index is not None and tenant_index.is_tenant is True  # type: ignore[union-attr]


def test_rerunning_ingestion_leaves_the_same_point_count(
    client: TestClient, container: Container
) -> None:
    job, _ = ingest(client, container)
    store = container.vector_store
    before = store.get_version_points(job.tenant_id, job.document_version_id)

    container.ingestion_pipeline().run(job, job_id="rerun")  # also re-runs ensure_collection

    after = store.get_version_points(job.tenant_id, job.document_version_id)
    assert len(after) == len(before) > 0
    assert {p.id for p in after} == {p.id for p in before}


def test_ensure_collection_is_idempotent_and_rejects_a_dimension_change(
    container: Container,
) -> None:
    store = container.vector_store
    dim = container.embedder.dimension
    store.ensure_collection(dim)
    store.ensure_collection(dim)
    with pytest.raises(ValueError, match="dense size"):
        store.ensure_collection(dim + 1)


def test_tenantless_calls_are_rejected(container: Container) -> None:
    store = container.vector_store
    with pytest.raises(TenantRequiredError):
        store.get_version_points(None, uuid4())  # type: ignore[arg-type]
    with pytest.raises(TenantRequiredError):
        store.delete_version(None, uuid4())  # type: ignore[arg-type]
    with pytest.raises(TenantRequiredError):
        store.upsert(None, [])  # type: ignore[arg-type]


def test_two_tenants_never_see_each_others_points(container: Container) -> None:
    store = container.vector_store
    store.ensure_collection(container.embedder.dimension)
    version = uuid4()  # same version id on purpose: only the tenant id may separate them
    tenant_a, tenant_b = uuid4(), uuid4()

    def point(tenant: UUID) -> VectorPoint:
        return VectorPoint(
            id=uuid4(),
            dense=container.embedder.embed_documents(["hello"])[0],
            sparse=SparseVector((1,), (1.0,)),
            tenant_id=tenant,
            collection_id=uuid4(),
            document_id=uuid4(),
            document_version_id=version,
            chunk_id=uuid4(),
            kind=ChunkKind.TEXT,
            page_start=1,
            page_end=1,
        )

    a, b = point(tenant_a), point(tenant_b)
    store.upsert(tenant_a, [a])
    store.upsert(tenant_b, [b])

    assert [p.id for p in store.get_version_points(tenant_a, version)] == [a.id]
    assert [p.id for p in store.get_version_points(tenant_b, version)] == [b.id]

    store.delete_version(tenant_a, version)  # must not touch tenant B
    assert store.get_version_points(tenant_a, version) == []
    assert [p.id for p in store.get_version_points(tenant_b, version)] == [b.id]

    with pytest.raises(TenantRequiredError):  # cannot write a point under someone else's tenant
        store.upsert(tenant_a, [b])
