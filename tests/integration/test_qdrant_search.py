"""Hybrid search against the real Qdrant from Compose (`make up`), with fake models."""

from uuid import UUID, uuid4

import pytest

from finrag.core.container import Container
from finrag.core.domain import ChunkKind, VectorPoint
from finrag.core.errors import TenantRequiredError

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def collection(container: Container) -> None:
    container.vector_store.ensure_collection(container.embedder.dimension)


QUESTION = "how much cash did the company hold at year end"


def index(container: Container, tenant_id: UUID, collection_id: UUID, text: str) -> UUID:
    chunk_id = uuid4()
    container.vector_store.upsert(
        tenant_id,
        [
            VectorPoint(
                id=uuid4(),
                dense=container.embedder.embed_documents([text])[0],
                sparse=container.sparse_encoder.encode_documents([text])[0],
                tenant_id=tenant_id,
                collection_id=collection_id,
                document_id=uuid4(),
                document_version_id=uuid4(),
                chunk_id=chunk_id,
                kind=ChunkKind.TEXT,
                page_start=1,
                page_end=1,
            )
        ],
    )
    return chunk_id


def search(container: Container, tenant_id: UUID | None, collection_id: UUID) -> list[UUID]:
    hits = container.vector_store.search(
        tenant_id,  # type: ignore[arg-type]
        collection_id,
        container.embedder.embed_query(QUESTION),
        container.sparse_encoder.encode_query(QUESTION),
        dense_k=50,
        sparse_k=50,
        rrf_k=60,
        limit=100,
    )
    return [h.chunk_id for h in hits]


def test_search_returns_the_best_matching_chunk_first(container: Container) -> None:
    tenant, collection = uuid4(), uuid4()
    best = index(container, tenant, collection, QUESTION)
    index(container, tenant, collection, "an unrelated passage about office leases")

    assert search(container, tenant, collection)[0] == best


def test_search_never_returns_another_tenants_or_collections_chunks(container: Container) -> None:
    tenant, collection = uuid4(), uuid4()
    mine = index(container, tenant, collection, "cash was held at year end")
    index(container, uuid4(), collection, QUESTION)  # another Tenant, a perfect match
    index(container, tenant, uuid4(), QUESTION)  # same Tenant, another Collection

    assert search(container, tenant, collection) == [mine]


def test_search_without_a_tenant_is_refused(container: Container) -> None:
    with pytest.raises(TenantRequiredError):
        search(container, None, uuid4())
