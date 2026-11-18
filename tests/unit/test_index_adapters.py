"""Unit tests for the Voyage, FastEmbed BM25 and Qdrant adapters: no network, no model download."""

import json
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from finrag.core.config import Settings
from finrag.core.container import build_indexing_adapters
from finrag.core.domain import ChunkKind, SparseVector, VectorPoint
from finrag.core.errors import TenantRequiredError, TransientError
from finrag.indexing.bm25 import FastEmbedBm25Encoder
from finrag.indexing.in_memory import HashEmbedder, HashSparseEncoder, InMemoryVectorStore
from finrag.indexing.qdrant_store import (
    QdrantVectorStore,
    point_to_struct,
    record_to_point,
    tenant_filter,
)
from finrag.indexing.voyage import VoyageEmbedder


def voyage(handler, batch_size: int = 2) -> VoyageEmbedder:  # type: ignore[no-untyped-def]
    client = httpx.Client(
        base_url="https://voyage.test/v1",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer k"},
    )
    return VoyageEmbedder(
        "k",
        "voyage-finance-2",
        3,
        base_url="http://voyage.test/v1",
        batch_size=batch_size,
        timeout_seconds=1.0,
        client=client,
    )


def ok(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    data = [{"index": i, "embedding": [float(len(t)), 0.0, 1.0]} for i, t in enumerate(texts)]
    return httpx.Response(200, json={"data": data[::-1]})  # out of order on purpose


def test_voyage_sends_document_input_type_and_model_and_batches() -> None:
    requests: list[dict] = []  # type: ignore[type-arg]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return ok(request)

    vectors = voyage(handler).embed_documents(["a", "bb", "ccc"])

    assert [r["input"] for r in requests] == [["a", "bb"], ["ccc"]]
    assert all(r["input_type"] == "document" for r in requests)
    assert all(r["model"] == "voyage-finance-2" for r in requests)
    assert vectors == [(1.0, 0.0, 1.0), (2.0, 0.0, 1.0), (3.0, 0.0, 1.0)]  # order restored


@pytest.mark.parametrize("status", [429, 500, 503])
def test_voyage_maps_rate_limit_and_5xx_to_transient(status: int) -> None:
    with pytest.raises(TransientError):
        voyage(lambda r: httpx.Response(status)).embed_documents(["a"])


def test_voyage_maps_timeouts_to_transient() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(TransientError):
        voyage(boom).embed_documents(["a"])


def test_voyage_does_not_retry_auth_errors() -> None:
    with pytest.raises(httpx.HTTPStatusError):
        voyage(lambda r: httpx.Response(401, request=r)).embed_documents(["a"])


def test_voyage_rejects_wrong_dimension() -> None:
    bad = lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})  # noqa: E731
    with pytest.raises(ValueError, match="unexpected shapes"):
        voyage(bad).embed_documents(["a"])


def test_voyage_requires_an_api_key_from_settings() -> None:
    with pytest.raises(ValueError, match="FINRAG_VOYAGE_API_KEY"):
        VoyageEmbedder.from_settings(Settings(voyage_api_key=""))  # type: ignore[arg-type]
    embedder = VoyageEmbedder.from_settings(
        Settings(voyage_api_key="secret", voyage_dimension=512)  # type: ignore[arg-type]
    )
    assert embedder.dimension == 512


def test_bm25_encoder_maps_model_output_and_loads_lazily() -> None:
    loads = 0

    class FakeModel:
        def embed(self, documents):  # type: ignore[no-untyped-def]
            return [SimpleNamespace(indices=[3, 9], values=[0.5, 1.5]) for _ in documents]

    def factory() -> FakeModel:
        nonlocal loads
        loads += 1
        return FakeModel()

    encoder = FastEmbedBm25Encoder(model_factory=factory)
    assert loads == 0
    out = encoder.encode_documents(["x", "y"])
    encoder.encode_documents(["z"])

    assert out == [SparseVector((3, 9), (0.5, 1.5))] * 2
    assert loads == 1


def make_point(tenant_id: UUID | None = None) -> VectorPoint:
    return VectorPoint(
        id=uuid4(),
        dense=(0.1, 0.2),
        sparse=SparseVector((1, 2), (1.0, 2.0)),
        tenant_id=tenant_id or uuid4(),
        collection_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        chunk_id=uuid4(),
        kind=ChunkKind.TABLE,
        page_start=2,
        page_end=3,
    )


def test_point_payload_and_named_vectors() -> None:
    point = make_point()
    struct = point_to_struct(point)

    assert struct.payload == {
        "tenant_id": str(point.tenant_id),
        "collection_id": str(point.collection_id),
        "document_id": str(point.document_id),
        "document_version_id": str(point.document_version_id),
        "chunk_id": str(point.chunk_id),
        "kind": "table",
        "page_start": 2,
        "page_end": 3,
    }
    assert set(struct.vector) == {"dense", "bm25"}  # type: ignore[arg-type]


def test_point_round_trips_through_a_qdrant_record() -> None:
    point = make_point()
    struct = point_to_struct(point)
    record = SimpleNamespace(id=struct.id, payload=struct.payload, vector=struct.vector)

    assert record_to_point(record) == point


def test_tenant_filter_requires_a_tenant_and_always_includes_it() -> None:
    with pytest.raises(TenantRequiredError):
        tenant_filter(None)
    tenant, version = uuid4(), uuid4()
    flt = tenant_filter(tenant, document_version_id=version)
    assert [c.key for c in flt.must] == ["tenant_id", "document_version_id"]  # type: ignore[union-attr]


def test_store_rejects_tenantless_calls_before_touching_the_network() -> None:
    store = QdrantVectorStore(client=None, collection="x")  # type: ignore[arg-type]
    with pytest.raises(TenantRequiredError):
        store.upsert(None, [])  # type: ignore[arg-type]
    with pytest.raises(TenantRequiredError):
        store.delete_version(None, uuid4())  # type: ignore[arg-type]
    with pytest.raises(TenantRequiredError):
        store.get_version_points(None, uuid4())  # type: ignore[arg-type]
    with pytest.raises(TenantRequiredError):  # a point of another tenant in the batch
        store.upsert(uuid4(), [make_point()])


class _FailingClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def delete(self, *args: object, **kwargs: object) -> None:
        raise self.error


def _delete_with(error: Exception) -> None:
    store = QdrantVectorStore(client=_FailingClient(error), collection="x")  # type: ignore[arg-type]
    store.delete_version(uuid4(), uuid4())


def test_qdrant_failures_map_to_transient_only_when_retryable() -> None:
    headers = httpx.Headers()
    for retryable in (
        ResponseHandlingException(Exception("timeout")),
        UnexpectedResponse(429, "Too Many", b"", headers),
        UnexpectedResponse(503, "Down", b"", headers),
    ):
        with pytest.raises(TransientError):
            _delete_with(retryable)
    bad_request = UnexpectedResponse(400, "Bad", b"", headers)
    with pytest.raises(UnexpectedResponse):
        _delete_with(bad_request)


def test_container_picks_adapters_from_settings() -> None:
    settings = Settings(
        embedder_backend="hash", sparse_backend="hash", vector_store_backend="memory"
    )
    embedder, sparse, store = build_indexing_adapters(settings)
    assert isinstance(embedder, HashEmbedder)
    assert isinstance(sparse, HashSparseEncoder)
    assert isinstance(store, InMemoryVectorStore)

    real = Settings(voyage_api_key="k", qdrant_collection="c")  # type: ignore[arg-type]
    embedder, sparse, store = build_indexing_adapters(real)
    assert isinstance(embedder, VoyageEmbedder)
    assert isinstance(sparse, FastEmbedBm25Encoder)
    assert isinstance(store, QdrantVectorStore)
