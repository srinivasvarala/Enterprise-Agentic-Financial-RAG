"""Qdrant adapter for the VectorStore Protocol (ADR-0001).

One collection holds every Tenant's points: a named cosine `dense` vector and a named `bm25`
sparse vector with the IDF modifier. Isolation is by payload: `tenant_id` is a tenant-optimised
keyword index and every call filters on it. Calls without a tenant id are rejected before any
network traffic.
"""

import logging
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from finrag.core.domain import ChunkKind, SparseVector, VectorPoint
from finrag.core.errors import TransientError
from finrag.core.tenancy import require_points_of_tenant, require_tenant

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"
UPSERT_BATCH = 128
SCROLL_PAGE = 256

# tenant_id gets the tenant index (co-locates a Tenant's vectors); the rest are plain indexes.
_KEYWORD_FIELDS = ("collection_id", "document_id", "document_version_id", "chunk_id", "kind")
_INTEGER_FIELDS = ("page_start", "page_end")


def point_to_struct(point: VectorPoint) -> models.PointStruct:
    return models.PointStruct(
        id=str(point.id),
        vector={
            DENSE: list(point.dense),
            SPARSE: models.SparseVector(
                indices=list(point.sparse.indices), values=list(point.sparse.values)
            ),
        },
        payload={
            "tenant_id": str(point.tenant_id),
            "collection_id": str(point.collection_id),
            "document_id": str(point.document_id),
            "document_version_id": str(point.document_version_id),
            "chunk_id": str(point.chunk_id),
            "kind": point.kind.value,
            "page_start": point.page_start,
            "page_end": point.page_end,
        },
    )


def record_to_point(record: Any) -> VectorPoint:
    payload = record.payload
    vectors = record.vector
    sparse = vectors[SPARSE]
    return VectorPoint(
        id=UUID(str(record.id)),
        dense=tuple(vectors[DENSE]),
        sparse=SparseVector(tuple(sparse.indices), tuple(sparse.values)),
        tenant_id=UUID(payload["tenant_id"]),
        collection_id=UUID(payload["collection_id"]),
        document_id=UUID(payload["document_id"]),
        document_version_id=UUID(payload["document_version_id"]),
        chunk_id=UUID(payload["chunk_id"]),
        kind=ChunkKind(payload["kind"]),
        page_start=payload["page_start"],
        page_end=payload["page_end"],
    )


def tenant_filter(tenant_id: UUID | None, **equals: UUID) -> models.Filter:
    """The only way this module builds a filter: tenant id is mandatory."""
    conditions: list[Any] = [
        models.FieldCondition(
            key="tenant_id", match=models.MatchValue(value=str(require_tenant(tenant_id)))
        )
    ]
    conditions += [
        models.FieldCondition(key=key, match=models.MatchValue(value=str(value)))
        for key, value in equals.items()
    ]
    return models.Filter(must=conditions)


def _to_transient(exc: Exception) -> Exception:
    """Map a qdrant-client failure to TransientError when a retry may help, else return it."""
    if isinstance(exc, ResponseHandlingException):  # timeouts, connection errors
        return TransientError(f"qdrant unavailable: {exc}")
    if isinstance(exc, UnexpectedResponse) and (exc.status_code in (429, 500, 502, 503, 504)):
        return TransientError(f"qdrant returned {exc.status_code}")
    return exc


def _guarded[T](fn: Callable[[], T]) -> T:
    try:
        return fn()
    except (ResponseHandlingException, UnexpectedResponse) as exc:
        mapped = _to_transient(exc)
        if mapped is exc:
            raise
        raise mapped from exc


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    @classmethod
    def from_url(cls, url: str, collection: str, timeout_seconds: float) -> "QdrantVectorStore":
        return cls(QdrantClient(url=url, timeout=int(timeout_seconds)), collection)

    def ensure_collection(self, dimension: int) -> None:
        _guarded(lambda: self._ensure_collection(dimension))

    def upsert(self, tenant_id: UUID, points: Sequence[VectorPoint]) -> None:
        require_points_of_tenant(require_tenant(tenant_id), points)
        for i in range(0, len(points), UPSERT_BATCH):
            batch = [point_to_struct(p) for p in points[i : i + UPSERT_BATCH]]
            _guarded(partial(self._client.upsert, self._collection, batch, wait=True))

    def delete_version(self, tenant_id: UUID, document_version_id: UUID) -> None:
        flt = tenant_filter(tenant_id, document_version_id=document_version_id)
        _guarded(
            lambda: self._client.delete(
                self._collection, models.FilterSelector(filter=flt), wait=True
            )
        )

    def get_version_points(self, tenant_id: UUID, document_version_id: UUID) -> list[VectorPoint]:
        flt = tenant_filter(tenant_id, document_version_id=document_version_id)
        return [record_to_point(r) for r in self._scroll(flt)]

    # -- internals ---------------------------------------------------------------------------

    def _scroll(self, flt: models.Filter) -> Iterator[Any]:
        offset: Any = None
        while True:
            records, offset = _guarded(partial(self._scroll_page, flt, offset))
            yield from records
            if offset is None:
                return

    def _scroll_page(self, flt: models.Filter, offset: Any) -> Any:
        return self._client.scroll(
            self._collection,
            scroll_filter=flt,
            limit=SCROLL_PAGE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

    def _ensure_collection(self, dimension: int) -> None:
        if not self._client.collection_exists(self._collection):
            try:
                self._client.create_collection(
                    self._collection,
                    vectors_config={
                        DENSE: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
                    },
                    sparse_vectors_config={
                        SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                    },
                )
                log.info("qdrant collection created", extra={"collection": self._collection})
            except UnexpectedResponse as exc:
                if exc.status_code != 409:  # 409: another process created it first
                    raise
        info = self._client.get_collection(self._collection)
        self._check_dimension(info, dimension)
        self._ensure_payload_indexes(info.payload_schema)

    def _check_dimension(self, info: Any, dimension: int) -> None:
        vectors = info.config.params.vectors
        existing = vectors.get(DENSE) if isinstance(vectors, dict) else None
        if existing is None or existing.size != dimension:
            raise ValueError(
                f"collection {self._collection!r} has dense size "
                f"{getattr(existing, 'size', None)}, expected {dimension}; "
                "changing the embedding model needs a new collection and a re-index"
            )

    def _ensure_payload_indexes(self, existing: dict[str, Any]) -> None:
        wanted: dict[str, Any] = {
            "tenant_id": models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD, is_tenant=True
            ),
            **dict.fromkeys(_KEYWORD_FIELDS, models.PayloadSchemaType.KEYWORD),
            **dict.fromkeys(_INTEGER_FIELDS, models.PayloadSchemaType.INTEGER),
        }
        for field, field_schema in wanted.items():
            if field not in existing:
                self._client.create_payload_index(
                    self._collection, field_name=field, field_schema=field_schema, wait=True
                )
