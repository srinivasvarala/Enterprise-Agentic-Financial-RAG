"""Dependency-free stand-ins for the indexing Protocols.

Selected by the `*_backend` settings (offline smoke runs) and used by unit tests, so no API key,
model download or Qdrant server is needed. They keep everything in process memory, so nothing
survives a restart.
"""

import hashlib
import math
from collections.abc import Sequence
from uuid import UUID

from finrag.core.domain import SearchHit, SparseVector, VectorPoint
from finrag.core.tenancy import require_points_of_tenant, require_tenant


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def _bucket(token: str, modulo: int) -> int:
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % modulo


class HashEmbedder:
    """Deterministic bag-of-hashed-words embedding, L2-normalised."""

    def __init__(self, dimension: int = 32) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed(text)

    def _embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self._dimension
        for token in _tokens(text):
            vector[_bucket(token, self._dimension)] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return tuple(v / norm for v in vector)


class HashSparseEncoder:
    """Term-frequency sparse vectors over hashed vocabulary indices."""

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self._encode(text) for text in texts]

    def encode_query(self, text: str) -> SparseVector:
        return self._encode(text)

    def _encode(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for token in _tokens(text):
            idx = _bucket(token, 2**20)
            counts[idx] = counts.get(idx, 0.0) + 1.0
        ordered = sorted(counts)
        return SparseVector(tuple(ordered), tuple(counts[i] for i in ordered))


class InMemoryVectorStore:
    """Tenant-scoped point store that refuses calls without a tenant id."""

    def __init__(self) -> None:
        self.dimension: int | None = None
        self._points: dict[tuple[UUID, UUID], VectorPoint] = {}  # (tenant_id, point id)

    def ensure_collection(self, dimension: int) -> None:
        self.dimension = dimension

    def upsert(self, tenant_id: UUID, points: Sequence[VectorPoint]) -> None:
        require_tenant(tenant_id)
        require_points_of_tenant(tenant_id, points)
        for point in points:
            self._points[(tenant_id, point.id)] = point

    def delete_version(self, tenant_id: UUID, document_version_id: UUID) -> None:
        require_tenant(tenant_id)
        for key in [
            k
            for k, p in self._points.items()
            if k[0] == tenant_id and p.document_version_id == document_version_id
        ]:
            del self._points[key]

    def get_version_points(self, tenant_id: UUID, document_version_id: UUID) -> list[VectorPoint]:
        require_tenant(tenant_id)
        return [
            p
            for (t, _), p in self._points.items()
            if t == tenant_id and p.document_version_id == document_version_id
        ]

    def search(
        self,
        tenant_id: UUID,
        collection_id: UUID,
        dense: tuple[float, ...],
        sparse: SparseVector,
        *,
        dense_k: int,
        sparse_k: int,
        rrf_k: int,
        limit: int,
    ) -> list[SearchHit]:
        """Mimics the Qdrant adapter: two ranked branches fused by reciprocal rank fusion."""
        require_tenant(tenant_id)
        scope = [
            p
            for (t, _), p in self._points.items()
            if t == tenant_id and p.collection_id == collection_id
        ]
        query_terms = dict(zip(sparse.indices, sparse.values, strict=True))
        by_dense = sorted(
            scope, key=lambda p: -sum(a * b for a, b in zip(dense, p.dense, strict=True))
        )
        by_sparse = sorted(
            (p for p in scope if set(p.sparse.indices) & set(query_terms)),
            key=lambda p: (
                -sum(
                    query_terms.get(i, 0.0) * v
                    for i, v in zip(p.sparse.indices, p.sparse.values, strict=True)
                )
            ),
        )
        fused: dict[UUID, float] = {}
        for branch in (by_dense[:dense_k], by_sparse[:sparse_k]):
            for rank, point in enumerate(branch, start=1):
                fused[point.chunk_id] = fused.get(point.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda item: -item[1])
        return [SearchHit(chunk_id, score) for chunk_id, score in ranked[:limit]]
