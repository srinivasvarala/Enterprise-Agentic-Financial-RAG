"""Dependency-free stand-ins for the indexing Protocols.

Used by unit tests and wired in the composition root until the real Voyage / FastEmbed / Qdrant
adapters land. They keep everything in process memory, so nothing survives a restart.
"""

import hashlib
import math
from collections.abc import Sequence
from uuid import UUID

from finrag.core.domain import SparseVector, VectorPoint
from finrag.core.errors import TenantRequiredError


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

    def _embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self._dimension
        for token in _tokens(text):
            vector[_bucket(token, self._dimension)] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return tuple(v / norm for v in vector)


class HashSparseEncoder:
    """Term-frequency sparse vectors over hashed vocabulary indices."""

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        out: list[SparseVector] = []
        for text in texts:
            counts: dict[int, float] = {}
            for token in _tokens(text):
                idx = _bucket(token, 2**20)
                counts[idx] = counts.get(idx, 0.0) + 1.0
            ordered = sorted(counts)
            out.append(SparseVector(tuple(ordered), tuple(counts[i] for i in ordered)))
        return out


class InMemoryVectorStore:
    """Tenant-scoped point store that refuses calls without a tenant id."""

    def __init__(self) -> None:
        self.dimension: int | None = None
        self._points: dict[tuple[UUID, UUID], VectorPoint] = {}  # (tenant_id, point id)

    def ensure_collection(self, dimension: int) -> None:
        self.dimension = dimension

    def upsert(self, tenant_id: UUID, points: Sequence[VectorPoint]) -> None:
        _require(tenant_id)
        for point in points:
            if point.tenant_id != tenant_id:
                raise TenantRequiredError("point tenant does not match the write's tenant")
            self._points[(tenant_id, point.id)] = point

    def delete_version(self, tenant_id: UUID, document_version_id: UUID) -> None:
        _require(tenant_id)
        for key in [
            k
            for k, p in self._points.items()
            if k[0] == tenant_id and p.document_version_id == document_version_id
        ]:
            del self._points[key]

    def get_version_points(self, tenant_id: UUID, document_version_id: UUID) -> list[VectorPoint]:
        _require(tenant_id)
        return [
            p
            for (t, _), p in self._points.items()
            if t == tenant_id and p.document_version_id == document_version_id
        ]


def _require(tenant_id: UUID | None) -> None:
    if tenant_id is None:
        raise TenantRequiredError("tenant_id is required on every vector store call")
