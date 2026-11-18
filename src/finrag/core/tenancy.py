"""Tenant guards shared by every VectorStore implementation."""

from collections.abc import Iterable
from uuid import UUID

from finrag.core.domain import VectorPoint
from finrag.core.errors import TenantRequiredError


def require_tenant(tenant_id: UUID | None) -> UUID:
    if tenant_id is None:
        raise TenantRequiredError("tenant_id is required on every vector store call")
    return tenant_id


def require_points_of_tenant(tenant_id: UUID, points: Iterable[VectorPoint]) -> None:
    """Reject a write that carries a point of another Tenant."""
    for point in points:
        if point.tenant_id != tenant_id:
            raise TenantRequiredError("point tenant does not match the write's tenant")
