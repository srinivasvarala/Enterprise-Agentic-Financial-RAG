"""FastAPI dependencies, including the single place requests are resolved to a Principal."""

from typing import Annotated

from fastapi import Depends, Request

from finrag.core.container import Container, build_container
from finrag.core.domain import Principal
from finrag.ingestion.service import DocumentService


def get_container(request: Request) -> Container:
    """Builds adapters lazily on first use, so the app can start without any services."""
    app = request.app
    if getattr(app.state, "container", None) is None:
        app.state.container = build_container(app.state.settings)
    container: Container = app.state.container
    return container


def current_principal(request: Request) -> Principal:
    """Resolve the caller to a Tenant + Collection. M6 replaces this with API-key auth."""
    settings = request.app.state.settings
    return Principal(
        tenant_id=settings.default_tenant_id, collection_id=settings.default_collection_id
    )


def get_document_service(
    container: Annotated[Container, Depends(get_container)],
) -> DocumentService:
    return DocumentService(
        container.session_factory, container.blob_store, container.job_queue, container.settings
    )


PrincipalDep = Annotated[Principal, Depends(current_principal)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
