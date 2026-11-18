"""Document upload and tracking routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile

from finrag.api.deps import DocumentServiceDep, PrincipalDep
from finrag.ingestion.service import (
    ChunkPage,
    DocumentView,
    EmptyFileError,
    FileTooLargeError,
    UnsupportedMediaTypeError,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", status_code=202, response_model=DocumentView)
def upload_document(
    file: Annotated[UploadFile, File()],
    principal: PrincipalDep,
    service: DocumentServiceDep,
    response: Response,
) -> DocumentView:
    max_bytes = service.max_upload_bytes
    data = file.file.read(max_bytes + 1)  # never buffer more than the limit allows
    try:
        document, created = service.upload(
            principal, file.filename or "unnamed", file.content_type or "", data
        )
    except UnsupportedMediaTypeError as exc:
        raise HTTPException(415, f"Unsupported media type: {exc}. Use PDF or plain text.") from exc
    except FileTooLargeError as exc:
        raise HTTPException(413, f"File exceeds the {max_bytes} byte upload limit.") from exc
    except EmptyFileError as exc:
        raise HTTPException(422, "The uploaded file is empty.") from exc
    if not created:
        response.status_code = 200
    return document


@router.get("", response_model=list[DocumentView])
def list_documents(principal: PrincipalDep, service: DocumentServiceDep) -> list[DocumentView]:
    return service.list_documents(principal)


@router.get("/{document_id}", response_model=DocumentView)
def get_document(
    document_id: UUID, principal: PrincipalDep, service: DocumentServiceDep
) -> DocumentView:
    document = service.get_document(principal, document_id)
    if document is None:
        raise HTTPException(404, "Document not found.")
    return document


@router.get("/{document_id}/chunks", response_model=ChunkPage)
def list_chunks(
    document_id: UUID,
    principal: PrincipalDep,
    service: DocumentServiceDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChunkPage:
    page = service.list_chunks(principal, document_id, limit, offset)
    if page is None:
        raise HTTPException(404, "Document not found.")
    return page
