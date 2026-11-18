"""POST /query against the real stack (Qdrant, Postgres, S3) with fake models."""

import json
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import select

from finrag.api.deps import current_principal
from finrag.core.container import Container
from finrag.core.domain import IngestionJob, Principal
from finrag.storage.models import DocumentVersion

pytestmark = pytest.mark.integration


def ingest(client: TestClient, container: Container, text: str) -> UUID:
    response = client.post("/documents", files={"file": ("f.txt", text.encode(), "text/plain")})
    document_id = UUID(response.json()["id"])
    with container.session_factory() as session:
        version = session.scalars(
            select(DocumentVersion).where(DocumentVersion.document_id == document_id)
        ).one()
    container.ingestion_pipeline().run(
        IngestionJob(version.tenant_id, version.document_id, version.id), job_id="integration"
    )
    return document_id


def answer_of(response: Response) -> dict[str, Any]:
    frames = [f for f in response.text.strip().split("\n\n") if f.startswith("event: answer")]
    assert len(frames) == 1
    data: dict[str, Any] = json.loads(frames[0].split("data: ", 1)[1])
    return data


def test_a_question_is_answered_from_an_ingested_document_with_a_citation_to_it(
    client: TestClient, container: Container
) -> None:
    marker = uuid4().hex
    document_id = ingest(
        client, container, f"Treasury{marker} holdings were $42 million at year end."
    )

    answer = answer_of(
        client.post("/query", json={"question": f"What were Treasury{marker} holdings"})
    )

    assert answer["refusal_reason"] is None
    assert str(document_id) == answer["citations"][0]["document_id"]
    assert "$42 million" in answer["text"]


def test_another_tenant_cannot_get_an_answer_from_this_tenants_document(
    client: TestClient, container: Container
) -> None:
    marker = uuid4().hex
    ingest(client, container, f"Treasury{marker} holdings were $42 million at year end.")
    stranger = Principal(tenant_id=uuid4(), collection_id=uuid4())
    client.app.dependency_overrides[current_principal] = lambda: stranger  # type: ignore[attr-defined]
    try:
        answer = answer_of(
            client.post("/query", json={"question": f"What were Treasury{marker} holdings"})
        )
    finally:
        client.app.dependency_overrides.pop(current_principal)  # type: ignore[attr-defined]

    assert answer["refusal_reason"] == "insufficient_evidence"
    assert answer["citations"] == []
