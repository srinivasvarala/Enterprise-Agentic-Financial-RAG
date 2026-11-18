"""Seam 1 against real Postgres, SeaweedFS (S3) and Procrastinate. Needs `make up`."""

import hashlib
import os
from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from finrag.core.config import Settings
from finrag.core.container import Container
from finrag.storage.models import Base, Collection, Tenant

pytestmark = pytest.mark.integration


def upload(client: TestClient, data: bytes, media_type: str = "text/plain"):
    return client.post("/documents", files={"file": ("filing.txt", data, media_type)})


def unique_bytes() -> bytes:
    return f"Item 1A. Risk Factors {uuid4()}".encode()


def test_migrations_match_models(container: Container) -> None:
    with container.session_factory() as session:
        context = MigrationContext.configure(
            session.connection(),
            opts={
                "include_object": lambda obj, name, type_, reflected, compare_to: (
                    not (
                        (type_ == "table" and (name or "").startswith("procrastinate_"))
                        or (type_ == "index" and (name or "").startswith("procrastinate_"))
                    )
                )
            },
        )
        assert compare_metadata(context, Base.metadata) == []


def test_seed_creates_default_tenant_and_collection(
    container: Container, settings: Settings
) -> None:
    with container.session_factory() as session:
        assert session.get(Tenant, settings.default_tenant_id) is not None
        collection = session.get(Collection, settings.default_collection_id)
        assert collection is not None
        assert collection.tenant_id == settings.default_tenant_id


def test_upload_stores_original_bytes_in_s3(
    client: TestClient, container: Container, settings: Settings
) -> None:
    data = os.urandom(512)  # arbitrary bytes must round-trip unchanged
    response = upload(client, data, "application/pdf")

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    key = f"{settings.default_tenant_id}/{hashlib.sha256(data).hexdigest()}"
    assert container.blob_store.get(key) == data


def test_upload_enqueues_a_procrastinate_job(client: TestClient, container: Container) -> None:
    doc_id = upload(client, unique_bytes()).json()["id"]

    with container.session_factory() as session:
        count = session.execute(
            text("SELECT count(*) FROM procrastinate_jobs WHERE args->>'document_id' = :id"),
            {"id": doc_id},
        ).scalar_one()
    assert count == 1


def test_duplicate_upload_returns_existing_without_new_job(
    client: TestClient, container: Container
) -> None:
    data = unique_bytes()
    first = upload(client, data)
    second = upload(client, data)

    assert (first.status_code, second.status_code) == (202, 200)
    assert second.json() == first.json()
    with container.session_factory() as session:
        jobs = session.execute(
            text("SELECT count(*) FROM procrastinate_jobs WHERE args->>'document_id' = :id"),
            {"id": first.json()["id"]},
        ).scalar_one()
    assert jobs == 1


def test_list_and_get_round_trip(client: TestClient) -> None:
    created = upload(client, unique_bytes()).json()

    assert client.get(f"/documents/{created['id']}").json() == created
    assert created["id"] in {d["id"] for d in client.get("/documents").json()}


def test_documents_carry_tenant_and_collection_ids(
    client: TestClient, container: Container, settings: Settings
) -> None:
    from finrag.storage.models import Document

    created = upload(client, unique_bytes()).json()

    with container.session_factory() as session:
        row = session.scalars(select(Document).where(Document.id == created["id"])).one()
    assert row.tenant_id == settings.default_tenant_id
    assert row.collection_id == settings.default_collection_id


def test_unknown_document_is_404(client: TestClient) -> None:
    assert client.get(f"/documents/{uuid4()}").status_code == 404


def test_validation_errors(client: TestClient, settings: Settings) -> None:
    assert upload(client, b"x", "application/zip").status_code == 415
    assert upload(client, b"x" * (settings.max_upload_bytes + 1)).status_code == 413
    assert upload(client, b"").status_code == 422
