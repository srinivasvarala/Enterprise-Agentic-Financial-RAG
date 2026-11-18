"""Seam 1 against real Postgres, SeaweedFS (S3) and Procrastinate. Needs `make up`."""

import hashlib
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import select, text, update

from finrag.core.config import Settings
from finrag.core.container import Container
from finrag.core.domain import IngestionJob
from finrag.jobs.reconcile import reconcile_stuck_documents
from finrag.storage.models import Base, Collection, Document, DocumentVersion, Tenant

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


def run_ingestion(container: Container, doc_id: str) -> IngestionJob:
    """Run the ingestion job synchronously against real Postgres and S3."""
    with container.session_factory() as session:
        version = session.scalars(
            select(DocumentVersion).where(DocumentVersion.document_id == UUID(doc_id))
        ).one()
    job = IngestionJob(version.tenant_id, version.document_id, version.id)
    container.ingestion_pipeline().run(job, job_id="integration")
    return job


def test_text_document_is_ingested_to_ready_on_real_postgres_and_s3(
    client: TestClient, container: Container
) -> None:
    body = "\n\n".join(f"Paragraph {i} {uuid4()} " + "word " * 400 for i in range(4))
    doc_id = upload(client, body.encode()).json()["id"]

    job = run_ingestion(container, doc_id)

    document = client.get(f"/documents/{doc_id}").json()
    page = client.get(f"/documents/{doc_id}/chunks", params={"limit": 2}).json()
    assert document["status"] == "ready"
    assert document["chunk_count"] == page["total"] > 2
    assert [c["ordinal"] for c in page["items"]] == [0, 1]
    store = container.vector_store
    assert len(store.get_version_points(job.tenant_id, job.document_version_id)) == page["total"]

    run_ingestion(container, doc_id)  # re-run: replaces, never duplicates
    assert client.get(f"/documents/{doc_id}/chunks").json()["total"] == page["total"]
    assert len(store.get_version_points(job.tenant_id, job.document_version_id)) == page["total"]


def test_corrupt_text_fails_with_a_reason_on_real_postgres(
    client: TestClient, container: Container
) -> None:
    doc_id = upload(client, b"\xff\xfe\x00" + uuid4().bytes).json()["id"]

    run_ingestion(container, doc_id)

    document = client.get(f"/documents/{doc_id}").json()
    assert document["status"] == "failed"
    assert document["failure_reason"]


def test_chunks_of_an_unknown_document_are_404(client: TestClient) -> None:
    assert client.get(f"/documents/{uuid4()}/chunks").status_code == 404


def test_validation_errors(client: TestClient, settings: Settings) -> None:
    assert upload(client, b"x", "application/zip").status_code == 415
    assert upload(client, b"x" * (settings.max_upload_bytes + 1)).status_code == 413
    assert upload(client, b"").status_code == 422


def _jobs_for(container: Container, version_id: UUID) -> int:
    with container.session_factory() as session:
        return session.scalar(
            text("SELECT count(*) FROM procrastinate_jobs WHERE args->>'document_version_id' = :v"),
            {"v": str(version_id)},
        )


def test_reconciler_redefers_a_lost_job_once_and_never_duplicates_a_queued_one(
    client: TestClient, container: Container, settings: Settings
) -> None:
    doc_id = upload(client, unique_bytes()).json()["id"]
    with container.session_factory() as session:
        version = session.scalars(
            select(DocumentVersion).where(DocumentVersion.document_id == UUID(doc_id))
        ).one()
        old = datetime.now(UTC) - timedelta(seconds=settings.stuck_document_seconds + 60)
        session.execute(update(Document).where(Document.id == UUID(doc_id)).values(updated_at=old))
        session.commit()

    def reconcile() -> list[IngestionJob]:
        jobs = reconcile_stuck_documents(
            container.session_factory, container.job_queue, settings.stuck_document_seconds
        )
        return [j for j in jobs if j.document_id == version.document_id]

    assert _jobs_for(container, version.id) == 1  # the upload's own job is still waiting
    assert reconcile() == []  # so nothing is enqueued on top of it
    assert _jobs_for(container, version.id) == 1

    with container.session_factory() as session:  # the job is lost (e.g. the enqueue failed)
        session.execute(
            text("DELETE FROM procrastinate_jobs WHERE args->>'document_version_id' = :v"),
            {"v": str(version.id)},
        )
        session.commit()
    redeferred = reconcile()
    assert [(j.tenant_id, j.document_version_id) for j in redeferred] == [
        (version.tenant_id, version.id)
    ]
    assert reconcile() == []
    assert _jobs_for(container, version.id) == 1
