"""Seam 1: the documents HTTP API through create_app, with in-memory fakes."""

import hashlib

from harness import Env
from httpx import Response

PDF = b"%PDF-1.7 fake filing"
TXT = b"Item 7. Management's Discussion and Analysis"


def upload(
    env: Env, data: bytes = TXT, filename: str = "notes.txt", media_type: str = "text/plain"
) -> Response:
    return env.client.post("/documents", files={"file": (filename, data, media_type)})


def test_upload_text_returns_202_with_queued_document(env: Env) -> None:
    response = upload(env)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["filename"] == "notes.txt"
    assert body["media_type"] == "text/plain"
    assert body["size_bytes"] == len(TXT)
    assert body["failure_reason"] is None
    assert body["chunk_count"] == 0
    assert body["id"] and body["created_at"] and body["updated_at"]


def test_upload_pdf_is_accepted(env: Env) -> None:
    response = upload(env, PDF, "10-K.pdf", "application/pdf")

    assert response.status_code == 202
    assert response.json()["media_type"] == "application/pdf"


def test_media_type_parameters_are_ignored(env: Env) -> None:
    assert upload(env, media_type="text/plain; charset=utf-8").status_code == 202


def test_unsupported_media_type_is_415(env: Env) -> None:
    response = upload(env, b"PK", "sheet.xlsx", "application/vnd.ms-excel")

    assert response.status_code == 415
    assert env.client.get("/documents").json() == []


def test_over_size_limit_is_413(env: Env) -> None:
    response = upload(env, b"x" * (env.settings.max_upload_bytes + 1))

    assert response.status_code == 413
    assert env.blobs.objects == {}


def test_exactly_at_size_limit_is_accepted(env: Env) -> None:
    assert upload(env, b"x" * env.settings.max_upload_bytes).status_code == 202


def test_empty_document_is_422(env: Env) -> None:
    assert upload(env, b"").status_code == 422


def test_original_bytes_are_stored_unchanged(env: Env) -> None:
    upload(env, PDF, "10-K.pdf", "application/pdf")

    assert list(env.blobs.objects.values()) == [PDF]


def test_new_document_enqueues_one_ingestion_job(env: Env) -> None:
    doc_id = upload(env).json()["id"]

    assert len(env.jobs.jobs) == 1
    job = env.jobs.jobs[0]
    assert str(job.document_id) == doc_id
    assert job.tenant_id == env.settings.default_tenant_id


def test_identical_content_returns_existing_document_without_side_effects(env: Env) -> None:
    first = upload(env)
    second = upload(env, filename="renamed.txt")

    assert second.status_code == 200
    assert second.json() == first.json()
    assert len(env.client.get("/documents").json()) == 1
    assert len(env.blobs.objects) == 1
    assert len(env.jobs.jobs) == 1


def test_different_content_creates_a_second_document(env: Env) -> None:
    upload(env, b"one")
    second = upload(env, b"two")

    assert second.status_code == 202
    assert len(env.client.get("/documents").json()) == 2


def test_identical_content_in_another_tenant_is_not_a_duplicate(env: Env) -> None:
    first = upload(env).json()
    env.act_as(env.other_tenant())

    second = upload(env)

    assert second.status_code == 202
    assert second.json()["id"] != first["id"]
    assert len(env.jobs.jobs) == 2


def test_list_returns_the_principals_documents(env: Env) -> None:
    a = upload(env, b"one").json()
    b = upload(env, b"two").json()

    listed = env.client.get("/documents")

    assert listed.status_code == 200
    assert {d["id"] for d in listed.json()} == {a["id"], b["id"]}
    assert set(listed.json()[0]) >= {
        "id", "filename", "media_type", "size_bytes", "status",
        "failure_reason", "chunk_count", "created_at", "updated_at",
    }  # fmt: skip


def test_get_returns_one_document(env: Env) -> None:
    created = upload(env).json()

    response = env.client.get(f"/documents/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


def test_get_unknown_document_is_404(env: Env) -> None:
    assert env.client.get("/documents/6f1c1f0e-4c3b-4a53-9d55-0e6e9d1f8a11").status_code == 404


def test_other_tenants_document_is_404_never_403(env: Env) -> None:
    created = upload(env).json()
    env.act_as(env.other_tenant())

    assert env.client.get(f"/documents/{created['id']}").status_code == 404
    assert env.client.get("/documents").json() == []


def test_documents_belong_to_default_collection(env: Env) -> None:
    created = upload(env).json()

    assert created["collection_id"] == str(env.settings.default_collection_id)


def test_blob_key_is_content_addressed_per_tenant(env: Env) -> None:
    upload(env)

    (key,) = env.blobs.objects
    assert key == f"{env.settings.default_tenant_id}/{hashlib.sha256(TXT).hexdigest()}"
