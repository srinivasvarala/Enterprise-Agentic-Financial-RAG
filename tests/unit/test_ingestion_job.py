"""Seam 1: upload through the API, run the ingestion job, observe via the API (in-memory fakes)."""

import json
import logging
from typing import Any
from uuid import UUID, uuid4, uuid5

import pytest
from harness import Env
from httpx import Response
from sqlalchemy import select

from finrag.core.domain import ParsedDocument
from finrag.core.errors import TenantRequiredError, TransientError
from finrag.core.logging import JsonFormatter
from finrag.storage.models import Chunk

TEXT = (
    b"Liquidity and capital resources improved in fiscal 2025.\n\n"
    b"Cash from operations was strong.\n\n"
    b"Capital expenditures were stable."
)


def upload(env: Env, data: bytes = TEXT, filename: str = "notes.txt") -> Response:
    return env.client.post("/documents", files={"file": (filename, data, "text/plain")})


def doc(env: Env, doc_id: str) -> dict[str, Any]:
    body: dict[str, Any] = env.client.get(f"/documents/{doc_id}").json()
    return body


def chunks(env: Env, doc_id: str, **params: int) -> dict[str, Any]:
    body: dict[str, Any] = env.client.get(f"/documents/{doc_id}/chunks", params=params).json()
    return body


def long_body(paragraphs: int, words: int) -> bytes:
    return b"\n\n".join(f"P{i} ".encode() + b"word " * words for i in range(paragraphs))


def test_text_document_goes_to_ready_with_chunks_in_reading_order(env: Env) -> None:
    doc_id = upload(env).json()["id"]

    env.run_jobs()

    document = doc(env, doc_id)
    assert document["status"] == "ready"
    assert document["failure_reason"] is None
    page = chunks(env, doc_id)
    items = page["items"]
    assert document["chunk_count"] == len(items) == page["total"] >= 1
    assert [c["ordinal"] for c in items] == list(range(len(items)))
    assert "\n\n".join(c["text"] for c in items) == TEXT.decode()
    first = items[0]
    assert first["kind"] == "text"
    assert first["page_start"] == first["page_end"] == 1
    assert first["token_count"] > 0
    assert first["section_path"] == []


def test_document_passes_through_every_stage_in_order(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="finrag")
    doc_id = upload(env).json()["id"]
    seen: list[str] = []
    env.vectors.before_upsert = lambda: seen.append(doc(env, doc_id)["status"])

    env.run_jobs()

    transitions = [r.status for r in caplog.records if hasattr(r, "status")]  # type: ignore[attr-defined]
    assert transitions == ["parsing", "chunking", "embedding", "ready"]
    assert seen == ["embedding"]


def test_logs_carry_tenant_document_and_job_ids_at_every_stage(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="finrag")
    doc_id = upload(env).json()["id"]

    env.run_jobs()

    stage_logs = [r for r in caplog.records if r.name == "finrag.ingestion.pipeline"]
    assert len(stage_logs) >= 4
    for record in stage_logs:
        assert record.tenant_id == str(env.settings.default_tenant_id)  # type: ignore[attr-defined]
        assert record.document_id == doc_id  # type: ignore[attr-defined]
        assert record.job_id == "test-job-0"  # type: ignore[attr-defined]


def test_json_formatter_emits_extras_as_json_fields() -> None:
    record = logging.LogRecord("finrag.x", logging.INFO, "f", 1, "hello %s", ("w",), None)
    record.tenant_id = "t-1"

    parsed = json.loads(JsonFormatter().format(record))

    assert {k: v for k, v in parsed.items() if k != "ts"} == {
        "level": "INFO",
        "logger": "finrag.x",
        "event": "hello w",
        "tenant_id": "t-1",
    }


def test_chunks_are_persisted_before_points_are_upserted(env: Env) -> None:
    doc_id = upload(env).json()["id"]
    visible: list[int] = []
    env.vectors.before_upsert = lambda: visible.append(chunks(env, doc_id)["total"])

    env.run_jobs()

    assert visible == [doc(env, doc_id)["chunk_count"]]
    assert visible[0] > 0


def test_chunk_count_shows_in_document_list_and_detail(env: Env) -> None:
    doc_id = upload(env).json()["id"]
    env.run_jobs()

    listed = {d["id"]: d for d in env.client.get("/documents").json()}
    assert listed[doc_id]["chunk_count"] == doc(env, doc_id)["chunk_count"] >= 1


@pytest.mark.parametrize(
    ("data", "reason_part"),
    [
        (b"\xff\xfe\x00bad", "UTF-8"),
        (b"abc\x00def", "binary"),
        (b"   \n\n  ", "no text"),
    ],
)
def test_parse_errors_fail_immediately_with_a_readable_reason(
    env: Env, data: bytes, reason_part: str
) -> None:
    doc_id = upload(env, data).json()["id"]

    env.run_jobs()

    document = doc(env, doc_id)
    assert document["status"] == "failed"
    assert reason_part in document["failure_reason"]
    assert document["chunk_count"] == 0
    assert chunks(env, doc_id)["items"] == []
    assert env.embedder.calls == 0
    assert env.sleeps == []


def test_pdf_without_a_parser_fails_with_a_reason(env: Env) -> None:
    resp = env.client.post("/documents", files={"file": ("a.pdf", b"%PDF-1.7", "application/pdf")})

    env.run_jobs()

    document = doc(env, resp.json()["id"])
    assert document["status"] == "failed"
    assert "application/pdf" in document["failure_reason"]


def test_transient_errors_retry_with_backoff_then_succeed(env: Env) -> None:
    env.embedder.failures, env.embedder.error = 2, TransientError("429 rate limited")
    doc_id = upload(env).json()["id"]

    env.run_jobs()

    assert doc(env, doc_id)["status"] == "ready"
    assert env.embedder.calls == 3
    assert env.sleeps == [2.0, 4.0]  # base 2s, doubling


def test_transient_errors_beyond_the_limit_fail_with_a_generic_reason(env: Env) -> None:
    env.embedder.failures, env.embedder.error = 99, TransientError("503 unavailable")
    doc_id = upload(env).json()["id"]
    (job,) = env.run_jobs()

    document = doc(env, doc_id)
    assert document["status"] == "failed"
    assert "embedding" in document["failure_reason"]
    assert "3 attempts" in document["failure_reason"]
    assert "503 unavailable" not in document["failure_reason"]
    assert env.embedder.calls == env.settings.ingest_max_attempts
    assert env.vectors.get_version_points(job.tenant_id, job.document_version_id) == []


def test_provider_error_details_are_logged_but_never_shown_to_the_admin(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="finrag")
    leak = "connect to qdrant.internal.corp:6333 failed"
    env.embedder.failures, env.embedder.error = 99, TransientError(leak)
    doc_id = upload(env).json()["id"]
    (job,) = env.run_jobs()

    body = env.client.get(f"/documents/{doc_id}").text
    assert "qdrant.internal.corp" not in body
    detail = [r for r in caplog.records if "qdrant.internal.corp" in str(getattr(r, "error", ""))]
    assert detail
    assert all(
        getattr(r, "tenant_id", None) == str(job.tenant_id)
        and getattr(r, "document_id", None) == str(job.document_id)
        and getattr(r, "job_id", None) == "test-job-0"
        for r in detail
    )


class _FlakyParser:
    def parse(self, data: bytes) -> ParsedDocument:
        raise TransientError("blob host s3.internal unreachable")


def test_a_transient_error_that_was_never_retried_does_not_claim_attempts(env: Env) -> None:
    env.container.parsers = {"text/plain": _FlakyParser()}
    doc_id = upload(env).json()["id"]

    env.run_jobs()

    document = doc(env, doc_id)
    assert document["status"] == "failed"
    assert "attempts" not in document["failure_reason"]
    assert "s3.internal" not in document["failure_reason"]
    assert "parsing" in document["failure_reason"]


def test_unexpected_errors_fail_without_leaking_internals(env: Env) -> None:
    env.embedder.failures, env.embedder.error = 1, KeyError("secret internals")
    doc_id = upload(env).json()["id"]

    env.run_jobs()

    document = doc(env, doc_id)
    assert document["status"] == "failed"
    assert "secret" not in document["failure_reason"]
    assert "embedding" in document["failure_reason"]


def test_rerunning_a_job_replaces_chunks_and_points_without_duplicates(env: Env) -> None:
    doc_id = upload(env, long_body(6, 300)).json()["id"]
    (job,) = env.run_jobs()
    first = chunks(env, doc_id)
    assert first["total"] > 1

    env.pipeline().run(job)

    assert chunks(env, doc_id) == first  # identical ids, ordinals and text
    assert doc(env, doc_id)["chunk_count"] == first["total"]
    points = env.vectors.get_version_points(job.tenant_id, job.document_version_id)
    # Point ids follow the documented formula, not the code under test.
    assert {p.id for p in points} == {
        uuid5(job.document_version_id, str(i)) for i in range(first["total"])
    }
    assert len(points) == first["total"]
    assert {UUID(c["id"]) for c in first["items"]} == {p.id for p in points}


def test_points_carry_dense_and_sparse_vectors_and_tenant_payload(env: Env) -> None:
    upload(env)
    (job,) = env.run_jobs()

    points = env.vectors.get_version_points(job.tenant_id, job.document_version_id)
    assert points
    for p in points:
        assert len(p.dense) == env.embedder.dimension
        assert p.sparse.indices
        assert p.tenant_id == env.settings.default_tenant_id
        assert p.collection_id == env.settings.default_collection_id
        assert (p.document_id, p.document_version_id) == (job.document_id, job.document_version_id)


def test_chunks_record_tenant_collection_document_and_version(env: Env) -> None:
    doc_id = upload(env).json()["id"]
    (job,) = env.run_jobs()

    with env.container.session_factory() as session:
        rows = session.scalars(select(Chunk)).all()
    assert rows
    for row in rows:
        assert row.tenant_id == env.settings.default_tenant_id
        assert row.collection_id == env.settings.default_collection_id
        assert str(row.document_id) == doc_id
        assert row.document_version_id == job.document_version_id


def test_chunks_of_another_tenants_document_are_404(env: Env) -> None:
    doc_id = upload(env).json()["id"]
    env.run_jobs()
    env.act_as(env.other_tenant())

    assert env.client.get(f"/documents/{doc_id}/chunks").status_code == 404
    assert env.client.get(f"/documents/{uuid4()}/chunks").status_code == 404


def test_chunks_are_paginated_in_ordinal_order(env: Env) -> None:
    doc_id = upload(env, long_body(5, 500)).json()["id"]
    env.run_jobs()

    everything = chunks(env, doc_id)["items"]
    assert len(everything) >= 4
    page = chunks(env, doc_id, limit=2, offset=1)
    assert page["items"] == everything[1:3]
    assert (page["total"], page["limit"], page["offset"]) == (len(everything), 2, 1)
    assert chunks(env, doc_id, offset=len(everything))["items"] == []
    assert env.client.get(f"/documents/{doc_id}/chunks?limit=0").status_code == 422


def test_vector_store_rejects_calls_without_a_tenant(env: Env) -> None:
    with pytest.raises(TenantRequiredError):
        env.vectors.get_version_points(None, uuid4())  # type: ignore[arg-type]
    with pytest.raises(TenantRequiredError):
        env.vectors.upsert(None, [])  # type: ignore[arg-type]
    with pytest.raises(TenantRequiredError):
        env.vectors.delete_version(None, uuid4())  # type: ignore[arg-type]
