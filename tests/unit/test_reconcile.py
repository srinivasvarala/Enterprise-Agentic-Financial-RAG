"""Seam 1: the reconciler re-defers Documents a crashed worker or a failed enqueue left behind."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from harness import Env
from procrastinate.testing import InMemoryConnector
from sqlalchemy import update

from finrag.core.domain import DocumentStatus
from finrag.jobs.queue import INGEST_TASK, ProcrastinateJobQueue, build_app
from finrag.jobs.reconcile import reconcile_stuck_documents
from finrag.storage.models import Document

STUCK_AFTER = 900


def upload(env: Env, text: bytes = b"Revenue grew in fiscal 2025.") -> str:
    resp = env.client.post("/documents", files={"file": ("n.txt", text, "text/plain")})
    doc_id: str = resp.json()["id"]
    return doc_id


def age(env: Env, doc_id: str, status: DocumentStatus, seconds: int) -> None:
    when = datetime.now(UTC) - timedelta(seconds=seconds)
    with env.container.session_factory() as session:
        session.execute(
            update(Document)
            .where(Document.id == UUID(doc_id))
            .values(status=status.value, updated_at=when)
        )
        session.commit()


def reconcile(env: Env, queue: Any = None) -> list[Any]:
    return reconcile_stuck_documents(env.container.session_factory, queue or env.jobs, STUCK_AFTER)


def test_a_document_stuck_mid_stage_is_redeferred_and_recovers_to_ready(env: Env) -> None:
    doc_id = upload(env)
    env.run_jobs()  # the original job ran; now simulate a crash that left it in embedding
    age(env, doc_id, DocumentStatus.EMBEDDING, STUCK_AFTER + 60)

    redeferred = reconcile(env)

    assert [str(j.document_id) for j in redeferred] == [doc_id]
    env.run_jobs()
    assert env.client.get(f"/documents/{doc_id}").json()["status"] == "ready"


def test_a_queued_document_whose_enqueue_failed_is_redeferred(env: Env) -> None:
    doc_id = upload(env)
    env.jobs.jobs.clear()  # the enqueue after commit was lost
    age(env, doc_id, DocumentStatus.QUEUED, STUCK_AFTER + 60)

    assert [str(j.document_id) for j in reconcile(env)] == [doc_id]


def test_recent_and_terminal_documents_are_left_alone(env: Env) -> None:
    fresh = upload(env, b"fresh")
    ready = upload(env, b"ready")
    failed = upload(env, b"failed")
    age(env, fresh, DocumentStatus.PARSING, 10)
    age(env, ready, DocumentStatus.READY, STUCK_AFTER + 60)
    age(env, failed, DocumentStatus.FAILED, STUCK_AFTER + 60)
    env.jobs.jobs.clear()

    assert reconcile(env) == []
    assert env.jobs.jobs == []


def test_each_job_carries_its_own_documents_tenant(env: Env) -> None:
    mine = upload(env, b"mine")
    other = env.other_tenant()
    env.act_as(other)
    theirs = upload(env, b"theirs")
    for doc_id in (mine, theirs):
        age(env, doc_id, DocumentStatus.CHUNKING, STUCK_AFTER + 60)
    env.jobs.jobs.clear()

    jobs = {str(j.document_id): j.tenant_id for j in reconcile(env)}

    assert jobs == {mine: env.settings.default_tenant_id, theirs: other.tenant_id}


def test_a_document_that_already_has_a_queued_job_is_not_enqueued_twice(env: Env) -> None:
    doc_id = upload(env)
    age(env, doc_id, DocumentStatus.QUEUED, STUCK_AFTER + 60)
    connector = InMemoryConnector()
    queue = ProcrastinateJobQueue(build_app(env.settings, connector=connector))

    first = reconcile(env, queue)
    second = reconcile(env, queue)

    assert len(first) == 1
    assert second == []
    assert [j["task_name"] for j in connector.jobs.values()] == [INGEST_TASK]
    queue.close()
