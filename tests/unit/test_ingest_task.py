"""The Procrastinate task runs the pipeline, driven through an in-memory connector."""

from harness import Env
from procrastinate.testing import InMemoryConnector

from finrag.jobs.queue import INGEST_TASK, build_app


def test_deferred_task_runs_the_pipeline_with_the_procrastinate_job_id(env: Env) -> None:
    doc_id = env.client.post(
        "/documents", files={"file": ("n.txt", b"Hello filing.", "text/plain")}
    ).json()["id"]
    (job,) = env.jobs.jobs
    connector = InMemoryConnector()
    app = build_app(env.settings, connector=connector, pipeline_provider=env.pipeline)

    with app.open():
        app.configure_task(INGEST_TASK).defer(
            tenant_id=str(job.tenant_id),
            document_id=str(job.document_id),
            document_version_id=str(job.document_version_id),
        )
        app.run_worker(wait=False)

    assert env.client.get(f"/documents/{doc_id}").json()["status"] == "ready"
    assert [j["status"] for j in connector.jobs.values()] == ["succeeded"]
