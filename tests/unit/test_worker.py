"""Worker entrypoint: single-job concurrency and fail-fast startup."""

from typing import Any

import pytest
from procrastinate import PsycopgConnector

from finrag.core.config import Settings
from finrag.jobs import worker


class _StubApp:
    def __init__(self) -> None:
        self.worker_kwargs: dict[str, Any] | None = None

    def run_worker(self, **kwargs: Any) -> None:
        self.worker_kwargs = kwargs


def test_worker_runs_one_job_at_a_time_and_waits_for_jobs() -> None:
    app = _StubApp()

    worker.run(app)  # type: ignore[arg-type]

    assert app.worker_kwargs == {"concurrency": 1, "wait": True}


def test_worker_fails_at_startup_when_the_pipeline_cannot_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(_: Settings) -> Any:
        def provide() -> Any:
            raise RuntimeError("FINRAG_VOYAGE_API_KEY is not set")

        return provide

    monkeypatch.setattr(worker, "lazy_pipeline", broken)
    monkeypatch.setattr(worker, "run", lambda app: pytest.fail("worker must not start"))

    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        worker.main(Settings(_env_file=None))  # type: ignore[call-arg]


def test_worker_uses_the_async_connector_its_loop_requires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A sync connector fails at run time ("needs an asynchronous connector"), which the
    # in-memory connector used by the task tests cannot reveal.
    seen: dict[str, Any] = {}

    def spy_build_app(settings: Settings, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(worker, "lazy_pipeline", lambda _: lambda: object())
    monkeypatch.setattr(worker, "build_app", spy_build_app)
    monkeypatch.setattr(worker, "run", lambda app: None)

    worker.main(Settings(_env_file=None))  # type: ignore[call-arg]

    assert isinstance(seen["connector"], PsycopgConnector)
