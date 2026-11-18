"""Worker entrypoint: `python -m finrag.jobs.worker` consumes the Procrastinate queue.

The pipeline is built before the first job is picked up, so a misconfigured worker (for example
the Voyage backend with no API key) fails at startup instead of failing every job.
"""

import logging

from procrastinate import App, PsycopgConnector

from finrag.core.config import Settings, get_settings
from finrag.core.logging import configure_logging
from finrag.jobs.queue import build_app, lazy_pipeline
from finrag.storage.db import psycopg_dsn

log = logging.getLogger(__name__)

# Docling's converter is not thread-safe, and one PDF already needs ~1 GB: one job at a time.
WORKER_CONCURRENCY = 1


def run(app: App) -> None:
    log.info("worker started (concurrency=%d)", WORKER_CONCURRENCY)
    app.run_worker(concurrency=WORKER_CONCURRENCY, wait=True)  # opens its own connection


def main(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    provider = lazy_pipeline(settings)
    provider()  # fail fast on bad configuration
    # The worker loop is async, so it needs the async connector (the API enqueues synchronously).
    run(
        build_app(
            settings,
            connector=PsycopgConnector(conninfo=psycopg_dsn(settings)),
            pipeline_provider=provider,
        )
    )


if __name__ == "__main__":
    main()
