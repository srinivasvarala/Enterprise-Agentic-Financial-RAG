"""FastAPI application factory."""

from fastapi import FastAPI

from finrag import __version__
from finrag.api.routes import documents, health, query
from finrag.core.config import Settings, get_settings
from finrag.core.container import Container
from finrag.core.logging import configure_logging


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Build the app. Adapters are wired lazily from Settings unless a container is given."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title="FinRAG", version=__version__)
    app.state.settings = settings
    app.state.container = container
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    return app


app = create_app()
