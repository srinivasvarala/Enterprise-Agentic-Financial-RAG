"""FastAPI application factory."""

from fastapi import FastAPI

from finrag import __version__
from finrag.api.routes import health
from finrag.core.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="FinRAG", version=__version__)
    app.state.settings = settings
    app.include_router(health.router)
    return app


app = create_app()
