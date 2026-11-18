"""Database engine helpers."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from finrag.core.config import Settings


def build_engine(settings: Settings) -> Engine:
    return create_engine(settings.database_url.get_secret_value(), pool_pre_ping=True)


def build_session_factory(engine: Engine) -> sessionmaker:  # type: ignore[type-arg]
    return sessionmaker(engine, expire_on_commit=False)


def psycopg_dsn(settings: Settings) -> str:
    """The SQLAlchemy URL rendered as a plain libpq/psycopg DSN (for Procrastinate)."""
    url = make_url(settings.database_url.get_secret_value()).set(drivername="postgresql")
    return url.render_as_string(hide_password=False)
