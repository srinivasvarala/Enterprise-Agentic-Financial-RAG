"""Alembic environment. The database URL comes from Settings, never from alembic.ini."""

from alembic import context
from sqlalchemy import Connection

from finrag.core.config import get_settings
from finrag.storage.db import build_engine
from finrag.storage.models import Base

config = context.config
target_metadata = Base.metadata


def _include_object(obj: object, name: str | None, type_: str, *_: object) -> bool:
    # Procrastinate's tables are managed by its own schema (see migration 0003).
    return not (name or "").startswith("procrastinate_")


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection, target_metadata=target_metadata, include_object=_include_object
    )
    with context.begin_transaction():
        context.run_migrations()


# Callers (tests, run_migrations) may hand in a connection; otherwise build one from Settings.
supplied: Connection | None = config.attributes.get("connection")
if supplied is not None:
    _run(supplied)
else:
    with build_engine(get_settings()).connect() as conn:
        _run(conn)
        conn.commit()
