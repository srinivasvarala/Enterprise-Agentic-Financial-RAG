"""Programmatic Alembic upgrade."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def upgrade_to_head(connection: Connection) -> None:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    command.upgrade(config, "head")
