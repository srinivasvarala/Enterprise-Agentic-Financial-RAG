"""Seed the default Tenant and Collection whose ids live in config.

Revision ID: 0002
Revises: 0001
"""

from alembic import op
from sqlalchemy.orm import Session

from finrag.core.config import get_settings
from finrag.storage.seed import seed_defaults

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    session = Session(bind=op.get_bind())
    seed_defaults(session, get_settings())
    session.flush()


def downgrade() -> None:
    pass  # seeded rows are removed with their tables by 0001's downgrade
