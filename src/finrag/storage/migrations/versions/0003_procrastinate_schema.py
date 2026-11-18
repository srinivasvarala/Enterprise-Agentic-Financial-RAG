"""Apply Procrastinate's job-queue schema alongside our own tables.

Revision ID: 0003
Revises: 0002
"""

from alembic import op
from procrastinate.schema import SchemaManager

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    # Raw DBAPI cursor: the schema contains '%' characters that SQLAlchemy would treat as params.
    cursor = op.get_bind().connection.cursor()
    cursor.execute(SchemaManager.get_schema())
    cursor.close()


def downgrade() -> None:
    raise NotImplementedError(
        "Procrastinate's schema is upgraded, not downgraded, with its own migrations"
    )
