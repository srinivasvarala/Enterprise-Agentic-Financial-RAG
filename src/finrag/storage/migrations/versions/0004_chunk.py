"""Chunk table.

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"


def upgrade() -> None:
    op.create_table(
        "chunk",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenant.id"), nullable=False),
        sa.Column("collection_id", sa.Uuid(), sa.ForeignKey("collection.id"), nullable=False),
        sa.Column("document_id", sa.Uuid(), sa.ForeignKey("document.id"), nullable=False),
        sa.Column(
            "document_version_id", sa.Uuid(), sa.ForeignKey("document_version.id"), nullable=False
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("section_path", sa.JSON(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("bounding_boxes", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("document_version_id", "ordinal", name="uq_chunk_version_ordinal"),
    )
    op.create_index(
        "ix_chunk_tenant_document_ordinal", "chunk", ["tenant_id", "document_id", "ordinal"]
    )


def downgrade() -> None:
    op.drop_table("chunk")
