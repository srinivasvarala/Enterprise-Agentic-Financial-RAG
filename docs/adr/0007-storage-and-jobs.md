# ADR-0007: Postgres system of record, S3 blobs, Procrastinate jobs

**Status:** Accepted (2026-11-18)

## Decision
- **Postgres** (SQLAlchemy 2 + Alembic) holds tenants, users, API keys, collections, documents,
  versions, chunk records and ingestion status.
- **S3-compatible blob storage** for originals: SeaweedFS in Compose (MinIO stopped publishing community images), AWS S3 later — same adapter.
- **Procrastinate** (Postgres-backed) for the ingestion job queue: no Redis.
- Dedupe by SHA-256 per tenant; a changed re-upload creates a new version and atomically replaces
  that document's chunks; delete removes blob, rows and vectors.

## Consequences
One fewer service to run. If job throughput ever outgrows Postgres, the `jobs` module is the only
place that changes.
