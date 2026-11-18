"""Application settings. The only place that reads environment variables."""

from enum import StrEnum
from functools import lru_cache
from uuid import UUID

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    PROD = "prod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FINRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Environment = Environment.LOCAL
    log_level: str = "INFO"

    # Postgres: system of record (ADR-0007)
    database_url: SecretStr = SecretStr("postgresql+psycopg://finrag:finrag@localhost:5432/finrag")

    # Qdrant: hybrid search index (ADR-0001)
    qdrant_url: str = "http://localhost:6333"

    # S3-compatible blob storage (SeaweedFS locally, AWS S3 in cloud)
    s3_endpoint_url: str = "http://localhost:8333"
    s3_bucket: str = "finrag-documents"
    s3_access_key: SecretStr = SecretStr("finrag")
    s3_secret_key: SecretStr = SecretStr("finrag-secret")

    # Seeded default Tenant and Collection every request resolves to until M6 (API-key auth)
    default_tenant_id: UUID = UUID("00000000-0000-4000-8000-000000000001")
    default_collection_id: UUID = UUID("00000000-0000-4000-8000-000000000002")

    # Uploads
    max_upload_bytes: int = 50 * 1024 * 1024

    # Chunking and ingestion
    chunk_target_tokens: int = 500  # soft: text chunks close at a paragraph boundary past this
    chunk_max_tokens: int = 800  # hard ceiling for a text chunk
    chunk_overlap_tokens: int = 50  # sentence-aligned overlap between chunks of a section
    table_max_tokens: int = 800  # larger tables split into row groups repeating the header
    embedding_batch_size: int = 64
    ingest_max_attempts: int = 3  # per stage, for transient provider errors
    ingest_retry_base_seconds: float = 2.0  # exponential backoff: base * 2**(attempt - 1)

    # OpenTelemetry (Phoenix locally)
    otel_exporter_endpoint: str = "http://localhost:6006/v1/traces"


@lru_cache
def get_settings() -> Settings:
    return Settings()
