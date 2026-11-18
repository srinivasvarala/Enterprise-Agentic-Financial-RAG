"""Application settings. The only place that reads environment variables."""

from enum import StrEnum
from functools import lru_cache

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

    # OpenTelemetry (Phoenix locally)
    otel_exporter_endpoint: str = "http://localhost:6006/v1/traces"


@lru_cache
def get_settings() -> Settings:
    return Settings()
