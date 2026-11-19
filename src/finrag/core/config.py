"""Application settings. The only place that reads environment variables."""

from enum import StrEnum
from functools import lru_cache
from typing import Literal
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
    qdrant_collection: str = "finrag_chunks"
    qdrant_timeout_seconds: float = 30.0

    # Which adapters the composition root wires. Real ones by default; tests pick fakes so they
    # need no API key and no model download (see .env.example).
    vector_store_backend: Literal["qdrant", "memory"] = "qdrant"
    embedder_backend: Literal["voyage", "hash"] = "voyage"
    sparse_backend: Literal["fastembed", "hash"] = "fastembed"

    # Voyage dense embeddings (ADR-0005). The model is a finance-tuned default, swappable by config.
    voyage_api_key: SecretStr = SecretStr("")
    voyage_model: str = "voyage-finance-2"
    voyage_dimension: int = 1024  # output size of voyage_model; the Qdrant dense vector uses it
    voyage_base_url: str = "https://api.voyageai.com/v1"
    voyage_batch_size: int = 64
    voyage_timeout_seconds: float = 60.0

    # Answer generation (ADR-0005). `extractive` quotes the top chunk: an offline stand-in.
    generator_backend: Literal["claude", "extractive"] = "claude"
    anthropic_api_key: SecretStr = SecretStr("")  # empty: the SDK resolves its own credentials
    answer_model: str = "claude-sonnet-5-5"
    answer_max_tokens: int = 16000  # thinking tokens count toward this
    answer_timeout_seconds: float = 120.0

    # Which Reranker the composition root wires; `overlap` is an offline stand-in for tests
    reranker_backend: Literal["bge", "overlap"] = "bge"
    reranker_model: str = "BAAI/bge-reranker-base"  # downloaded on first use

    # FastEmbed BM25 sparse encoder (model files download on first use)
    fastembed_bm25_model: str = "Qdrant/bm25"
    fastembed_cache_dir: str | None = None

    # S3-compatible blob storage (SeaweedFS locally, AWS S3 in cloud)
    s3_endpoint_url: str = "http://localhost:8333"
    s3_bucket: str = "finrag-documents"
    s3_access_key: SecretStr = SecretStr("finrag")
    s3_secret_key: SecretStr = SecretStr("finrag-secret")

    # Seeded default Tenant and Collection every request resolves to until M6 (API-key auth)
    default_tenant_id: UUID = UUID("00000000-0000-4000-8000-000000000001")
    default_collection_id: UUID = UUID("00000000-0000-4000-8000-000000000002")

    # Questions
    max_question_chars: int = 2000

    # Uploads
    max_upload_bytes: int = 50 * 1024 * 1024

    # PDF parsing (Docling). SEC filings are born-digital, so OCR is off.
    pdf_ocr_enabled: bool = False

    # Chunking and ingestion
    chunk_target_tokens: int = 500  # soft: text chunks close at a paragraph boundary past this
    chunk_max_tokens: int = 800  # hard ceiling for a text chunk
    chunk_overlap_tokens: int = 50  # sentence-aligned overlap between chunks of a section
    table_max_tokens: int = 800  # larger tables split into row groups repeating the header
    embedding_batch_size: int = 64
    ingest_max_attempts: int = 3  # per stage, for transient provider errors
    ingest_retry_base_seconds: float = 2.0  # exponential backoff: base * 2**(attempt - 1)
    # A Document not queued/terminal and untouched this long is presumed orphaned (worker crash,
    # lost enqueue) and its ingestion job is re-deferred by the reconciler. Must exceed the
    # longest single stage (Docling on a large 10-K).
    stuck_document_seconds: int = 900

    # Retrieval: each branch's candidates, RRF constant and Evidence size (tuned via the evals)
    retrieve_dense_k: int = 50
    retrieve_sparse_k: int = 50
    rrf_k: int = 60
    rerank_top_n: int = 8
    # Reranker scores run 0..1. Permissive until the Golden Set calibrates it: with no Evidence the
    # answer is a Refusal, and an early false refusal costs more than a weak chunk the validator
    # will still check.
    min_rerank_score: float = 0.0

    # Provenance of an eval run, passed in by `make eval-retrieval` (the container has no git)
    eval_image_digest: str = "unknown"
    eval_git_sha: str = "unknown"

    # OpenTelemetry (Phoenix locally)
    otel_exporter_endpoint: str = "http://localhost:6006/v1/traces"


@lru_cache
def get_settings() -> Settings:
    return Settings()
