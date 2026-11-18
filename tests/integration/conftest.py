"""Integration fixtures: real Postgres (fresh database) and real S3 (fresh bucket)."""

from collections.abc import Iterator
from uuid import uuid4

import boto3
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from finrag.api.app import create_app
from finrag.core.config import Environment, Settings
from finrag.core.container import Container, build_container
from finrag.storage.migrate import upgrade_to_head


def _delete_bucket(settings: Settings) -> None:
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key.get_secret_value(),
        aws_secret_access_key=settings.s3_secret_key.get_secret_value(),
        region_name="us-east-1",
    )
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=settings.s3_bucket):
        for obj in page.get("Contents", []):
            client.delete_object(Bucket=settings.s3_bucket, Key=obj["Key"])
    client.delete_bucket(Bucket=settings.s3_bucket)


@pytest.fixture(scope="session")
def settings() -> Iterator[Settings]:
    """Settings pointing at a throwaway database and bucket, migrated to head."""
    base = Settings()
    suffix = uuid4().hex[:10]
    db_name = f"finrag_test_{suffix}"
    base_url = make_url(base.database_url.get_secret_value())
    admin = create_engine(base_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    test_url = base_url.set(database=db_name)
    test_settings = Settings(
        env=Environment.TEST,
        database_url=test_url.render_as_string(hide_password=False),  # type: ignore[arg-type]
        s3_bucket=f"finrag-test-{suffix}",
        max_upload_bytes=64 * 1024,
        # real Qdrant, fake models: no Voyage key and no BM25 model download needed
        vector_store_backend="qdrant",
        embedder_backend="hash",
        sparse_backend="hash",
        reranker_backend="overlap",
        generator_backend="extractive",
        qdrant_collection=f"finrag_test_{suffix}",
    )
    engine = create_engine(test_url)
    with engine.connect() as conn:
        upgrade_to_head(conn)
        conn.commit()
    engine.dispose()
    yield test_settings
    QdrantClient(url=test_settings.qdrant_url).delete_collection(test_settings.qdrant_collection)
    _delete_bucket(test_settings)  # SeaweedFS reserves a volume per bucket; leaked ones fill it
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture(scope="session")
def container(settings: Settings) -> Iterator[Container]:
    built = build_container(settings)
    yield built
    close = getattr(built.job_queue, "close", None)
    if close is not None:
        close()


@pytest.fixture
def client(settings: Settings, container: Container) -> TestClient:
    return TestClient(create_app(settings, container=container))
