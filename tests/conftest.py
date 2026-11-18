from collections.abc import Iterator

import pytest
from fakes import FlakyEmbedder, InMemoryBlobStore, InMemoryJobQueue, SpyVectorStore
from fastapi.testclient import TestClient
from harness import Env
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from finrag.api.app import create_app
from finrag.core.config import Environment, Settings
from finrag.core.container import Container
from finrag.indexing.in_memory import HashSparseEncoder
from finrag.ingestion.text_parser import TextParser
from finrag.storage.models import Base
from finrag.storage.seed import seed_defaults


@pytest.fixture
def env() -> Iterator[Env]:
    settings = Settings(env=Environment.TEST, max_upload_bytes=64 * 1024)
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    with session_factory() as session:
        seed_defaults(session, settings)
        session.commit()
    blobs, jobs = InMemoryBlobStore(), InMemoryJobQueue()
    embedder, vectors = FlakyEmbedder(), SpyVectorStore()
    container = Container(
        settings=settings,
        session_factory=session_factory,
        blob_store=blobs,
        job_queue=jobs,
        parsers={"text/plain": TextParser()},
        embedder=embedder,
        sparse_encoder=HashSparseEncoder(),
        vector_store=vectors,
    )
    app = create_app(settings, container=container)
    yield Env(
        app=app,
        client=TestClient(app),
        blobs=blobs,
        jobs=jobs,
        settings=settings,
        container=container,
        vectors=vectors,
        embedder=embedder,
        sleeps=[],
    )
