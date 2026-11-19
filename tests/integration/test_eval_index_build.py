"""Build an eval index from the 3-page fixture PDF: real Docling, Postgres, Qdrant and S3.

Hash models stand in for Voyage and FastEmbed, so no paid key or model download is needed. Needs
`make up` for the services; the Qdrant collection and bucket are the throwaway ones of the
integration `settings` fixture, never the real eval collection.
"""

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from finrag.core.config import Settings
from finrag.core.container import Container, build_queueless_container
from finrag.core.domain import Principal
from finrag.evaluation.environment import (
    EVAL_COLLECTION_ID,
    EVAL_TENANT_ID,
    PipelineFilingIndexer,
    eval_principal,
    seed_eval_tenant,
)
from finrag.evaluation.fetch import ManifestEntry
from finrag.evaluation.index import build_index

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent.parent / "fixtures" / "example_10k_extract.pdf"
ACCESSION = "0000000000-26-000001"


@pytest.fixture
def eval_container(settings: Settings) -> Iterator[Container]:
    eval_settings = settings.model_copy(
        update={
            "default_tenant_id": EVAL_TENANT_ID,
            "default_collection_id": EVAL_COLLECTION_ID,
            "max_upload_bytes": 5 * 1024 * 1024,
        }
    )
    container = build_queueless_container(eval_settings)
    seed_eval_tenant(container.session_factory)
    container.ingestion_pipeline().startup_check()
    yield container


@pytest.fixture
def manifest_and_cache(tmp_path: Path) -> tuple[list[ManifestEntry], Path]:
    shutil.copy(FIXTURE, tmp_path / f"{ACCESSION}.pdf")
    entry = ManifestEntry(
        company="Fixture",
        fiscal_year=2025,
        period_end="2025-12-31",
        form="10-K",
        accession_number=ACCESSION,
        url="https://example.test/fixture.htm",
        canonicalisation=1,
        canonical_sha256="0" * 64,
    )
    return [entry], tmp_path


def test_index_build_ingests_the_fixture_once_and_keeps_it_from_the_default_tenant(
    eval_container: Container, manifest_and_cache: tuple[list[ManifestEntry], Path]
) -> None:
    entries, cache_dir = manifest_and_cache
    indexer = PipelineFilingIndexer(eval_container, eval_container.ingestion_pipeline())

    first = build_index(entries, cache_dir, indexer)
    second = build_index(entries, cache_dir, indexer)

    assert [r.ingested for r in first] == [True]
    assert first[0].peak_mb > 0
    assert [r.ingested for r in second] == [False]  # a second run adds nothing

    service = eval_container.document_service()
    (document,) = service.list_documents(eval_principal())
    assert document.status == "ready"
    assert document.chunk_count > 0

    # other integration tests put documents in the default tenant of the shared database, so
    # assert on the eval document's own chunks rather than on emptiness
    eval_chunks = service.list_chunks(eval_principal(), document.id, limit=100, offset=0)
    assert eval_chunks is not None
    eval_chunk_ids = {c.id for c in eval_chunks.items}
    default = Principal(Settings().default_tenant_id, Settings().default_collection_id)
    assert document.id not in {d.id for d in service.list_documents(default)}
    assert service.list_chunks(default, document.id, limit=100, offset=0) is None
    retriever = eval_container.retriever()
    question = "Total net sales by segment"
    found = {c.chunk_id for c in retriever.retrieve(eval_principal(), question)}
    assert found and found <= eval_chunk_ids  # searchable, and only eval chunks come back
    leaked = {c.chunk_id for c in retriever.retrieve(default, question)} & eval_chunk_ids
    assert leaked == set()  # invisible to the default tenant
