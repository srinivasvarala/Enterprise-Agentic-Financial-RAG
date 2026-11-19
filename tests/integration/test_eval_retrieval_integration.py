"""The retrieval-only eval through the real stack: Docling, Postgres, Qdrant and S3.

The 3-page fixture, a tiny hand-built Golden Set and hash models: no paid key or model download.
Everything lives in throwaway names (own Qdrant collection, the session's bucket; the document
is deleted at the end), never in the real eval collection or bucket. Also exercises what the unit
tests cannot: the real `QdrantPointCounter`, and `record_index` after a build.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from finrag.core.config import Settings
from finrag.core.container import Container, build_queueless_container
from finrag.evaluation.environment import (
    EVAL_COLLECTION_ID,
    EVAL_TENANT_ID,
    PipelineFilingIndexer,
    eval_principal,
    seed_eval_tenant,
)
from finrag.evaluation.evaluate import EvaluationPipeline
from finrag.evaluation.fetch import ManifestEntry
from finrag.evaluation.fingerprint import (
    IndexLocations,
    QdrantPointCounter,
    StaleIndexError,
    record_index,
)
from finrag.evaluation.index import build_index
from finrag.evaluation.retrieval import RetrievalRun, format_summary, run_retrieval_eval
from finrag.generation.extractive import ExtractiveGenerator
from finrag.storage.models import Chunk, Document, DocumentVersion

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent.parent / "fixtures" / "example_10k_extract.pdf"
ACCESSION = "0000000000-26-000002"
GOLDEN = (
    '{"id": "t1", "category": "text", "question": "How does management monitor the factors?", '
    '"reference_answer": "ongoing", "gold_spans": [{"document": "fixture", "pages": [1], '
    '"text": "Management monitors these factors on an ongoing basis"}]}\n'
    '{"id": "u1", "category": "unanswerable", "question": "Who won the 2030 cup?", '
    '"reference_answer": "n/a", "gold_spans": []}\n'
)


@pytest.fixture
def own_settings(settings: Settings) -> Iterator[Settings]:
    """Eval Tenant and Collection ids, but a throwaway Qdrant collection (the session bucket)."""
    suffix = uuid4().hex[:10]
    own = settings.model_copy(
        update={
            "default_tenant_id": EVAL_TENANT_ID,
            "default_collection_id": EVAL_COLLECTION_ID,
            "max_upload_bytes": 5 * 1024 * 1024,
            "qdrant_collection": f"finrag_test_retrieval_{suffix}",
        }
    )
    yield own
    QdrantClient(url=own.qdrant_url).delete_collection(own.qdrant_collection)


@pytest.fixture
def container(own_settings: Settings) -> Iterator[Container]:
    built = build_queueless_container(own_settings)
    seed_eval_tenant(built.session_factory)
    built.ingestion_pipeline().startup_check()
    yield built
    with built.session_factory() as session:  # leave the shared test database as it was found
        mine = [d.id for d in session.query(Document).filter(Document.tenant_id == EVAL_TENANT_ID)]
        for model in (Chunk, DocumentVersion):
            session.query(model).filter(model.document_id.in_(mine)).delete()  # type: ignore[attr-defined]
        session.query(Document).filter(Document.id.in_(mine)).delete()
        session.commit()


def test_retrieval_only_run_over_the_real_index_writes_results_and_refuses_when_stale(
    container: Container, own_settings: Settings, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{ACCESSION}.pdf").write_bytes(FIXTURE.read_bytes() + b"\n% retrieval test\n")
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
    (tmp_path / "manifest.json").write_text("[]")
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    (golden_dir / "tiny.jsonl").write_text(GOLDEN)
    where = IndexLocations(
        manifest=tmp_path / "manifest.json",
        render_record=cache / "render.json",
        index_record=tmp_path / "results" / "index.json",
    )
    counter = QdrantPointCounter(
        own_settings.qdrant_url, own_settings.qdrant_collection, own_settings.qdrant_timeout_seconds
    )

    build_index(
        [entry],
        cache,
        PipelineFilingIndexer(container, container.ingestion_pipeline()),
    )
    record_index(where, own_settings, counter)
    recorded = json.loads(where.index_record.read_text())
    assert recorded["expected_points"] > 0
    assert recorded["expected_points"] == counter.count_points()

    def run(second: int) -> RetrievalRun:
        return run_retrieval_eval(
            golden_dir=golden_dir,
            where=where,
            settings=own_settings,
            counter=counter,
            pipeline=EvaluationPipeline(container.retriever(), ExtractiveGenerator()),
            principal=eval_principal(),
            results_dir=tmp_path / "results",
            image_digest="sha256:test",
            git_sha="abc1234",
            now=datetime(2026, 11, 19, 9, 0, second, tzinfo=UTC),
        )

    first = run(0)

    data = json.loads(first.path.read_text())
    assert first.path.name == "retrieval-20261119T090000Z-abc1234.json"
    assert data["run"]["image_digest"] == "sha256:test"
    assert data["index"] == recorded
    overall = data["report"]["overall"]
    assert overall["questions"] == 2
    assert overall["recall_at_100"] == 1.0  # every chunk of a 3-page document is a candidate
    assert overall["false_refusal_rate"] is None  # retrieval-only: nothing was answered
    assert "recall@8" in format_summary(first.report)

    # the collection changes behind the record's back: the real point counter notices
    client = QdrantClient(url=own_settings.qdrant_url)
    point_ids, _ = client.scroll(own_settings.qdrant_collection, limit=1, with_payload=False)
    client.delete(own_settings.qdrant_collection, points_selector=[point_ids[0].id])
    with pytest.raises(StaleIndexError, match="make eval-index"):
        run(1)
