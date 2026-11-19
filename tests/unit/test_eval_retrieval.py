"""Seam S5: the retrieval-only eval command's orchestration, with a fake pipeline and counter.

Reads the results file and the summary a developer sees; never an internal step.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from finrag.core.config import Settings
from finrag.core.domain import EvidenceChunk, Principal
from finrag.evaluation.environment import eval_principal
from finrag.evaluation.evaluate import PipelineResult
from finrag.evaluation.fingerprint import IndexLocations, StaleIndexError, record_index
from finrag.evaluation.retrieval import (
    NoGoldenSetError,
    RetrievalRun,
    format_summary,
    run_retrieval_eval,
)
from finrag.retrieval.retriever import RetrievalTrace, ScoredChunk

NOW = datetime(2026, 11, 19, 14, 30, 5, tzinfo=UTC)
GOLDEN = (
    '{"id": "q1", "category": "text", "question": "What were net sales?", '
    '"reference_answer": "100", "gold_spans": [{"document": "apple", "pages": [1], '
    '"text": "Net sales were 100 billion"}]}\n'
    '{"id": "q2", "category": "unanswerable", "question": "Who won the cup?", '
    '"reference_answer": "n/a", "gold_spans": []}\n'
)


class FakeCounter:
    def __init__(self, points: int) -> None:
        self.points = points

    def count_points(self) -> int:
        return self.points


class FakePipeline:
    """Always retrieves one chunk containing the gold text; records the calls it got."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def run(self, principal: Principal, question: str, *, answer: bool) -> PipelineResult:
        self.calls.append((question, answer))
        chunk = EvidenceChunk(uuid4(), uuid4(), "Net sales were 100 billion dollars.", (), 1, 1)
        scored = (ScoredChunk(chunk, 0.9),)
        return PipelineResult(RetrievalTrace(scored, scored, (chunk,)), None)


class World:
    def __init__(self, tmp_path: Path) -> None:
        self.golden_dir = tmp_path / "golden"
        self.results_dir = tmp_path / "results"
        (tmp_path / "manifest.json").write_text("[]")
        self.where = IndexLocations(
            manifest=tmp_path / "manifest.json",
            render_record=tmp_path / "render.json",
            index_record=self.results_dir / "index.json",
        )
        self.settings = Settings(_env_file=None)  # type: ignore[call-arg]
        self.counter = FakeCounter(12)
        self.pipeline = FakePipeline()

    def build_index(self) -> None:
        record_index(self.where, self.settings, self.counter)

    def write_golden(self, name: str = "set.jsonl", text: str = GOLDEN) -> None:
        self.golden_dir.mkdir(exist_ok=True)
        (self.golden_dir / name).write_text(text)

    def run(self) -> RetrievalRun:
        return run_retrieval_eval(
            golden_dir=self.golden_dir,
            where=self.where,
            settings=self.settings,
            counter=self.counter,
            pipeline=self.pipeline,
            principal=eval_principal(),
            results_dir=self.results_dir,
            image_digest="sha256:abc123",
            git_sha="deadbee",
            now=NOW,
        )


@pytest.fixture
def world(tmp_path: Path) -> World:
    w = World(tmp_path)
    w.write_golden()
    w.build_index()
    return w


def test_a_fresh_index_gives_a_results_file_named_by_time_and_sha(world: World) -> None:
    path = world.run().path

    assert path.parent == world.results_dir
    assert path.name == "retrieval-20261119T143005Z-deadbee.json"
    data = json.loads(path.read_text())
    assert data["run"] == {
        "kind": "retrieval",
        "started_at": "2026-11-19T14:30:05+00:00",
        "git_sha": "deadbee",
        "image_digest": "sha256:abc123",
    }
    assert data["models"]["embedder"] == world.settings.voyage_model
    assert data["models"]["sparse"] == world.settings.fastembed_bm25_model
    assert data["models"]["reranker"] == world.settings.reranker_model
    assert data["models"]["answer"] == world.settings.answer_model
    assert data["models"]["judge"] == world.settings.judge_model
    assert data["index"] == json.loads(world.where.index_record.read_text())
    assert data["report"]["overall"]["recall_at_8"] == 1.0
    assert set(data["report"]["by_category"]) == {"text", "unanswerable"}


def test_the_run_is_retrieval_only(world: World) -> None:
    world.run()

    assert world.pipeline.calls and not any(answer for _, answer in world.pipeline.calls)


def test_a_stale_index_is_refused_before_any_question_is_asked(world: World) -> None:
    world.counter.points = 3  # the collection changed since the build

    with pytest.raises(StaleIndexError, match="make eval-index"):
        world.run()

    assert world.pipeline.calls == []
    assert not list(world.results_dir.glob("retrieval-*.json"))


def test_no_golden_set_says_where_to_put_one(world: World) -> None:
    (world.golden_dir / "set.jsonl").unlink()

    with pytest.raises(NoGoldenSetError, match=r"evals/golden/\*\.jsonl"):
        world.run()


def test_every_jsonl_in_the_golden_directory_is_loaded(world: World) -> None:
    world.write_golden("more.jsonl", GOLDEN.replace('"q1"', '"q3"').replace('"q2"', '"q4"'))

    assert world.run().report.overall.questions == 4


def test_the_summary_lists_overall_and_each_category(world: World) -> None:
    table = format_summary(world.run().report)

    assert "overall" in table and "text" in table and "unanswerable" in table
    assert "recall@8" in table and "mrr" in table
    assert "1.00" in table  # recall for the text category
    assert "-" in table  # unanswerable has no recall: shown as a dash, not 0
