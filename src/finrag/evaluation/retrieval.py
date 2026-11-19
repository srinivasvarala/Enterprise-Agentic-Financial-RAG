"""`make eval-retrieval`: the Golden Set run retrieval-only over the real eval index (ADR-0010).

Needs only embeddings (no answer or judge calls), so it is cheap to run often. It refuses when the
index is stale, writes an untracked results file under `evals/results/` and prints a summary table.

Results file (`retrieval-<UTC timestamp>-<git sha>.json`):
`run` (kind, started_at, git_sha, image_digest), `models` (embedder, sparse, reranker, answer,
judge),
`index` (the fingerprint recorded by the build) and `report` (`Report.to_dict()`).
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finrag.core.config import Settings
from finrag.core.domain import Principal
from finrag.evaluation.evaluate import EvalPipeline, Metrics, Report, evaluate
from finrag.evaluation.fetch import CACHE_DIR, MANIFEST_PATH, REPO_ROOT
from finrag.evaluation.fingerprint import (
    IndexLocations,
    PointCounter,
    StaleIndexError,
    require_fresh_index,
)
from finrag.evaluation.golden import GoldenSet, GoldenSetError, Question, load_golden_set
from finrag.evaluation.index import RESULTS_DIR

GOLDEN_DIR = REPO_ROOT / "evals" / "golden"


class NoGoldenSetError(Exception):
    """There is no Golden Set to run yet."""


@dataclass(frozen=True)
class RetrievalRun:
    path: Path
    report: Report


def load_golden_dir(golden_dir: Path) -> GoldenSet:
    files = sorted(golden_dir.glob("*.jsonl")) if golden_dir.is_dir() else []
    if not files:
        raise NoGoldenSetError(
            "no Golden Set found: put question files at evals/golden/*.jsonl "
            f"(looked in {golden_dir})."
        )
    questions: list[Question] = []
    for file in files:
        try:
            questions.extend(load_golden_set(file.read_text()).questions)
        except GoldenSetError as exc:
            raise GoldenSetError(f"{file.name}: {exc}") from exc
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise GoldenSetError("a question id appears in more than one Golden Set file")
    return GoldenSet(tuple(questions))


def results_record(
    kind: str,
    report: Report,
    *,
    where: IndexLocations,
    settings: Settings,
    image_digest: str,
    git_sha: str,
    now: datetime,
) -> dict[str, Any]:
    """The results file body shared by the retrieval-only and the full run. Holds no secret."""
    return {
        "run": {
            "kind": kind,
            "started_at": now.isoformat(),
            "git_sha": git_sha,
            "image_digest": image_digest,
        },
        "models": {
            "embedder": settings.voyage_model,
            "sparse": settings.fastembed_bm25_model,
            "reranker": settings.reranker_model,
            "answer": settings.answer_model,
            "judge": settings.judge_model,
        },
        "index": json.loads(where.index_record.read_text()),
        "report": report.to_dict(),
    }


def run_retrieval_eval(
    *,
    golden_dir: Path,
    where: IndexLocations,
    settings: Settings,
    counter: PointCounter,
    pipeline: EvalPipeline,
    principal: Principal,
    results_dir: Path,
    image_digest: str,
    git_sha: str,
    now: datetime,
) -> RetrievalRun:
    """Load the Golden Set, refuse a stale index, evaluate retrieval-only, write the results."""
    golden = load_golden_dir(golden_dir)
    require_fresh_index(where, settings, counter)
    report = evaluate(golden, pipeline, principal, retrieval_only=True)
    record = results_record(
        "retrieval",
        report,
        where=where,
        settings=settings,
        image_digest=image_digest,
        git_sha=git_sha,
        now=now,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"retrieval-{now.astimezone(UTC):%Y%m%dT%H%M%SZ}-{git_sha}.json"
    path.write_text(json.dumps(record, indent=2))
    return RetrievalRun(path, report)


def _cell(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def format_summary(report: Report) -> str:
    """A fixed-width table: overall, then one row per category. A dash means not measured."""
    rows: list[tuple[str, Metrics]] = [("overall", report.overall), *report.by_category.items()]
    width = max(len(name) for name, _ in rows)
    header = f"{'':<{width}}  {'n':>3}  {'recall@8':>8}  {'recall@100':>10}  {'mrr':>5}"
    lines = [header]
    for name, m in rows:
        lines.append(
            f"{name:<{width}}  {m.questions:>3}  {_cell(m.recall_at_8):>8}  "
            f"{_cell(m.recall_at_100):>10}  {_cell(m.mrr):>5}"
        )
    return "\n".join(lines)


def open_retrieval_pipeline(settings: Settings) -> EvalPipeline:
    """The real retrieval from Settings. No generator is built: a retrieval-only run never asks."""
    from finrag.core.container import build_indexing_adapters, build_reranker
    from finrag.evaluation.evaluate import EvaluationPipeline
    from finrag.generation.extractive import ExtractiveGenerator
    from finrag.retrieval.retriever import Retriever
    from finrag.storage.chunk_repository import SqlChunkRepository
    from finrag.storage.db import build_engine, build_session_factory

    embedder, sparse, store = build_indexing_adapters(settings)
    chunks = SqlChunkRepository(build_session_factory(build_engine(settings)))
    retriever = Retriever(embedder, sparse, store, chunks, build_reranker(settings), settings)
    return EvaluationPipeline(retriever, ExtractiveGenerator())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Golden Set retrieval-only")
    parser.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    from finrag.evaluation.environment import (
        EvalEnvironmentError,
        eval_principal,
        require_eval_settings,
    )
    from finrag.evaluation.fingerprint import QdrantPointCounter
    from finrag.evaluation.render import RENDER_RECORD_NAME

    settings = Settings()
    try:
        require_eval_settings(settings)
        run = run_retrieval_eval(
            golden_dir=args.golden_dir,
            where=IndexLocations(
                manifest=args.manifest,
                render_record=args.cache_dir / RENDER_RECORD_NAME,
                index_record=args.results_dir / "index.json",
            ),
            settings=settings,
            counter=QdrantPointCounter(
                settings.qdrant_url, settings.qdrant_collection, settings.qdrant_timeout_seconds
            ),
            pipeline=open_retrieval_pipeline(settings),
            principal=eval_principal(),
            results_dir=args.results_dir,
            image_digest=settings.eval_image_digest,
            git_sha=settings.eval_git_sha,
            now=datetime.now(UTC),
        )
    except (EvalEnvironmentError, NoGoldenSetError, GoldenSetError, StaleIndexError) as exc:
        print(f"eval-retrieval refused: {exc}", file=sys.stderr)
        return 1
    print(format_summary(run.report))
    print(f"results: {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
