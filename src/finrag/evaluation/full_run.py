"""`make eval`: the full Golden Set run, retrieval, answers and judged metrics (ADR-0010).

PAID: every answered question is generated and then judged. The run therefore refuses to start
without a per-run cost cap, refuses a stale index, stops when projected spend passes the cap and
stops when too many questions fail to judge. A judge failure on one question is recorded (null
scores and the error class) and the run goes on.

Results file (`full-<UTC timestamp>-<git sha>.json`): the retrieval run's shape (`run`, `models`,
`index`, `report`) where `report.judged` is filled, plus `cost` (cap, tokens and USD per model).
No key is ever written to it.

    make eval COST_CAP_USD=20
"""

import argparse
import json
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finrag.core.config import Settings
from finrag.core.domain import Principal
from finrag.evaluation.dry_run import CostCapExceededError, SpendTracker, round_robin
from finrag.evaluation.evaluate import (
    EvalPipeline,
    JudgeFailureRateExceededError,
    QuestionFailureRateExceededError,
    Report,
    RunState,
    build_report,
    evaluate,
    search_depth,
)
from finrag.evaluation.fetch import CACHE_DIR, MANIFEST_PATH
from finrag.evaluation.fingerprint import (
    DocumentCounter,
    IndexLocations,
    PointCounter,
    StaleIndexError,
    require_fresh_index,
)
from finrag.evaluation.golden import GoldenSetError
from finrag.evaluation.index import RESULTS_DIR
from finrag.evaluation.judge import Judge, UsageSource
from finrag.evaluation.retrieval import (
    GOLDEN_DIR,
    NoGoldenSetError,
    format_summary,
    load_golden_dir,
    results_record,
)

FALLBACK_NOTE = (
    "Cost is approximate: generator tokens served by a fallback model are billed at that "
    "model's rate, not the configured one."
)


class EvalRefusedError(Exception):
    """The run was not started; the message says what to change."""


@dataclass(frozen=True, slots=True)
class FullRun:
    path: Path
    report: Report
    cap_usd: float
    generator_usd: float
    judge_usd: float

    @property
    def total_usd(self) -> float:
        return self.generator_usd + self.judge_usd


def _cost_record(spend: SpendTracker, cap_usd: float) -> dict[str, Any]:
    generated, judged = spend.generated(), spend.judged()
    generator_usd = spend.generator_price.cost_usd(generated)
    judge_usd = spend.judge_price.cost_usd(judged)
    return {
        "cap_usd": cap_usd,
        "generator_tokens": {"input": generated.input_tokens, "output": generated.output_tokens},
        "judge_tokens": {"input": judged.input_tokens, "output": judged.output_tokens},
        "generator_usd": generator_usd,
        "judge_usd": judge_usd,
        "total_usd": generator_usd + judge_usd,
        "note": FALLBACK_NOTE,
    }


def run_full_eval(
    *,
    golden_dir: Path,
    where: IndexLocations,
    settings: Settings,
    counter: PointCounter,
    pipeline: EvalPipeline,
    principal: Principal,
    judge: Judge,
    generator_usage: UsageSource,
    judge_usage: UsageSource,
    results_dir: Path,
    image_digest: str,
    git_sha: str,
    now: datetime,
    cost_cap_usd: float | None,
    max_judge_failure_rate: float,
    documents: DocumentCounter | None = None,
) -> FullRun:
    """Refuse without a cap or on a stale index, then answer and judge every question.

    Questions run in the dry run's round-robin order. A run that stops (cost cap, judge failure
    rate, Ctrl-C or any other error) first writes a partial results file, marked `incomplete`,
    then re-raises with the file's path as a note.
    """
    if cost_cap_usd is None or cost_cap_usd <= 0:
        raise EvalRefusedError(
            "a per-run cost cap is required and there is no default: pass COST_CAP_USD=<usd> "
            "(make eval COST_CAP_USD=20), a positive amount the owner has approved."
        )
    golden = round_robin(load_golden_dir(golden_dir))
    require_fresh_index(where, settings, counter, documents)
    spend = SpendTracker(
        settings,
        generator_usage,
        judge_usage,
        planned=len(golden.questions),
        cost_cap_usd=cost_cap_usd,
    )
    state = RunState()
    search_k, evidence_k = search_depth(settings), settings.rerank_top_n

    def record_of(report: Report) -> dict[str, Any]:
        record: dict[str, Any] = results_record(
            "full",
            report,
            where=where,
            settings=settings,
            image_digest=image_digest,
            git_sha=git_sha,
            now=now,
        )
        record["cost"] = _cost_record(spend, cost_cap_usd)
        return record

    stamp = f"{now.astimezone(UTC):%Y%m%dT%H%M%SZ}-{git_sha}"
    try:
        report = evaluate(
            golden,
            pipeline,
            principal,
            judge=judge,
            after_question=spend.check_cap,
            max_judge_failure_rate=max_judge_failure_rate,
            max_question_error_rate=max_judge_failure_rate,
            state=state,
            search_k=search_k,
            evidence_k=evidence_k,
        )
    except BaseException as exc:
        partial = record_of(
            build_report(state, judge=judge, search_k=search_k, evidence_k=evidence_k)
        )
        partial["incomplete"] = {
            "reason": type(exc).__name__,  # the class only: never a provider's message
            "questions_completed": len(state.results) + len(state.errors),
            "questions_total": len(golden.questions),
        }
        path = results_dir / f"full-{stamp}-incomplete.json"
        with suppress(OSError):  # never hide the real error behind a failed write
            results_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(partial, indent=2))
            exc.add_note(f"partial results (incomplete run): {path}")
        raise
    record = record_of(report)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"full-{stamp}.json"
    path.write_text(json.dumps(record, indent=2))
    cost = record["cost"]
    return FullRun(path, report, cost_cap_usd, cost["generator_usd"], cost["judge_usd"])


def _cell(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _judged_table(report: Report) -> list[str]:
    judged = report.judged
    if judged is None:
        return []
    rows = [("overall", judged.overall), *judged.by_category.items()]
    width = max(len(name) for name, _ in rows)
    lines = [
        f"{'':<{width}}  {'n':>3}  {'faithfulness':>12}  {'relevancy':>9}  {'correctness':>11}"
    ]
    for name, m in rows:
        lines.append(
            f"{name:<{width}}  {m.scored:>3}  {_cell(m.faithfulness):>12}  "
            f"{_cell(m.answer_relevancy):>9}  {_cell(m.answer_correctness):>11}"
        )
    lines.append(
        f"{judged.not_judged} of {len(judged.questions)} questions not judged "
        "(judge failed; see the results file)"
    )
    return lines


def format_full_summary(run: FullRun) -> str:
    return "\n".join(
        [
            format_summary(run.report),
            "",
            *_judged_table(run.report),
            "",
            f"cost: ${run.total_usd:.4f} (generator ${run.generator_usd:.4f}, "
            f"judge ${run.judge_usd:.4f}), cap ${run.cap_usd:.2f}",
            FALLBACK_NOTE,
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PAID: run the full Golden Set, answers and judge")
    parser.add_argument("--cost-cap-usd", type=float, default=None)
    parser.add_argument("--max-judge-failure-rate", type=float, default=None)
    parser.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    # Wired here, not at import: the judge needs the `evals` group and the pipeline the eval stack.
    from finrag.core.container import build_queueless_container
    from finrag.evaluation.deepeval_judge import DeepEvalJudge
    from finrag.evaluation.environment import (
        EvalEnvironmentError,
        eval_principal,
        require_eval_settings,
    )
    from finrag.evaluation.evaluate import EvaluationPipeline
    from finrag.evaluation.fingerprint import QdrantPointCounter, SqlDocumentCounter
    from finrag.evaluation.render import RENDER_RECORD_NAME
    from finrag.generation.claude import ClaudeAnswerGenerator

    settings = Settings()
    cap = args.cost_cap_usd if args.cost_cap_usd is not None else settings.eval_cost_cap_usd
    rate = (
        args.max_judge_failure_rate
        if args.max_judge_failure_rate is not None
        else settings.eval_max_judge_failure_rate
    )
    try:
        if cap is None or cap <= 0:  # before any client is built
            raise EvalRefusedError(
                "a per-run cost cap is required and there is no default: run "
                "`make eval COST_CAP_USD=<usd>` with an amount the owner has approved."
            )
        require_eval_settings(settings)
        container = build_queueless_container(settings)
        generator = container.generator
        if not isinstance(generator, ClaudeAnswerGenerator):
            raise EvalRefusedError("set FINRAG_GENERATOR_BACKEND=claude to run the real generator.")
        judge = DeepEvalJudge(settings)
        run = run_full_eval(
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
            pipeline=EvaluationPipeline(container.retriever(), generator),
            principal=eval_principal(),
            judge=judge,
            generator_usage=generator,
            judge_usage=judge,
            results_dir=args.results_dir,
            image_digest=settings.eval_image_digest,
            git_sha=settings.eval_git_sha,
            now=datetime.now(UTC),
            cost_cap_usd=cap,
            max_judge_failure_rate=rate,
            documents=SqlDocumentCounter.from_settings(settings),
        )
    except (
        EvalRefusedError,
        EvalEnvironmentError,
        NoGoldenSetError,
        GoldenSetError,
        StaleIndexError,
        CostCapExceededError,
        JudgeFailureRateExceededError,
        QuestionFailureRateExceededError,
    ) as exc:
        print(f"eval refused: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", []):
            print(note, file=sys.stderr)
        return 1
    except Exception as exc:  # no traceback and no provider text: the class, and the partial file
        print(f"eval failed: {type(exc).__name__}; see the log for details.", file=sys.stderr)
        for note in getattr(exc, "__notes__", []):
            print(note, file=sys.stderr)
        return 1
    print(format_full_summary(run))
    print(f"results: {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
