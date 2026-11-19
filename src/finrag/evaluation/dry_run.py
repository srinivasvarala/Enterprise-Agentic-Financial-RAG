"""The eval cost dry run (ADR-0010): a few questions, the measured cost, and a hard cap.

Nothing paid runs until a dry run has reported its measured cost and the owner has approved a
per-run cap. The dry run therefore refuses more than five questions unless a cap is supplied,
and stops the moment projected spend would pass that cap.

Cost is tokens times the prices in `Settings` (USD per million tokens). The token counts are what
the two models reported, not an estimate. DeepEval makes several judge calls per metric and the
count varies with the answer, so a full-run projection from five questions is only indicative.

    uv run python -m finrag.evaluation.dry_run [--limit N] [--cost-cap-usd X]
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

from finrag.core.config import Settings
from finrag.core.domain import Principal, TokenUsage
from finrag.evaluation.evaluate import EvalPipeline, evaluate
from finrag.evaluation.golden import GoldenSet, Question, load_golden_set
from finrag.evaluation.judge import Judge, UsageSource

MAX_UNCAPPED_QUESTIONS = 5
GOLDEN_DIR = Path("evals/golden")


class DryRunRefusedError(Exception):
    """The run was not started; the message says what to change."""


class CostCapExceededError(Exception):
    """Projected spend passed the cap, so the run stopped early."""

    def __init__(
        self, *, questions_run: int, spent_usd: float, projected_usd: float, cap_usd: float
    ) -> None:
        super().__init__(
            f"stopped after {questions_run} question(s): ${spent_usd:.4f} spent, projected "
            f"${projected_usd:.4f} for the run, over the ${cap_usd:.2f} cost cap"
        )
        self.questions_run = questions_run
        self.spent_usd = spent_usd
        self.projected_usd = projected_usd
        self.cap_usd = cap_usd


@dataclass(frozen=True, slots=True)
class Price:
    """USD per million tokens."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float

    def cost_usd(self, usage: TokenUsage) -> float:
        return (
            usage.input_tokens * self.input_usd_per_mtok
            + usage.output_tokens * self.output_usd_per_mtok
        ) / 1_000_000


@dataclass(frozen=True, slots=True)
class DryRunReport:
    questions_run: int
    golden_questions: int
    generator_model: str
    judge_model: str
    generator_usage: TokenUsage
    judge_usage: TokenUsage
    generator_cost_usd: float
    judge_cost_usd: float
    total_cost_usd: float
    projected_full_run_usd: float  # mean cost per question run, times the whole Golden Set

    def render(self) -> str:
        def line(role: str, model: str, usage: TokenUsage, cost: float) -> str:
            return (
                f"  {role:<10} {model:<20} in {usage.input_tokens:>10,}  "
                f"out {usage.output_tokens:>9,}  ${cost:.4f}"
            )

        return "\n".join(
            [
                f"Dry run: {self.questions_run} of {self.golden_questions} questions",
                line(
                    "generator", self.generator_model, self.generator_usage, self.generator_cost_usd
                ),
                line("judge", self.judge_model, self.judge_usage, self.judge_cost_usd),
                f"  measured total ${self.total_cost_usd:.4f}",
                f"  projected for all {self.golden_questions} questions "
                f"${self.projected_full_run_usd:.2f} (indicative: judge calls vary per answer)",
            ]
        )


def _sample(golden: GoldenSet, limit: int) -> GoldenSet:
    """The first `limit` questions, round-robin across categories, so cost is representative."""
    by_category: dict[str, list[Question]] = defaultdict(list)
    for q in golden.questions:
        by_category[q.category].append(q)
    rounds = zip_longest(*(by_category[c] for c in sorted(by_category)))
    picked = [q for batch in rounds for q in batch if q is not None]
    return GoldenSet(tuple(picked[:limit]))


def _delta(source: UsageSource, before: TokenUsage) -> TokenUsage:
    now = source.usage()
    return TokenUsage(
        now.input_tokens - before.input_tokens, now.output_tokens - before.output_tokens
    )


def run_dry_run(
    golden: GoldenSet,
    pipeline: EvalPipeline,
    principal: Principal,
    judge: Judge,
    *,
    generator_usage: UsageSource,
    judge_usage: UsageSource,
    settings: Settings,
    limit: int = MAX_UNCAPPED_QUESTIONS,
    cost_cap_usd: float | None = None,
) -> DryRunReport:
    """Answer and judge `limit` questions, measure what they cost, and stop at the cost cap."""
    if cost_cap_usd is not None and cost_cap_usd <= 0:
        raise DryRunRefusedError("The cost cap must be a positive amount in USD.")
    if limit < 1:
        raise DryRunRefusedError("The question limit must be at least 1.")
    if limit > MAX_UNCAPPED_QUESTIONS and cost_cap_usd is None:
        raise DryRunRefusedError(
            f"A dry run covers at most {MAX_UNCAPPED_QUESTIONS} questions. To run {limit}, "
            "supply a per-run cost cap (--cost-cap-usd) the owner has approved."
        )
    sample = _sample(golden, limit)
    generator_price = Price(settings.answer_input_usd_per_mtok, settings.answer_output_usd_per_mtok)
    judge_price = Price(settings.judge_input_usd_per_mtok, settings.judge_output_usd_per_mtok)
    generator_before, judge_before = generator_usage.usage(), judge_usage.usage()
    planned = len(sample.questions)

    def spent() -> float:
        return generator_price.cost_usd(
            _delta(generator_usage, generator_before)
        ) + judge_price.cost_usd(_delta(judge_usage, judge_before))

    def check_cap(done: int) -> None:
        if cost_cap_usd is None:
            return
        so_far = spent()
        projected = so_far / done * planned
        if projected > cost_cap_usd:
            raise CostCapExceededError(
                questions_run=done, spent_usd=so_far, projected_usd=projected, cap_usd=cost_cap_usd
            )

    evaluate(sample, pipeline, principal, judge=judge, after_question=check_cap)

    generated, judged = _delta(generator_usage, generator_before), _delta(judge_usage, judge_before)
    total = generator_price.cost_usd(generated) + judge_price.cost_usd(judged)
    return DryRunReport(
        questions_run=planned,
        golden_questions=len(golden.questions),
        generator_model=settings.answer_model,
        judge_model=judge.model_id,
        generator_usage=generated,
        judge_usage=judged,
        generator_cost_usd=generator_price.cost_usd(generated),
        judge_cost_usd=judge_price.cost_usd(judged),
        total_cost_usd=total,
        projected_full_run_usd=total / planned * len(golden.questions),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the cost of a few judged questions.")
    parser.add_argument("--limit", type=int, default=MAX_UNCAPPED_QUESTIONS)
    parser.add_argument("--cost-cap-usd", type=float, default=None)
    parser.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR)
    args = parser.parse_args(argv)

    # Wired here, not at import: the judge needs the `evals` group and the pipeline needs the
    # eval stack, neither of which the unit tests have.
    from finrag.core.config import get_settings
    from finrag.core.container import build_queueless_container
    from finrag.evaluation.deepeval_judge import DeepEvalJudge
    from finrag.evaluation.environment import eval_principal, require_eval_settings
    from finrag.evaluation.evaluate import EvaluationPipeline
    from finrag.generation.claude import ClaudeAnswerGenerator

    settings = get_settings()
    require_eval_settings(settings)
    files = sorted(args.golden_dir.glob("*.jsonl"))
    if not files:
        print(f"dry run refused: no Golden Set files in {args.golden_dir}.", file=sys.stderr)
        return 1
    golden = load_golden_set("\n".join(f.read_text() for f in files))
    container = build_queueless_container(settings)
    generator = container.generator
    if not isinstance(generator, ClaudeAnswerGenerator):
        print(
            "dry run refused: set FINRAG_GENERATOR_BACKEND=claude to measure the real generator.",
            file=sys.stderr,
        )
        return 1
    judge = DeepEvalJudge(settings)
    try:
        report = run_dry_run(
            golden,
            EvaluationPipeline(container.retriever(), generator),
            eval_principal(),
            judge,
            generator_usage=generator,
            judge_usage=judge,
            settings=settings,
            limit=args.limit,
            cost_cap_usd=args.cost_cap_usd,
        )
    except (DryRunRefusedError, CostCapExceededError) as exc:
        print(f"dry run refused: {exc}", file=sys.stderr)
        return 1
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
