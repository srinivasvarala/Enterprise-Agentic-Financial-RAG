"""The eval cost dry run (ADR-0010): measured usage and cost, a five-question limit, a cost cap.

Everything is fake: a fake judge and fake usage sources that count scripted tokens, so no paid
call can happen here.
"""

from collections.abc import Sequence

import pytest
from test_evaluate import PRINCIPAL, World, span

from finrag.core.config import Settings
from finrag.core.domain import DraftAnswer, EvidenceChunk, TokenUsage
from finrag.evaluation.dry_run import CostCapExceededError, DryRunRefusedError, run_dry_run
from finrag.evaluation.golden import GoldenSet, Question
from finrag.evaluation.judge import JudgeCase, JudgeScores
from finrag.generation.extractive import ExtractiveGenerator

SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]  # judge $4/$20, answer $2/$10 per Mtok


class Meter:
    """A fake usage source: the model it stands for charges a fixed number of tokens per call."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.per_call = TokenUsage(input_tokens, output_tokens)
        self.total = TokenUsage()

    def charge(self) -> None:
        self.total += self.per_call

    def usage(self) -> TokenUsage:
        return self.total


class MeteredGenerator(ExtractiveGenerator):
    def __init__(self, meter: Meter) -> None:
        self.meter = meter

    def generate(self, question: str, evidence: Sequence[EvidenceChunk]) -> DraftAnswer:
        self.meter.charge()
        return super().generate(question, evidence)


class MeteredJudge:
    model_id = "fake-judge"

    def __init__(self, meter: Meter) -> None:
        self.meter = meter
        self.calls = 0
        self.asked: list[str] = []

    def judge(self, case: JudgeCase) -> JudgeScores:
        self.calls += 1
        self.asked.append(case.question)
        self.meter.charge()
        return JudgeScores(1.0, 1.0, 1.0)


class Rig:
    """A world whose generator costs 1000 in / 200 out and judge 5000 in / 500 out per question.

    Per question: answer 1000*$2/M + 200*$10/M = $0.004; judge 5000*$4/M + 500*$20/M = $0.03.
    """

    def __init__(self, categories: Sequence[str]) -> None:
        self.generator_meter = Meter(1000, 200)
        self.judge_meter = Meter(5000, 500)
        self.world = World(generator=MeteredGenerator(self.generator_meter))
        self.world.index("Net sales were 100 billion dollars in 2024.", score=0.9)
        self.judge = MeteredJudge(self.judge_meter)
        self.golden = GoldenSet(
            tuple(
                Question(
                    f"q{n}",
                    category,
                    f"question q{n}?",
                    "100 billion",
                    (span("Net sales were 100 billion"),),
                )
                for n, category in enumerate(categories)
            )
        )

    def run(self, **kwargs: object):  # type: ignore[no-untyped-def]
        return run_dry_run(
            self.golden,
            self.world.pipeline,
            PRINCIPAL,
            self.judge,
            generator_usage=self.generator_meter,
            judge_usage=self.judge_meter,
            settings=SETTINGS,
            **kwargs,  # type: ignore[arg-type]
        )


TEXT = ("text",) * 8


def test_it_reports_the_measured_tokens_and_cost_of_each_model() -> None:
    rig = Rig(("text",) * 3)

    report = rig.run()

    assert report.questions_run == 3
    assert report.generator_usage == TokenUsage(3000, 600)
    assert report.judge_usage == TokenUsage(15000, 1500)
    assert report.generator_cost_usd == pytest.approx(0.012)
    assert report.judge_cost_usd == pytest.approx(0.09)
    assert report.total_cost_usd == pytest.approx(0.102)


def test_it_runs_at_most_five_questions_by_default() -> None:
    rig = Rig(TEXT)

    report = rig.run()

    assert report.questions_run == 5
    assert rig.judge.calls == 5


def test_it_samples_across_categories_rather_than_taking_the_first_five() -> None:
    rig = Rig(("text",) * 6 + ("table", "multi-span"))

    rig.run()

    assert {"question q6?", "question q7?"} <= set(rig.judge.asked)  # the table and multi-span


def test_it_refuses_more_than_five_questions_without_a_cost_cap_and_calls_nothing() -> None:
    rig = Rig(TEXT)

    with pytest.raises(DryRunRefusedError, match="cost cap"):
        rig.run(limit=6)

    assert rig.judge.calls == 0
    assert rig.generator_meter.total == TokenUsage()


def test_it_runs_more_than_five_questions_when_a_cost_cap_is_supplied() -> None:
    rig = Rig(TEXT)

    report = rig.run(limit=8, cost_cap_usd=5.0)

    assert report.questions_run == 8


def test_it_stops_as_soon_as_projected_spend_would_exceed_the_cap() -> None:
    rig = Rig(TEXT)  # $0.034 per question: five would cost $0.17

    with pytest.raises(CostCapExceededError) as stopped:
        rig.run(cost_cap_usd=0.10)

    assert stopped.value.questions_run == 1  # after one question the projection is $0.17
    assert stopped.value.projected_usd == pytest.approx(0.17)
    assert rig.judge.calls == 1


def test_a_run_projected_to_fit_the_cap_completes() -> None:
    rig = Rig(TEXT)

    report = rig.run(cost_cap_usd=0.20)

    assert report.questions_run == 5
    assert report.total_cost_usd <= 0.20


def test_it_refuses_a_cap_that_is_not_positive() -> None:
    rig = Rig(TEXT)

    with pytest.raises(DryRunRefusedError, match="cost cap"):
        rig.run(cost_cap_usd=0.0)


def test_it_projects_the_cost_of_the_whole_golden_set_from_the_measured_mean() -> None:
    rig = Rig(("text",) * 8)

    report = rig.run()

    assert report.projected_full_run_usd == pytest.approx(0.034 * 8)


def test_the_printed_report_names_the_tokens_and_the_dollar_cost() -> None:
    rig = Rig(("text",) * 2)

    printed = rig.run().render()

    assert "fake-judge" in printed
    assert "10,000" in printed  # two questions: the judge's input tokens
    assert "$0.0680" in printed
