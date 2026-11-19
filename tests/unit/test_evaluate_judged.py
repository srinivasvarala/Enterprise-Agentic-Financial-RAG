"""Judged metrics in the evaluation report (ADR-0010, seams S1 and S6), with a fake judge.

Same real pipeline over in-memory adapters as `test_evaluate`; the judge is a script, so nothing
here calls a model.
"""

import json

import pytest
from test_evaluate import PRINCIPAL, World, question, span, unanswerable

from finrag.core.domain import Principal
from finrag.core.errors import TransientError
from finrag.evaluation.evaluate import (
    EvalPipeline,
    JudgeFailureRateExceededError,
    PipelineResult,
    evaluate,
)
from finrag.evaluation.golden import GoldenSet, Question
from finrag.evaluation.judge import JudgeCase, JudgeScores


class FakeJudge:
    """A Judge that scores by script and records what it was asked to judge."""

    model_id = "fake-judge"

    def __init__(
        self,
        scores: dict[str, JudgeScores] | None = None,
        fail: dict[str, Exception] | None = None,
    ) -> None:
        self.scores = scores or {}
        self.fail = fail or {}
        self.cases: list[JudgeCase] = []

    def judge(self, case: JudgeCase) -> JudgeScores:
        self.cases.append(case)
        if case.question in self.fail:
            raise self.fail[case.question]
        return self.scores.get(case.question, JudgeScores(1.0, 1.0, 1.0))


def judged_world() -> World:
    world = World()
    world.index("Net sales were 100 billion dollars in 2024.", score=0.9)
    return world


def test_without_a_judge_the_report_has_no_judged_section() -> None:
    world = judged_world()
    golden = GoldenSet((question("q1", span("Net sales were 100 billion")),))

    report = evaluate(golden, world.pipeline, PRINCIPAL)

    assert report.judged is None
    assert json.loads(report.to_json())["judged"] is None


def test_each_judged_metric_maps_into_the_judged_section_of_the_report() -> None:
    world = judged_world()
    golden = GoldenSet((question("q1", span("Net sales were 100 billion")),))
    judge = FakeJudge({"question q1?": JudgeScores(0.9, 0.8, 0.7)})

    report = evaluate(golden, world.pipeline, PRINCIPAL, judge=judge)

    judged = json.loads(report.to_json())["judged"]
    assert judged["judge_model"] == "fake-judge"
    assert judged["overall"] == {
        "scored": 1,
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "answer_correctness": 0.7,
    }
    assert judged["by_category"]["text"]["faithfulness"] == 0.9
    assert judged["questions"] == [
        {
            "id": "q1",
            "category": "text",
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "answer_correctness": 0.7,
            "error": None,
        }
    ]


def test_the_judge_sees_the_answer_the_reference_and_the_evidence_the_model_saw() -> None:
    world = judged_world()
    world.index("Unrelated filler about leases.", score=0.5)
    golden = GoldenSet(
        (Question("q1", "text", "How big?", "100 billion", (span("Net sales were 100 billion"),)),)
    )
    judge = FakeJudge()

    evaluate(golden, world.pipeline, PRINCIPAL, judge=judge)

    (case,) = judge.cases
    assert case.question == "How big?"
    assert case.reference_answer == "100 billion"
    assert case.answer == "Net sales were 100 billion dollars in 2024 [1]."
    assert case.evidence == (
        "Net sales were 100 billion dollars in 2024.",
        "Unrelated filler about leases.",
    )


def test_judged_scores_average_over_the_questions_the_judge_scored() -> None:
    world = judged_world()
    golden = GoldenSet(
        (
            question("q1", span("Net sales were 100 billion")),
            question("q2", span("Net sales were 100 billion")),
        )
    )
    judge = FakeJudge(
        {"question q1?": JudgeScores(1.0, 0.5, None), "question q2?": JudgeScores(0.5, 0.5, 0.5)}
    )

    judged = evaluate(golden, world.pipeline, PRINCIPAL, judge=judge).judged

    assert judged is not None
    assert judged.overall.scored == 2
    assert judged.overall.faithfulness == 0.75
    assert judged.overall.answer_correctness == 0.5  # the None is left out of the mean


def test_a_refused_or_unanswerable_question_is_not_judged() -> None:
    world = World(min_rerank_score=0.1)
    world.index("Net sales were 100 billion dollars in 2024.", score=0.0)  # refused: too weak
    golden = GoldenSet((question("q1", span("Net sales were 100 billion")), unanswerable("u1")))
    judge = FakeJudge()

    report = evaluate(golden, world.pipeline, PRINCIPAL, judge=judge)

    assert judge.cases == []
    assert report.judged is not None
    assert report.judged.overall.scored == 0
    assert report.judged.overall.faithfulness is None
    assert report.judged.questions == ()


def test_a_retrieval_only_run_never_calls_the_judge() -> None:
    world = judged_world()
    golden = GoldenSet((question("q1", span("Net sales were 100 billion")),))
    judge = FakeJudge()

    report = evaluate(golden, world.pipeline, PRINCIPAL, retrieval_only=True, judge=judge)

    assert judge.cases == []
    assert report.judged is None


def test_the_after_question_hook_runs_once_per_question_and_may_stop_the_run() -> None:
    world = judged_world()
    golden = GoldenSet(
        (question("q1", span("Net sales were 100 billion")), question("q2", span("x")))
    )
    seen: list[int] = []

    def stop_after_first(done: int) -> None:
        seen.append(done)
        raise StopIteration

    with pytest.raises(StopIteration):
        evaluate(golden, world.pipeline, PRINCIPAL, after_question=stop_after_first)

    assert seen == [1]


class ProviderDownError(Exception):
    pass


def three_questions() -> GoldenSet:
    return GoldenSet(
        tuple(question(f"q{n}", span("Net sales were 100 billion")) for n in (1, 2, 3))
    )


def test_a_judge_failure_on_one_question_is_recorded_and_the_run_continues() -> None:
    judge = FakeJudge(fail={"question q2?": ProviderDownError("401 sk-ant-SECRET rejected")})

    report = evaluate(three_questions(), judged_world().pipeline, PRINCIPAL, judge=judge)

    assert len(report.questions) == 3
    assert len(judge.cases) == 3  # q3 was still judged after q2 failed
    judged = report.judged
    assert judged is not None
    failed = judged.questions[1]
    assert (failed.id, failed.faithfulness, failed.answer_correctness) == ("q2", None, None)
    assert failed.error == "ProviderDownError"
    assert "SECRET" not in report.to_json()  # the class name only, never the provider message
    assert judged.not_judged == 1
    assert judged.overall.scored == 2
    assert judged.overall.faithfulness == 1.0  # the mean skips the failed question


def test_too_many_judge_failures_stop_the_run_with_a_clear_message() -> None:
    judge = FakeJudge(fail={f"question q{n}?": ProviderDownError("x") for n in (1, 2, 3)})

    with pytest.raises(JudgeFailureRateExceededError, match=r"2 of 3.*judge"):
        evaluate(
            three_questions(),
            judged_world().pipeline,
            PRINCIPAL,
            judge=judge,
            max_judge_failure_rate=0.5,
        )

    assert len(judge.cases) == 2  # stopped as soon as the second failure passed half of 3


def test_failures_at_exactly_the_allowed_fraction_do_not_stop_the_run() -> None:
    judge = FakeJudge(fail={"question q1?": ProviderDownError("x")})

    report = evaluate(
        three_questions(),
        judged_world().pipeline,
        PRINCIPAL,
        judge=judge,
        max_judge_failure_rate=1 / 3,
    )

    assert report.judged is not None
    assert report.judged.not_judged == 1


class FlakyPipeline:
    """Fails (like a provider outage) on the questions named, otherwise delegates."""

    def __init__(self, inner: EvalPipeline, fail: dict[str, Exception]) -> None:
        self.inner, self.fail, self.asked = inner, fail, 0

    def run(self, principal: Principal, question: str, *, answer: bool) -> PipelineResult:
        self.asked += 1
        if question in self.fail:
            raise self.fail[question]
        return self.inner.run(principal, question, answer=answer)


def test_a_provider_error_on_one_question_is_recorded_and_left_out_of_the_metrics() -> None:
    pipeline = FlakyPipeline(
        judged_world().pipeline, {"question q2?": TransientError("503 sk-ant-SECRET at http://x")}
    )

    report = evaluate(three_questions(), pipeline, PRINCIPAL, judge=FakeJudge())

    assert pipeline.asked == 3  # the run went on after q2
    assert [q.id for q in report.questions] == ["q1", "q3"]
    assert [(e.id, e.error) for e in report.errors] == [("q2", "TransientError")]
    assert report.overall.questions == 2  # excluded from every metric
    assert report.overall.recall_at_8 == 1.0
    assert "SECRET" not in report.to_json() and "http" not in report.to_json()


def test_an_unexpected_error_on_one_question_is_recorded_the_same_way() -> None:
    pipeline = FlakyPipeline(judged_world().pipeline, {"question q1?": KeyError("boom")})

    report = evaluate(three_questions(), pipeline, PRINCIPAL, retrieval_only=True)

    assert [(e.id, e.error) for e in report.errors] == [("q1", "KeyError")]
    assert len(report.questions) == 2


def test_the_judge_failure_rate_is_measured_over_the_questions_eligible_for_judging() -> None:
    world = judged_world()
    golden = GoldenSet(
        (
            question("q1", span("Net sales were 100 billion")),
            question("q2", span("Net sales were 100 billion")),
            unanswerable("u1"),
            unanswerable("u2"),
        )
    )
    judge = FakeJudge(fail={f"question q{n}?": ProviderDownError("x") for n in (1, 2)})

    # 2 failures of 4 questions is not over 50%, but of the 2 judged-eligible ones it is 100%
    with pytest.raises(JudgeFailureRateExceededError, match=r"2 of 2"):
        evaluate(golden, world.pipeline, PRINCIPAL, judge=judge, max_judge_failure_rate=0.5)


def test_the_report_records_the_search_and_evidence_depth_it_was_measured_at() -> None:
    golden = GoldenSet((question("q1", span("Net sales were 100 billion")),))

    report = evaluate(
        golden, judged_world().pipeline, PRINCIPAL, search_k=40, evidence_k=5, retrieval_only=True
    )

    data = json.loads(report.to_json())
    assert (data["search_k"], data["evidence_k"]) == (40, 5)


def test_a_search_depth_smaller_than_the_candidates_limits_recall_at_100() -> None:
    world = judged_world()
    for n in range(5):
        world.index(f"filler passage {n} about leases", score=0.9)
    golden = GoldenSet((question("q1", span("Net sales were 100 billion")),))
    deep = evaluate(golden, world.pipeline, PRINCIPAL, retrieval_only=True, search_k=100)
    shallow = evaluate(golden, world.pipeline, PRINCIPAL, retrieval_only=True, search_k=0)

    assert deep.overall.recall_at_100 == 1.0
    assert shallow.overall.recall_at_100 == 0.0
