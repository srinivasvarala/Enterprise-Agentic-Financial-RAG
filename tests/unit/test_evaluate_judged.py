"""Judged metrics in the evaluation report (ADR-0010, seams S1 and S6), with a fake judge.

Same real pipeline over in-memory adapters as `test_evaluate`; the judge is a script, so nothing
here calls a model.
"""

import json

import pytest
from test_evaluate import PRINCIPAL, World, question, span, unanswerable

from finrag.evaluation.evaluate import evaluate
from finrag.evaluation.golden import GoldenSet, Question
from finrag.evaluation.judge import JudgeCase, JudgeScores


class FakeJudge:
    """A Judge that scores by script and records what it was asked to judge."""

    model_id = "fake-judge"

    def __init__(self, scores: dict[str, JudgeScores] | None = None) -> None:
        self.scores = scores or {}
        self.cases: list[JudgeCase] = []

    def judge(self, case: JudgeCase) -> JudgeScores:
        self.cases.append(case)
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
