"""The DeepEval adapter for the Judge Protocol (ADR-0010, seam S6), against a stub library.

DeepEval is installed only by `uv sync --group evals`, so these tests substitute a stub for the
library module: no network, no API key, and they run in the default dev install and in CI.
"""

import os
from types import SimpleNamespace
from typing import Any

import pytest

from finrag.core.config import Settings
from finrag.evaluation.deepeval_judge import (
    TELEMETRY_OPT_OUT_ENV,
    DeepEvalJudge,
    opt_out_of_telemetry,
)
from finrag.evaluation.judge import JudgeCase


class StubLibrary(SimpleNamespace):
    """A stand-in for the `deepeval` module that records what the adapter asked of it."""

    def __init__(self, scores: dict[str, float | None] | None = None) -> None:
        super().__init__()
        self.scores = scores or {"faithfulness": 1.0, "answer_relevancy": 1.0, "correctness": 1.0}
        self.models: list[Any] = []
        self.cases: dict[str, Any] = {}
        library = self

        class Model:
            def __init__(self, model: str, **kwargs: Any) -> None:
                self.model = model
                library.models.append(self)

            def calculate_cost(self, input_tokens: int, output_tokens: int) -> None:
                """The library calls this with each reply's token counts (a public hook)."""

        def metric(kind: str) -> Any:
            class Metric:
                def __init__(self, **kwargs: Any) -> None:
                    self.model = kwargs["model"]
                    self.score: float | None = None

                def measure(self, test_case: Any, **kwargs: Any) -> float | None:
                    library.cases[kind] = test_case
                    self.score = library.scores[kind]
                    return self.score

            return Metric

        self.AnthropicModel = Model
        self.FaithfulnessMetric = metric("faithfulness")
        self.AnswerRelevancyMetric = metric("answer_relevancy")
        self.GEval = metric("correctness")
        self.LLMTestCase = lambda **kw: SimpleNamespace(**kw)
        self.SingleTurnParams = SimpleNamespace(
            INPUT="input", ACTUAL_OUTPUT="actual_output", EXPECTED_OUTPUT="expected_output"
        )


CASE = JudgeCase(
    question="What were net sales?",
    answer="Net sales were $100 billion [1].",
    reference_answer="$100 billion",
    evidence=("Net sales were $100 billion in 2024.", "Leases note."),
)


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def _clean_telemetry_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TELEMETRY_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv("DEEPEVAL_TELEMETRY_ENABLED", raising=False)


def test_opting_out_sets_the_library_s_telemetry_flag() -> None:
    opt_out_of_telemetry()

    assert os.environ[TELEMETRY_OPT_OUT_ENV] == "YES"


def test_opting_out_overrides_an_environment_that_enabled_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TELEMETRY_OPT_OUT_ENV, "NO")
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_ENABLED", "YES")

    opt_out_of_telemetry()

    assert os.environ[TELEMETRY_OPT_OUT_ENV] == "YES"
    assert os.environ["DEEPEVAL_TELEMETRY_ENABLED"] == "NO"


def test_the_library_is_never_loaded_before_telemetry_is_opted_out() -> None:
    seen: list[str | None] = []

    def loader() -> StubLibrary:
        seen.append(os.environ.get(TELEMETRY_OPT_OUT_ENV))
        return StubLibrary()

    DeepEvalJudge(settings(), library_loader=loader)

    assert seen == ["YES"]


def test_the_real_library_reads_the_opt_out_as_telemetry_off() -> None:
    pytest.importorskip("deepeval")  # only in the `evals` dependency group
    opt_out_of_telemetry()

    from deepeval.telemetry import telemetry_opt_out

    assert telemetry_opt_out() is True


def test_each_metric_score_maps_to_its_field_of_the_judged_scores() -> None:
    library = StubLibrary({"faithfulness": 0.9, "answer_relevancy": 0.8, "correctness": 0.7})
    judge = DeepEvalJudge(settings(), library_loader=lambda: library)

    scores = judge.judge(CASE)

    assert (scores.faithfulness, scores.answer_relevancy, scores.answer_correctness) == (
        0.9,
        0.8,
        0.7,
    )


def test_a_metric_the_library_could_not_score_stays_none() -> None:
    library = StubLibrary({"faithfulness": None, "answer_relevancy": 0.8, "correctness": 0.7})
    judge = DeepEvalJudge(settings(), library_loader=lambda: library)

    assert judge.judge(CASE).faithfulness is None


def test_the_judge_model_defaults_to_opus_and_is_the_one_the_library_is_given() -> None:
    library = StubLibrary()
    default = DeepEvalJudge(settings(), library_loader=lambda: library)
    custom = DeepEvalJudge(settings(judge_model="claude-x"), library_loader=lambda: library)

    assert (default.model_id, custom.model_id) == ("claude-opus-5-5", "claude-x")
    assert [m.model for m in library.models] == ["claude-opus-5-5", "claude-x"]


def test_every_metric_is_scored_by_the_one_judge_model() -> None:
    library = StubLibrary()
    judge = DeepEvalJudge(settings(), library_loader=lambda: library)

    judge.judge(CASE)

    assert len(library.models) == 1  # so one meter sees every judge call


def test_faithfulness_is_judged_against_the_evidence_the_model_saw() -> None:
    library = StubLibrary()

    DeepEvalJudge(settings(), library_loader=lambda: library).judge(CASE)

    faithfulness = library.cases["faithfulness"]
    assert faithfulness.retrieval_context == list(CASE.evidence)
    assert faithfulness.actual_output == CASE.answer
    assert faithfulness.input == CASE.question
    assert library.cases["correctness"].expected_output == CASE.reference_answer


def test_judge_token_usage_is_what_the_library_model_reported() -> None:
    library = StubLibrary()
    judge = DeepEvalJudge(settings(), library_loader=lambda: library)
    library.models[0].calculate_cost(100, 20)
    library.models[0].calculate_cost(50, 5)

    usage = judge.usage()

    assert (usage.input_tokens, usage.output_tokens) == (150, 25)
