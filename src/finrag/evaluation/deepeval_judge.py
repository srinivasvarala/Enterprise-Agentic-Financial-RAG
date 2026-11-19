"""The DeepEval adapter for the Judge Protocol (ADR-0003, ADR-0010).

DeepEval lives in the `evals` dependency group, so it is loaded lazily, and only after its
telemetry is opted out: filing text and questions must never be reported anywhere.
"""

import importlib
import os
from collections.abc import Callable
from typing import Any

from finrag.core.config import Settings
from finrag.core.domain import TokenUsage
from finrag.evaluation.judge import JudgeCase, JudgeScores

TELEMETRY_OPT_OUT_ENV = "DEEPEVAL_TELEMETRY_OPT_OUT"
LEGACY_TELEMETRY_ENV = "DEEPEVAL_TELEMETRY_ENABLED"  # deprecated alias; any OFF signal wins

CORRECTNESS_CRITERIA = (
    "Decide whether the actual output states the same facts and figures as the expected output. "
    "Penalise a missing or different figure, a wrong period or unit, and any contradiction. "
    "Ignore wording, ordering and citation markers such as [1]."
)


def opt_out_of_telemetry() -> None:
    """Switch the library's usage telemetry off; overrides whatever the environment says."""
    os.environ[TELEMETRY_OPT_OUT_ENV] = "YES"  # noqa: TID251  # a write, not a config read
    os.environ[LEGACY_TELEMETRY_ENV] = "NO"  # noqa: TID251


def load_deepeval() -> Any:
    """Import the library, opting out of telemetry first."""
    opt_out_of_telemetry()
    return importlib.import_module("deepeval")  # needs `uv sync --group evals`


class DeepEvalJudge:
    """Faithfulness, answer relevancy and answer correctness (G-Eval) via DeepEval."""

    def __init__(
        self, settings: Settings, library_loader: Callable[[], Any] = load_deepeval
    ) -> None:
        opt_out_of_telemetry()  # before the library is first imported, whatever the loader is
        self._lib = library_loader()
        self.model_id = settings.judge_model
        self._tokens = TokenUsage()
        self._model = self._build_model(settings)

    def _build_model(self, settings: Settings) -> Any:
        meter = self
        base = self._lib.AnthropicModel

        class MeteredModel(base):  # type: ignore[misc,valid-type]
            """The library's Anthropic model; it reports each reply's token counts here."""

            def calculate_cost(self, input_tokens: int, output_tokens: int) -> Any:
                meter._tokens += TokenUsage(input_tokens, output_tokens)
                return super().calculate_cost(input_tokens, output_tokens)

        return MeteredModel(
            model=settings.judge_model,
            api_key=settings.anthropic_api_key.get_secret_value() or None,
            generation_kwargs={"max_tokens": settings.judge_max_tokens},
        )

    def usage(self) -> TokenUsage:
        return self._tokens

    def judge(self, case: JudgeCase) -> JudgeScores:
        lib = self._lib
        params = lib.SingleTurnParams
        test_case = lib.LLMTestCase(
            input=case.question,
            actual_output=case.answer,
            expected_output=case.reference_answer,
            retrieval_context=list(case.evidence),
        )
        faithfulness = lib.FaithfulnessMetric(model=self._model, async_mode=False)
        relevancy = lib.AnswerRelevancyMetric(model=self._model, async_mode=False)
        correctness = lib.GEval(
            name="Answer correctness",
            criteria=CORRECTNESS_CRITERIA,
            evaluation_params=[params.INPUT, params.ACTUAL_OUTPUT, params.EXPECTED_OUTPUT],
            model=self._model,
            async_mode=False,
        )
        return JudgeScores(
            faithfulness=_score(faithfulness, test_case),
            answer_relevancy=_score(relevancy, test_case),
            answer_correctness=_score(correctness, test_case),
        )


def _score(metric: Any, test_case: Any) -> float | None:
    metric.measure(test_case, _show_indicator=False)
    score = metric.score
    return None if score is None else float(score)
