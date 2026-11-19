"""The judge boundary (ADR-0010): the harness depends on this Protocol, never on the library.

A judge scores an Answer on three measures from a stronger, different model than the generator.
"""

from dataclasses import dataclass
from typing import Protocol

from finrag.core.domain import TokenUsage


@dataclass(frozen=True, slots=True)
class JudgeCase:
    """What the judge sees for one answered question."""

    question: str
    answer: str
    reference_answer: str
    evidence: tuple[str, ...]  # the Evidence the generator saw, as text


@dataclass(frozen=True, slots=True)
class JudgeScores:
    """Each score is 0..1, or None when the judge could not produce it."""

    faithfulness: float | None  # is every claim supported by the Evidence
    answer_relevancy: float | None  # does the answer address the question
    answer_correctness: float | None  # does it agree with the reference answer


class Judge(Protocol):
    model_id: str

    def judge(self, case: JudgeCase) -> JudgeScores: ...


class UsageSource(Protocol):
    """Anything that counts the tokens a model has used so far (cumulative)."""

    def usage(self) -> TokenUsage: ...
