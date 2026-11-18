"""Answering a question: retrieve Evidence, draft an answer, verify it (ADR-0008)."""

from collections.abc import Callable

from finrag.core.domain import Answer, AnswerStage, Principal, RefusalReason
from finrag.core.errors import ModelOutputError
from finrag.core.interfaces import AnswerGenerator, EvidenceRetriever
from finrag.generation.refusal import refuse
from finrag.generation.validation import validate_answer


class AnswerService:
    def __init__(self, retriever: EvidenceRetriever, generator: AnswerGenerator) -> None:
        self._retriever = retriever
        self._generator = generator

    def answer(
        self,
        principal: Principal,
        question: str,
        on_stage: Callable[[AnswerStage], None] = lambda stage: None,
    ) -> Answer:
        """A verified Answer, or a Refusal. Provider failures (TransientError) propagate.

        `on_stage` is told each stage as it starts, so a caller can show progress.
        """
        on_stage(AnswerStage.RETRIEVING)
        evidence = self._retriever.retrieve(principal, question)
        if not evidence:
            return refuse(RefusalReason.INSUFFICIENT_EVIDENCE)  # no point paying for a model call
        on_stage(AnswerStage.GENERATING)
        try:
            draft = self._generator.generate(question, evidence)
        except ModelOutputError:
            return refuse(RefusalReason.UNVERIFIABLE)
        on_stage(AnswerStage.VALIDATING)
        return validate_answer(draft, evidence)
