"""Answering a question: retrieve Evidence, draft an answer, verify it (ADR-0008)."""

from finrag.core.domain import Answer, Principal, RefusalReason
from finrag.core.errors import ModelOutputError
from finrag.core.interfaces import AnswerGenerator, EvidenceRetriever
from finrag.generation.refusal import refuse
from finrag.generation.validation import validate_answer


class AnswerService:
    def __init__(self, retriever: EvidenceRetriever, generator: AnswerGenerator) -> None:
        self._retriever = retriever
        self._generator = generator

    def answer(self, principal: Principal, question: str) -> Answer:
        """A verified Answer, or a Refusal. Provider failures (TransientError) propagate."""
        evidence = self._retriever.retrieve(principal, question)
        if not evidence:
            return refuse(RefusalReason.INSUFFICIENT_EVIDENCE)  # no point paying for a model call
        try:
            draft = self._generator.generate(question, evidence)
        except ModelOutputError:
            return refuse(RefusalReason.UNVERIFIABLE)
        return validate_answer(draft, evidence)
