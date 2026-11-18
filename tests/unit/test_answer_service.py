"""Answering a question end to end: retrieval, generation and validation (ADR-0008)."""

from collections.abc import Sequence
from uuid import uuid4

import pytest

from finrag.core.domain import (
    DraftAnswer,
    DraftCitation,
    EvidenceChunk,
    Principal,
    RefusalReason,
)
from finrag.core.errors import ModelOutputError, TransientError
from finrag.generation.extractive import ExtractiveGenerator
from finrag.generation.service import AnswerService

PRINCIPAL = Principal(tenant_id=uuid4(), collection_id=uuid4())

CASH = EvidenceChunk(
    chunk_id=uuid4(),
    document_id=uuid4(),
    text="Cash and cash equivalents were $1,234 million at year end.",
    section_path=("Part II", "Liquidity"),
    page_start=41,
    page_end=41,
)
GOOD_DRAFT = DraftAnswer(
    text="Cash was $1,234 million [1].",
    citations=(
        DraftCitation(
            marker=1,
            chunk_id=CASH.chunk_id,
            quoted_span="Cash and cash equivalents were $1,234 million",
        ),
    ),
)


class StubRetriever:
    def __init__(self, evidence: tuple[EvidenceChunk, ...]) -> None:
        self.evidence = evidence

    def retrieve(self, principal: Principal, question: str) -> tuple[EvidenceChunk, ...]:
        return self.evidence


class ScriptedGenerator:
    """Returns a prepared draft, or raises a prepared error, and records its calls."""

    def __init__(self, outcome: DraftAnswer | Exception) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, Sequence[EvidenceChunk]]] = []

    def generate(self, question: str, evidence: Sequence[EvidenceChunk]) -> DraftAnswer:
        self.calls.append((question, evidence))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def service(
    evidence: tuple[EvidenceChunk, ...], outcome: DraftAnswer | Exception
) -> tuple[AnswerService, ScriptedGenerator]:
    generator = ScriptedGenerator(outcome)
    return AnswerService(StubRetriever(evidence), generator), generator


def test_no_evidence_is_refused_without_calling_the_model() -> None:
    answering, generator = service((), GOOD_DRAFT)

    answer = answering.answer(PRINCIPAL, "how much cash?")

    assert answer.refusal_reason is RefusalReason.INSUFFICIENT_EVIDENCE
    assert generator.calls == []


def test_a_valid_draft_becomes_an_answer_with_citations() -> None:
    answering, generator = service((CASH,), GOOD_DRAFT)

    answer = answering.answer(PRINCIPAL, "how much cash?")

    assert answer.refusal_reason is None
    assert answer.text == "Cash was $1,234 million [1]."
    assert [c.document_id for c in answer.citations] == [CASH.document_id]
    assert generator.calls == [("how much cash?", (CASH,))]


def test_a_draft_that_fails_validation_is_refused_as_unverifiable() -> None:
    uncited = DraftAnswer(text="Cash was $1,234 million.", citations=())
    answering, _ = service((CASH,), uncited)

    answer = answering.answer(PRINCIPAL, "how much cash?")

    assert answer.refusal_reason is RefusalReason.UNVERIFIABLE


def test_malformed_model_output_is_refused_as_unverifiable_not_raised() -> None:
    answering, _ = service((CASH,), ModelOutputError("not json"))

    answer = answering.answer(PRINCIPAL, "how much cash?")

    assert answer.refusal_reason is RefusalReason.UNVERIFIABLE


def test_a_provider_failure_propagates_so_the_caller_can_report_it() -> None:
    answering, _ = service((CASH,), TransientError("timeout"))

    with pytest.raises(TransientError):
        answering.answer(PRINCIPAL, "how much cash?")


def test_the_offline_extractive_generator_yields_a_verifiable_answer_from_the_top_chunk() -> None:
    answering = AnswerService(StubRetriever((CASH,)), ExtractiveGenerator())

    answer = answering.answer(PRINCIPAL, "how much cash?")

    assert answer.refusal_reason is None
    assert answer.text == "Cash and cash equivalents were $1,234 million at year end [1]."
    (citation,) = answer.citations
    assert citation.document_id == CASH.document_id
    assert citation.quoted_span == "Cash and cash equivalents were $1,234 million at year end"
