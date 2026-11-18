"""Turning the model's structured reply into a DraftAnswer (ADR-0008)."""

from typing import Any
from uuid import uuid4

import pytest

from finrag.core.domain import DraftAnswer, DraftCitation, EvidenceChunk
from finrag.core.errors import ModelOutputError
from finrag.generation.draft import parse_draft


def chunk() -> EvidenceChunk:
    return EvidenceChunk(uuid4(), uuid4(), "text", ("Part I",), 1, 1)


FIRST, SECOND = chunk(), chunk()


def reply(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "answer": "Cash was high [1]. Debt was low [2].",
        "citations": [
            {"marker": 1, "evidence_number": 1, "quoted_span": "cash was high"},
            {"marker": 2, "evidence_number": 2, "quoted_span": "debt was low"},
        ],
    }
    return base | overrides


def test_evidence_numbers_are_translated_to_chunk_ids() -> None:
    draft = parse_draft(reply(), (FIRST, SECOND))

    assert draft == DraftAnswer(
        text="Cash was high [1]. Debt was low [2].",
        citations=(
            DraftCitation(1, FIRST.chunk_id, "cash was high"),
            DraftCitation(2, SECOND.chunk_id, "debt was low"),
        ),
    )


@pytest.mark.parametrize(
    "bad",
    [
        "not an object",
        {},
        reply(answer=None),
        reply(answer=""),
        reply(citations="none"),
        reply(citations=[{"marker": 1, "evidence_number": 1}]),
        reply(citations=[{"marker": "1", "evidence_number": 1, "quoted_span": "x"}]),
        reply(citations=[{"marker": 1, "evidence_number": 3, "quoted_span": "x"}]),
        reply(citations=[{"marker": 1, "evidence_number": 0, "quoted_span": "x"}]),
    ],
    ids=[
        "not-an-object",
        "empty-object",
        "null-answer",
        "empty-answer",
        "citations-not-a-list",
        "missing-quoted-span",
        "marker-not-an-int",
        "evidence-number-beyond-evidence",
        "evidence-number-zero",
    ],
)
def test_a_reply_that_is_not_the_requested_shape_is_a_model_output_error(bad: object) -> None:
    with pytest.raises(ModelOutputError):
        parse_draft(bad, (FIRST, SECOND))
