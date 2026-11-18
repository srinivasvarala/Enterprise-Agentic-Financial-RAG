"""Parsing the model's structured reply into a DraftAnswer."""

from collections.abc import Sequence
from typing import Any, TypeGuard

from finrag.core.domain import DraftAnswer, DraftCitation, EvidenceChunk
from finrag.core.errors import ModelOutputError


def parse_draft(raw: object, evidence: Sequence[EvidenceChunk]) -> DraftAnswer:
    """Map `{answer, citations: [{marker, evidence_number, quoted_span}]}` onto chunk ids.

    The model sees Evidence as numbered 1..n and never handles chunk ids. Anything that is not
    that shape raises ModelOutputError.
    """
    if not isinstance(raw, dict):
        raise ModelOutputError("reply is not an object")
    answer = raw.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ModelOutputError("reply has no answer text")
    items = raw.get("citations")
    if not isinstance(items, list):
        raise ModelOutputError("reply has no citations list")
    return DraftAnswer(text=answer, citations=tuple(_citation(item, evidence) for item in items))


def _citation(item: Any, evidence: Sequence[EvidenceChunk]) -> DraftCitation:
    if not isinstance(item, dict):
        raise ModelOutputError("a citation is not an object")
    marker, number, span = item.get("marker"), item.get("evidence_number"), item.get("quoted_span")
    if not _is_int(marker) or not _is_int(number) or not isinstance(span, str):
        raise ModelOutputError("a citation is missing marker, evidence_number or quoted_span")
    if not 1 <= number <= len(evidence):
        raise ModelOutputError(f"evidence_number {number} is outside the Evidence")
    return DraftCitation(marker=marker, chunk_id=evidence[number - 1].chunk_id, quoted_span=span)


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)
