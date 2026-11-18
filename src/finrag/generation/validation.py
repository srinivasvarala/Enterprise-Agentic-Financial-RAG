"""Citation validation (ADR-0008): pure functions, no I/O."""

import re
from collections.abc import Sequence

from finrag.core.domain import Answer, Citation, DraftAnswer, EvidenceChunk, RefusalReason
from finrag.generation.refusal import refuse

MARKER = re.compile(r"\[(\d+)\]")


# Whole sentences that make no claim. Matched exactly, so a claim is never excused by a prefix.
LEAD_INS = frozenset({"based on the filings", "based on the documents"})
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    """Numbers as written, minus thousands separators. No unit conversion (ADR-0006)."""
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(MARKER.sub("", text))}


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _has_uncited_sentence(text: str) -> bool:
    for sentence in SENTENCE_END.split(text.strip()):
        if MARKER.search(sentence):
            continue
        if _normalise(sentence).rstrip(".!?").lower() in LEAD_INS:
            continue
        return True
    return False


def validate_answer(draft: DraftAnswer, evidence: Sequence[EvidenceChunk]) -> Answer:
    """Resolve every draft Citation against the Evidence."""
    if _has_uncited_sentence(draft.text):
        return refuse(RefusalReason.UNVERIFIABLE)
    cited_markers = {c.marker for c in draft.citations}
    if any(int(m) not in cited_markers for m in MARKER.findall(draft.text)):
        return refuse(RefusalReason.UNVERIFIABLE)
    by_id = {chunk.chunk_id: chunk for chunk in evidence}
    citations = []
    for cited in draft.citations:
        chunk = by_id.get(cited.chunk_id)
        if chunk is None:
            return refuse(RefusalReason.UNVERIFIABLE)
        if _normalise(cited.quoted_span) not in _normalise(chunk.text):
            return refuse(RefusalReason.UNVERIFIABLE)
        citations.append(
            Citation(
                marker=cited.marker,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                section_path=chunk.section_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                quoted_span=cited.quoted_span,
            )
        )
    cited_text = " ".join(by_id[c.chunk_id].text for c in draft.citations)
    if not _numbers(draft.text) <= _numbers(cited_text):
        return refuse(RefusalReason.UNVERIFIABLE)
    return Answer(text=draft.text, citations=tuple(citations))
