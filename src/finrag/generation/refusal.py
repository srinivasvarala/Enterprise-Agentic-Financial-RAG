"""Refusals: an Answer that declines, with fixed text per reason (ADR-0008, ADR-0009)."""

from finrag.core.domain import Answer, RefusalReason

REFUSAL_TEXT = {
    RefusalReason.INSUFFICIENT_EVIDENCE: (
        "The indexed documents do not contain enough evidence to answer this question."
    ),
    RefusalReason.UNVERIFIABLE: (
        "An answer was drafted, but its citations could not be verified against the documents."
    ),
    RefusalReason.OUT_OF_POLICY: "This question is outside what this service can answer.",
}


def refuse(reason: RefusalReason) -> Answer:
    return Answer(text=REFUSAL_TEXT[reason], citations=(), refusal_reason=reason)
