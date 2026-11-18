"""Offline AnswerGenerator: quotes the first sentence of the top chunk. Not a real model."""

import re
from collections.abc import Sequence

from finrag.core.domain import DraftAnswer, DraftCitation, EvidenceChunk
from finrag.core.errors import ModelOutputError

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ExtractiveGenerator:
    """Lets the API and its tests run with no API key; the validator still checks the result."""

    def generate(self, question: str, evidence: Sequence[EvidenceChunk]) -> DraftAnswer:
        if not evidence:
            raise ModelOutputError("no evidence to quote")
        top = evidence[0]
        sentence = _SENTENCE_END.split(top.text.strip())[0].rstrip(".!?")
        return DraftAnswer(
            text=f"{sentence} [1].",
            citations=(DraftCitation(marker=1, chunk_id=top.chunk_id, quoted_span=sentence),),
        )
