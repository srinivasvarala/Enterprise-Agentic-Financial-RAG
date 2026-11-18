"""Reranker adapters: the bge cross-encoder (ADR-0005) and a dependency-free offline stand-in."""

import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from finrag.core.config import Settings
from finrag.core.domain import EvidenceChunk


class _CrossEncoder(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> Any: ...


def _load_cross_encoder(model_name: str) -> _CrossEncoder:
    from sentence_transformers import CrossEncoder

    model: _CrossEncoder = CrossEncoder(model_name)
    return model


class BgeReranker:
    """Scores (question, chunk) pairs with a bge cross-encoder, mapped into 0..1.

    The model loads on first use, so startup never downloads it.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        model_factory: Callable[[], _CrossEncoder] | None = None,
    ) -> None:
        self._factory = model_factory or (lambda: _load_cross_encoder(model_name))
        self._model: _CrossEncoder | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "BgeReranker":
        return cls(settings.reranker_model)

    def score(self, question: str, chunks: Sequence[EvidenceChunk]) -> list[float]:
        if not chunks:
            return []
        if self._model is None:
            self._model = self._factory()
        logits = self._model.predict([(question, c.text) for c in chunks])
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


class OverlapReranker:
    """Fraction of the question's words found in the chunk. Offline stand-in, not a real model."""

    def score(self, question: str, chunks: Sequence[EvidenceChunk]) -> list[float]:
        wanted = set(question.lower().split())
        if not wanted:
            return [0.0 for _ in chunks]
        return [len(wanted & set(c.text.lower().split())) / len(wanted) for c in chunks]
