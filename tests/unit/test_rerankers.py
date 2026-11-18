"""The bge reranker adapter and the offline overlap reranker: no model download."""

from uuid import uuid4

import pytest

from finrag.core.domain import EvidenceChunk
from finrag.retrieval.rerankers import BgeReranker, OverlapReranker


def chunk(text: str) -> EvidenceChunk:
    return EvidenceChunk(uuid4(), uuid4(), text, ("Part I",), 1, 1)


class FakeCrossEncoder:
    def __init__(self, logits: list[float]) -> None:
        self.logits = logits
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = pairs
        return self.logits


def test_bge_scores_question_chunk_pairs_and_maps_logits_into_zero_to_one() -> None:
    model = FakeCrossEncoder([0.0, 20.0, -20.0])
    reranker = BgeReranker(model_factory=lambda: model)

    scores = reranker.score("cash?", [chunk("a"), chunk("b"), chunk("c")])

    assert model.pairs == [("cash?", "a"), ("cash?", "b"), ("cash?", "c")]
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] > 0.99
    assert scores[2] < 0.01


def test_bge_loads_its_model_lazily_and_only_once() -> None:
    loads = 0

    def factory() -> FakeCrossEncoder:
        nonlocal loads
        loads += 1
        return FakeCrossEncoder([0.0])

    reranker = BgeReranker(model_factory=factory)
    assert loads == 0
    assert reranker.score("q", []) == []
    assert loads == 0  # nothing to score, so no model
    reranker.score("q", [chunk("a")])
    reranker.score("q", [chunk("a")])
    assert loads == 1


def test_overlap_reranker_prefers_the_chunk_sharing_more_question_words() -> None:
    close, far = chunk("cash held at year end"), chunk("office leases expire soon")

    close_score, far_score = OverlapReranker().score("how much cash at year end", [close, far])

    assert 0.0 <= far_score < close_score <= 1.0
