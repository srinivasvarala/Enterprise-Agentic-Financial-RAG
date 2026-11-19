"""Evaluating a Golden Set against a pipeline into a report (ADR-0010, seam S1).

The pipeline is the real Retriever and AnswerService over in-memory adapters; only the Golden Set
is hand-built. Each test reads a number off the report, never an internal step.
"""

import json
from collections.abc import Sequence
from uuid import uuid4, uuid5

from fakes import FixedReranker, InMemoryChunkRepository

from finrag.core.config import Settings
from finrag.core.domain import ChunkKind, DraftAnswer, EvidenceChunk, Principal, VectorPoint
from finrag.core.errors import ModelOutputError
from finrag.core.interfaces import AnswerGenerator
from finrag.evaluation.evaluate import EvaluationPipeline, evaluate
from finrag.evaluation.golden import GoldenSet, Question, TextSpan, is_citation_hit
from finrag.generation.extractive import ExtractiveGenerator
from finrag.indexing.in_memory import HashEmbedder, HashSparseEncoder, InMemoryVectorStore
from finrag.retrieval.retriever import Retriever

PRINCIPAL = Principal(tenant_id=uuid4(), collection_id=uuid4())


class World:
    """An indexed Collection plus the pipeline the evaluation drives."""

    def __init__(
        self,
        rerank_top_n: int = 8,
        min_rerank_score: float = 0.0,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.embedder = HashEmbedder()
        self.sparse = HashSparseEncoder()
        self.vectors = InMemoryVectorStore()
        self.chunks = InMemoryChunkRepository()
        self.reranker = FixedReranker()
        retriever = Retriever(
            embedder=self.embedder,
            sparse_encoder=self.sparse,
            vector_store=self.vectors,
            chunks=self.chunks,
            reranker=self.reranker,
            settings=Settings(  # type: ignore[call-arg]
                _env_file=None, rerank_top_n=rerank_top_n, min_rerank_score=min_rerank_score
            ),
        )
        self.pipeline = EvaluationPipeline(retriever, generator or ExtractiveGenerator())

    def index(self, text: str, score: float = 0.5) -> EvidenceChunk:
        chunk_id, document_id = uuid4(), uuid4()
        chunk = EvidenceChunk(chunk_id, document_id, text, ("Part II",), 1, 1)
        self.chunks.add(PRINCIPAL.tenant_id, PRINCIPAL.collection_id, chunk)
        self.vectors.upsert(
            PRINCIPAL.tenant_id,
            [
                VectorPoint(
                    id=uuid5(chunk_id, "point"),
                    dense=self.embedder.embed_documents([text])[0],
                    sparse=self.sparse.encode_documents([text])[0],
                    tenant_id=PRINCIPAL.tenant_id,
                    collection_id=PRINCIPAL.collection_id,
                    document_id=document_id,
                    document_version_id=uuid4(),
                    chunk_id=chunk_id,
                    kind=ChunkKind.TEXT,
                    page_start=1,
                    page_end=1,
                )
            ],
        )
        self.reranker.scores[chunk_id] = score
        return chunk


def span(text: str) -> TextSpan:
    return TextSpan("apple", 1, 1, text)


def question(qid: str, *spans: TextSpan, category: str = "text") -> Question:
    return Question(qid, category, f"question {qid}?", "reference", tuple(spans))


def test_perfect_retrieval_scores_one_everywhere() -> None:
    world = World()
    world.index("Net sales were 100 billion dollars in 2024.", score=0.9)
    world.index("Unrelated filler about leases.")
    golden = GoldenSet((question("q1", span("Net sales were 100 billion")),))

    report = evaluate(golden, world.pipeline, PRINCIPAL)

    overall = report.overall
    assert (overall.recall_at_8, overall.recall_at_100, overall.mrr) == (1.0, 1.0, 1.0)


def test_a_hit_found_by_search_but_lost_by_the_reranker_counts_at_100_not_at_8() -> None:
    world = World(rerank_top_n=1)
    world.index("Net sales were 100 billion dollars in 2024.", score=0.2)
    world.index("Unrelated filler about leases.", score=0.9)
    golden = GoldenSet((question("q1", span("Net sales were 100 billion")),))

    overall = evaluate(golden, world.pipeline, PRINCIPAL).overall

    assert overall.recall_at_100 == 1.0
    assert overall.recall_at_8 == 0.0
    assert overall.mrr == 0.5  # ranked second by the reranker


def test_a_span_no_indexed_chunk_contains_is_a_miss_at_search() -> None:
    world = World()
    world.index("Unrelated filler about leases.", score=0.9)
    golden = GoldenSet((question("q1", span("Net sales were 100 billion")),))

    overall = evaluate(golden, world.pipeline, PRINCIPAL).overall

    assert (overall.recall_at_8, overall.recall_at_100, overall.mrr) == (0.0, 0.0, 0.0)


def test_a_multi_span_question_gets_partial_recall_for_the_spans_found() -> None:
    world = World(rerank_top_n=1)
    world.index("Net sales were 100 billion dollars in 2024.", score=0.9)
    world.index("Operating income was 30 billion dollars.", score=0.1)
    golden = GoldenSet(
        (
            question(
                "q1",
                span("Net sales were 100 billion"),
                span("Operating income was 30 billion"),
                category="multi-span",
            ),
        )
    )

    result = evaluate(golden, world.pipeline, PRINCIPAL).questions[0]

    assert result.recall_at_8 == 0.5  # only the top-1 Evidence is seen
    assert result.recall_at_100 == 1.0
    assert result.reciprocal_rank == 1.0


def unanswerable(qid: str) -> Question:
    return Question(qid, "unanswerable", f"question {qid}?", "not in the documents", ())


def test_an_unanswerable_question_that_is_refused_counts_as_a_correct_refusal() -> None:
    world = World(min_rerank_score=0.1)
    world.index("Unrelated filler about leases.", score=0.0)  # nothing relevant enough

    overall = evaluate(GoldenSet((unanswerable("u1"),)), world.pipeline, PRINCIPAL).overall

    assert overall.correct_refusal_rate == 1.0


def test_an_unanswerable_question_that_is_answered_is_not_a_correct_refusal() -> None:
    world = World()
    world.index("Unrelated filler about leases.", score=0.9)

    overall = evaluate(GoldenSet((unanswerable("u1"),)), world.pipeline, PRINCIPAL).overall

    assert overall.correct_refusal_rate == 0.0


def test_an_answerable_question_that_is_refused_is_a_false_refusal() -> None:
    world = World(min_rerank_score=0.1)
    world.index("Net sales were 100 billion dollars in 2024.", score=0.0)
    answered = question("q1", span("Net sales were 100 billion"))

    overall = evaluate(GoldenSet((answered,)), world.pipeline, PRINCIPAL).overall

    assert overall.false_refusal_rate == 1.0


def test_an_answerable_question_that_is_answered_is_no_false_refusal() -> None:
    world = World()
    world.index("Net sales were 100 billion dollars in 2024.", score=0.9)
    answered = question("q1", span("Net sales were 100 billion"))

    overall = evaluate(GoldenSet((answered,)), world.pipeline, PRINCIPAL).overall

    assert overall.false_refusal_rate == 0.0
    assert overall.correct_refusal_rate is None  # no unanswerable question was asked


class FailingGenerator:
    def generate(self, question: str, evidence: Sequence[EvidenceChunk]) -> DraftAnswer:
        raise ModelOutputError("not json")


class CountingGenerator(ExtractiveGenerator):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question: str, evidence: Sequence[EvidenceChunk]) -> DraftAnswer:
        self.calls += 1
        return super().generate(question, evidence)


def test_the_unverifiable_rate_is_the_share_of_questions_refused_as_unverifiable() -> None:
    world = World(generator=FailingGenerator())
    world.index("Net sales were 100 billion dollars in 2024.", score=0.9)
    golden = GoldenSet((question("q1", span("Net sales were 100 billion")),))

    report = evaluate(golden, world.pipeline, PRINCIPAL)

    assert report.overall.unverifiable_rate == 1.0
    assert report.questions[0].outcome == "unverifiable"


def test_a_citation_quoting_a_gold_span_is_a_citation_hit() -> None:
    world = World()
    world.index("Net sales were 100 billion dollars. Margins were thin.", score=0.9)
    golden = GoldenSet((question("q1", span("Net sales were 100 billion")),))

    overall = evaluate(golden, world.pipeline, PRINCIPAL).overall

    assert overall.citation_hit_rate == 1.0


def test_a_citation_quoting_other_text_than_the_gold_span_is_not_a_citation_hit() -> None:
    world = World()
    # the extractive generator quotes the first sentence; the Gold Span is in the second
    world.index("Margins were thin. Net sales were 100 billion dollars.", score=0.9)
    golden = GoldenSet((question("q1", span("Net sales were 100 billion")),))

    report = evaluate(golden, world.pipeline, PRINCIPAL)

    assert report.overall.recall_at_8 == 1.0  # the Evidence held it, the answer did not cite it
    assert report.overall.citation_hit_rate == 0.0


def test_a_retrieval_only_run_never_calls_the_generator_and_reports_no_answer_metrics() -> None:
    generator = CountingGenerator()
    world = World(generator=generator)
    world.index("Net sales were 100 billion dollars in 2024.", score=0.9)
    golden = GoldenSet((question("q1", span("Net sales were 100 billion")), unanswerable("u1")))

    overall = evaluate(golden, world.pipeline, PRINCIPAL, retrieval_only=True).overall

    assert generator.calls == 0
    assert overall.recall_at_8 == 1.0
    assert (overall.false_refusal_rate, overall.correct_refusal_rate) == (None, None)
    assert (overall.unverifiable_rate, overall.citation_hit_rate) == (None, None)


def test_the_report_is_broken_down_by_category_and_question_and_serialises_to_json() -> None:
    world = World()
    world.index("Net sales were 100 billion dollars in 2024.", score=0.9)
    golden = GoldenSet(
        (
            question("t1", span("Net sales were 100 billion")),
            question("t2", span("Gross margin was 40 percent")),
            unanswerable("u1"),
        )
    )

    report = evaluate(golden, world.pipeline, PRINCIPAL)
    data = json.loads(report.to_json())

    assert data["overall"]["questions"] == 3
    assert data["by_category"]["text"]["recall_at_8"] == 0.5
    assert data["by_category"]["unanswerable"]["recall_at_8"] is None
    assert data["by_category"]["unanswerable"]["correct_refusal_rate"] == 0.0
    assert [q["id"] for q in data["questions"]] == ["t1", "t2", "u1"]
    assert data["questions"][0]["outcome"] == "answered"


def test_an_unanswerable_question_refused_as_unverifiable_is_not_a_correct_refusal() -> None:
    world = World(generator=FailingGenerator())
    world.index("Unrelated filler about leases.", score=0.9)  # evidence found, draft unusable

    overall = evaluate(GoldenSet((unanswerable("u1"),)), world.pipeline, PRINCIPAL).overall

    assert overall.correct_refusal_rate == 0.0
    assert overall.unverifiable_rate == 1.0


def test_a_citation_hit_is_a_quote_containing_the_gold_span_or_most_of_it() -> None:
    gold = span("Net sales were 100 billion dollars in 2024")

    assert is_citation_hit(gold, "Net sales were 100 billion dollars in 2024, up 2%.")  # contains
    assert is_citation_hit(gold, "Net sales were 100 billion dollars")  # a verbatim half or more
    assert not is_citation_hit(gold, "Net sales")  # trivially short
    assert not is_citation_hit(gold, "Net sales were 100 billion dollars in 2023")  # not verbatim
    assert not is_citation_hit(gold, "")


def test_a_citation_quoting_most_of_a_longer_gold_span_is_a_citation_hit() -> None:
    world = World()
    world.index("Net sales were 100 billion dollars. Margins were thin and falling.", score=0.9)
    gold = span("Net sales were 100 billion dollars. Margins were thin")  # quote: first sentence

    overall = evaluate(GoldenSet((question("q1", gold),)), world.pipeline, PRINCIPAL).overall

    assert overall.citation_hit_rate == 1.0
