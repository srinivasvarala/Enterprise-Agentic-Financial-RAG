"""Evaluate a Golden Set against a pipeline into a Report (ADR-0010).

Retrieval metrics are judged with `is_hit` on chunk text, so they survive any chunking change.
A Refusal with reason `insufficient_evidence` is the right outcome for an unanswerable question.
"""

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from finrag.core.config import Settings
from finrag.core.domain import Answer, EvidenceChunk, Principal, RefusalReason
from finrag.core.interfaces import AnswerGenerator
from finrag.evaluation.golden import GoldenSet, GoldSpan, Question, is_citation_hit, is_hit
from finrag.evaluation.judge import Judge, JudgeCase, JudgeScores
from finrag.generation.service import AnswerService
from finrag.retrieval.retriever import RetrievalTrace, Retriever

EVIDENCE_K = 8  # default recall@8 depth; a real run passes `settings.rerank_top_n`
SEARCH_K = 100  # default recall@100 depth; a real run passes `search_depth(settings)`
ANSWERED = "answered"


class JudgeFailureRateExceededError(Exception):
    """Too many questions could not be judged for the run's scores to mean anything."""

    def __init__(self, *, failures: int, questions: int, allowed_rate: float) -> None:
        super().__init__(
            f"stopped: the judge failed on {failures} of {questions} questions eligible for "
            f"judging, over the allowed {allowed_rate:.0%}. Check the judge model and the "
            "Anthropic key, then rerun."
        )
        self.failures = failures
        self.questions = questions


def search_depth(settings: Settings) -> int:
    """Candidates the hybrid search asks for (dense plus sparse). Fusion may return fewer."""
    return settings.retrieve_dense_k + settings.retrieve_sparse_k


class QuestionFailureRateExceededError(Exception):
    """Too many questions failed in the pipeline for the run's metrics to mean anything."""

    def __init__(self, *, errors: int, questions: int, allowed_rate: float) -> None:
        super().__init__(
            f"stopped: {errors} of {questions} questions failed (a provider error), over the "
            f"allowed {allowed_rate:.0%}. Check the providers and the keys, then rerun."
        )
        self.errors = errors
        self.questions = questions


@dataclass(frozen=True, slots=True)
class PipelineResult:
    trace: RetrievalTrace
    answer: Answer | None  # None for a retrieval-only run


class EvalPipeline(Protocol):
    """What `evaluate` measures: retrieval traced, and optionally the answer on top of it."""

    def run(self, principal: Principal, question: str, *, answer: bool) -> PipelineResult: ...


class _CapturingRetriever:
    """Lets AnswerService retrieve while we keep the trace it produced."""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self.trace: RetrievalTrace | None = None

    def retrieve(self, principal: Principal, question: str) -> tuple[EvidenceChunk, ...]:
        self.trace = self._retriever.retrieve_traced(principal, question)
        return self.trace.evidence


class EvaluationPipeline:
    """The real retrieval and answering, with the retrieval traced."""

    def __init__(self, retriever: Retriever, generator: AnswerGenerator) -> None:
        self._retriever = retriever
        self._generator = generator

    def run(self, principal: Principal, question: str, *, answer: bool = True) -> PipelineResult:
        if not answer:
            return PipelineResult(self._retriever.retrieve_traced(principal, question), None)
        capture = _CapturingRetriever(self._retriever)
        result = AnswerService(capture, self._generator).answer(principal, question)
        assert capture.trace is not None  # AnswerService always retrieves first
        return PipelineResult(capture.trace, result)


@dataclass(frozen=True, slots=True)
class QuestionResult:
    id: str
    category: str
    recall_at_8: float | None  # None for an unanswerable question (no Gold Spans)
    recall_at_100: float | None
    reciprocal_rank: float | None
    outcome: str | None  # "answered" or the Refusal reason; None for a retrieval-only run
    citation_hit: bool | None  # None unless an answerable question was answered


@dataclass(frozen=True, slots=True)
class Metrics:
    """Each metric is None when nothing it is measured over was asked (or run)."""

    questions: int
    recall_at_8: float | None
    recall_at_100: float | None
    mrr: float | None
    false_refusal_rate: float | None  # of answerable questions
    correct_refusal_rate: float | None  # of unanswerable questions
    unverifiable_rate: float | None  # of all questions
    citation_hit_rate: float | None  # of answerable questions that were answered


@dataclass(frozen=True, slots=True)
class JudgedQuestion:
    id: str
    category: str
    faithfulness: float | None
    answer_relevancy: float | None
    answer_correctness: float | None
    error: str | None = None  # the exception class when the judge failed; never its message


@dataclass(frozen=True, slots=True)
class JudgedMetrics:
    """Mean judged scores over the questions scored; a measure is None if no score exists."""

    scored: int
    faithfulness: float | None
    answer_relevancy: float | None
    answer_correctness: float | None


@dataclass(frozen=True, slots=True)
class JudgedSection:
    """Judged answer quality: only answered, answerable questions are judged."""

    judge_model: str
    not_judged: int  # answered, answerable questions whose judging failed
    overall: JudgedMetrics
    by_category: dict[str, JudgedMetrics]
    questions: tuple[JudgedQuestion, ...]


@dataclass(frozen=True, slots=True)
class QuestionError:
    """A question the pipeline failed on. Excluded from every metric; only the class is kept."""

    id: str
    category: str
    error: str  # the exception class, never its message


@dataclass(frozen=True, slots=True)
class Report:
    overall: Metrics
    by_category: dict[str, Metrics]
    questions: tuple[QuestionResult, ...]
    judged: JudgedSection | None = None  # None when no judge ran
    errors: tuple[QuestionError, ...] = ()  # questions that failed, excluded from the metrics
    search_k: int = SEARCH_K  # the recall@100 depth this run used
    evidence_k: int = EVIDENCE_K  # the recall@8 depth this run used

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _recall(spans: Sequence[GoldSpan], chunks: Sequence[EvidenceChunk]) -> float:
    found = sum(1 for span in spans if any(is_hit(span, c.text) for c in chunks))
    return found / len(spans)


def _reciprocal_rank(spans: Sequence[GoldSpan], trace: RetrievalTrace) -> float:
    for rank, item in enumerate(trace.reranked, start=1):
        if any(is_hit(span, item.chunk.text) for span in spans):
            return 1 / rank
    return 0.0


def _evaluate_question(
    q: Question, result: PipelineResult, *, search_k: int, evidence_k: int
) -> QuestionResult:
    answer, trace = result.answer, result.trace
    outcome = None
    if answer is not None:
        outcome = ANSWERED if answer.refusal_reason is None else answer.refusal_reason.value
    cited = None
    if answer is not None and answer.refusal_reason is None and q.gold_spans:
        cited = any(
            is_citation_hit(s, c.quoted_span) for s in q.gold_spans for c in answer.citations
        )
    if not q.gold_spans:
        return QuestionResult(q.id, q.category, None, None, None, outcome, cited)
    return QuestionResult(
        q.id,
        q.category,
        _recall(q.gold_spans, trace.evidence[:evidence_k]),
        _recall(q.gold_spans, [c.chunk for c in trace.candidates[:search_k]]),
        _reciprocal_rank(q.gold_spans, trace),
        outcome,
        cited,
    )


def _mean(values: Sequence[float | bool | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _metrics(results: Sequence[QuestionResult]) -> Metrics:
    answerable = [r for r in results if r.recall_at_8 is not None]
    unanswerable = [r for r in results if r.recall_at_8 is None]
    judged = [r for r in results if r.outcome is not None]
    return Metrics(
        questions=len(results),
        recall_at_8=_mean([r.recall_at_8 for r in answerable]),
        recall_at_100=_mean([r.recall_at_100 for r in answerable]),
        mrr=_mean([r.reciprocal_rank for r in answerable]),
        false_refusal_rate=_mean([r.outcome != ANSWERED for r in answerable if r.outcome]),
        correct_refusal_rate=_mean(
            [r.outcome == RefusalReason.INSUFFICIENT_EVIDENCE for r in unanswerable if r.outcome]
        ),
        unverifiable_rate=_mean([r.outcome == RefusalReason.UNVERIFIABLE for r in judged]),
        citation_hit_rate=_mean([r.citation_hit for r in answerable]),
    )


def _judge_case(q: Question, result: PipelineResult) -> JudgeCase | None:
    """What to judge: an answered, answerable question. A Refusal has no answer to score."""
    answer = result.answer
    if answer is None or answer.refusal_reason is not None or not q.gold_spans:
        return None
    return JudgeCase(
        question=q.question,
        answer=answer.text,
        reference_answer=q.reference_answer,
        evidence=tuple(c.text for c in result.trace.evidence),
    )


def _judged_metrics(items: Sequence[JudgedQuestion]) -> JudgedMetrics:
    return JudgedMetrics(
        scored=sum(1 for i in items if i.error is None),
        faithfulness=_mean([i.faithfulness for i in items]),
        answer_relevancy=_mean([i.answer_relevancy for i in items]),
        answer_correctness=_mean([i.answer_correctness for i in items]),
    )


def _judged_section(judge: Judge, items: Sequence[JudgedQuestion]) -> JudgedSection:
    groups: dict[str, list[JudgedQuestion]] = defaultdict(list)
    for item in items:
        groups[item.category].append(item)
    return JudgedSection(
        judge_model=judge.model_id,
        not_judged=sum(1 for i in items if i.error is not None),
        overall=_judged_metrics(items),
        by_category={c: _judged_metrics(g) for c, g in sorted(groups.items())},
        questions=tuple(items),
    )


@dataclass(slots=True)
class RunState:
    """What a run has produced so far. Kept by the caller so a run that stops still has it."""

    results: list[QuestionResult] = field(default_factory=list)
    judged: list[JudgedQuestion] = field(default_factory=list)
    errors: list[QuestionError] = field(default_factory=list)
    not_judgeable: int = 0  # answerable questions that errored or were refused: never judged
    asked: int = 0  # questions started, errored ones included


def build_report(
    state: RunState,
    *,
    judge: Judge | None,
    retrieval_only: bool = False,
    search_k: int = SEARCH_K,
    evidence_k: int = EVIDENCE_K,
) -> Report:
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in state.results:
        groups[r.category].append(r)
    return Report(
        overall=_metrics(state.results),
        by_category={c: _metrics(rs) for c, rs in sorted(groups.items())},
        questions=tuple(state.results),
        judged=(
            _judged_section(judge, state.judged)
            if judge is not None and not retrieval_only
            else None
        ),
        errors=tuple(state.errors),
        search_k=search_k,
        evidence_k=evidence_k,
    )


def evaluate(
    golden: GoldenSet,
    pipeline: EvalPipeline,
    principal: Principal,
    *,
    retrieval_only: bool = False,
    judge: Judge | None = None,
    after_question: Callable[[int], None] | None = None,
    max_judge_failure_rate: float = 1.0,
    max_question_error_rate: float = 1.0,
    state: RunState | None = None,
    search_k: int = SEARCH_K,
    evidence_k: int = EVIDENCE_K,
) -> Report:
    """Ask every question through the pipeline; report overall, per category and per question.

    A retrieval-only run never calls the generator or the judge, so the answer and judged metrics
    are None. `after_question(done)` runs after each question; it may raise to stop the run (the
    cost cap does). Pass a `state` to keep what was done when the run stops by an exception.

    A question the pipeline fails on (a provider error) is recorded as an error, with the
    exception class only, and left out of every metric; the run goes on. A judge failure on one
    question is recorded (null scores, the error class) and the run goes on; once failures exceed
    `max_question_error_rate` of the Golden Set the run stops with
    `QuestionFailureRateExceededError`. Judge failures are counted against
    `max_judge_failure_rate` of the questions eligible for judging (answerable and answered, as
    far as the run has seen) the run stops.
    """
    state = state if state is not None else RunState()
    answerable = sum(1 for q in golden.questions if q.gold_spans)
    for done, q in enumerate(golden.questions, start=1):
        state.asked = done
        try:
            result = pipeline.run(principal, q.question, answer=not retrieval_only)
        except Exception as exc:  # a provider failure on this question; only the class is kept
            state.errors.append(QuestionError(q.id, q.category, type(exc).__name__))
            if len(state.errors) / len(golden.questions) > max_question_error_rate:
                raise QuestionFailureRateExceededError(
                    errors=len(state.errors),
                    questions=len(golden.questions),
                    allowed_rate=max_question_error_rate,
                ) from None
            if q.gold_spans:
                state.not_judgeable += 1
        else:
            state.results.append(
                _evaluate_question(q, result, search_k=search_k, evidence_k=evidence_k)
            )
            case = _judge_case(q, result) if judge is not None else None
            if q.gold_spans and result.answer is not None and case is None:
                state.not_judgeable += 1  # refused: there is no answer to judge
            if judge is not None and case is not None:
                try:
                    scores: JudgeScores = judge.judge(case)
                except Exception as exc:  # any provider failure; only the class is kept
                    state.judged.append(
                        JudgedQuestion(q.id, q.category, None, None, None, type(exc).__name__)
                    )
                else:
                    state.judged.append(
                        JudgedQuestion(
                            q.id,
                            q.category,
                            scores.faithfulness,
                            scores.answer_relevancy,
                            scores.answer_correctness,
                        )
                    )
        failures = sum(1 for j in state.judged if j.error is not None)
        eligible = max(answerable - state.not_judgeable, 1)
        if failures and failures / eligible > max_judge_failure_rate:
            raise JudgeFailureRateExceededError(
                failures=failures, questions=eligible, allowed_rate=max_judge_failure_rate
            )
        if after_question is not None:
            after_question(done)
    return build_report(
        state, judge=judge, retrieval_only=retrieval_only, search_k=search_k, evidence_k=evidence_k
    )
