"""The Golden Set: curated questions with Gold Spans, kept as JSONL data (ADR-0010)."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from finrag.core.domain import ParsedDocument, TableBlock


@dataclass(frozen=True, slots=True)
class TextSpan:
    document: str
    page_start: int
    page_end: int
    text: str


@dataclass(frozen=True, slots=True)
class TableSpan:
    document: str
    page_start: int
    page_end: int
    row_label: str
    figure: str


GoldSpan = TextSpan | TableSpan


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    category: str
    question: str
    reference_answer: str
    gold_spans: tuple[GoldSpan, ...]


@dataclass(frozen=True, slots=True)
class GoldenSet:
    questions: tuple[Question, ...]


@dataclass(frozen=True, slots=True)
class Problem:
    question_id: str
    message: str


CATEGORIES = ("text", "table", "multi-span", "unanswerable", "company-ambiguity")


class GoldenSetError(ValueError):
    """A Golden Set entry is malformed; the message names the question."""


def _field(raw: dict[str, Any], name: str, label: str) -> Any:
    if name not in raw:
        raise GoldenSetError(f"{label}: missing field {name!r}")
    return raw[name]


def _text(raw: dict[str, Any], name: str, label: str) -> str:
    value = _field(raw, name, label)
    if not isinstance(value, str) or not value.strip():
        raise GoldenSetError(f"{label}: field {name!r} must be a non-empty string")
    return value


def _parse_span(raw: Any, label: str) -> GoldSpan:
    if not isinstance(raw, dict):
        raise GoldenSetError(f"{label}: a Gold Span must be an object")
    document = _text(raw, "document", label)
    pages = _field(raw, "pages", label)
    if (
        not isinstance(pages, list)
        or len(pages) not in (1, 2)
        or not all(isinstance(p, int) and p >= 1 for p in pages)
        or pages[0] > pages[-1]
    ):
        raise GoldenSetError(
            f"{label}: 'pages' must be [page] or [first, last] with first <= last, from 1"
        )
    if "row_label" in raw or "figure" in raw:
        return TableSpan(
            document,
            pages[0],
            pages[-1],
            _text(raw, "row_label", label),
            _text(raw, "figure", label),
        )
    return TextSpan(document, pages[0], pages[-1], _text(raw, "text", label))


def _parse_question(row: Any, line_no: int) -> Question:
    if not isinstance(row, dict):
        raise GoldenSetError(f"line {line_no}: an entry must be a JSON object")
    qid = row.get("id")
    label = f"question {qid!r}" if isinstance(qid, str) and qid else f"line {line_no}"
    question_id = _text(row, "id", label)
    category = _text(row, "category", label)
    if category not in CATEGORIES:
        raise GoldenSetError(
            f"{label}: category {category!r} is not one of {', '.join(CATEGORIES)}"
        )
    raw_spans = _field(row, "gold_spans", label)
    if not isinstance(raw_spans, list):
        raise GoldenSetError(f"{label}: 'gold_spans' must be a list")
    if category == "unanswerable" and raw_spans:
        raise GoldenSetError(f"{label}: an unanswerable question must carry no Gold Spans")
    if category != "unanswerable" and not raw_spans:
        raise GoldenSetError(f"{label}: an answerable question needs at least one Gold Span")
    return Question(
        id=question_id,
        category=category,
        question=_text(row, "question", label),
        reference_answer=_text(row, "reference_answer", label),
        gold_spans=tuple(_parse_span(s, label) for s in raw_spans),
    )


def load_golden_set(jsonl: str) -> GoldenSet:
    """Parse JSONL text (one question per line); reject a malformed entry by naming it."""
    questions: list[Question] = []
    seen: set[str] = set()
    for line_no, line in enumerate(jsonl.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenSetError(f"line {line_no}: not valid JSON") from exc
        question = _parse_question(row, line_no)
        if question.id in seen:
            raise GoldenSetError(f"question {question.id!r}: id appears more than once")
        seen.add(question.id)
        questions.append(question)
    return GoldenSet(tuple(questions))


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _page_text(document: ParsedDocument, page: int) -> str:
    parts: list[str] = []
    for block in document.blocks:
        if block.page != page:
            continue
        if isinstance(block, TableBlock):
            parts.extend(p for p in (block.caption, block.header, block.markdown) if p)
        else:
            parts.append(block.text)
    return "\n".join(parts)  # newlines kept: a table row is one line


def _pages_text(document: ParsedDocument, start: int, end: int) -> str:
    return "\n".join(_page_text(document, page) for page in range(start, end + 1))


def _describe(span: GoldSpan) -> str:
    return span.text if isinstance(span, TextSpan) else f"{span.row_label} / {span.figure}"


def check_golden_set(golden: GoldenSet, documents: Mapping[str, ParsedDocument]) -> list[Problem]:
    problems: list[Problem] = []
    for question in golden.questions:
        for span in question.gold_spans:
            document = documents.get(span.document)
            if document is None:
                problems.append(
                    Problem(question.id, f"document {span.document!r} is not in the corpus")
                )
                continue
            if is_hit(span, _pages_text(document, span.page_start, span.page_end)):
                continue
            pages = sorted({b.page for b in document.blocks})
            elsewhere = [p for p in pages if is_hit(span, _page_text(document, p))]
            hint = f"; it appears on page {', '.join(map(str, elsewhere))}" if elsewhere else ""
            problems.append(
                Problem(
                    question.id,
                    f"Gold Span not found in {span.document!r} on pages "
                    f"{span.page_start}-{span.page_end}: {_describe(span)!r}{hint}",
                )
            )
    return problems


def is_hit(span: GoldSpan, chunk_text: str) -> bool:
    """A text Gold Span is a Hit when the chunk contains it. A table Gold Span is a Hit when one
    row of a Markdown table in the chunk holds both its row label and its figure."""
    if isinstance(span, TextSpan):
        return _normalise(span.text) in _normalise(chunk_text)
    label, figure = _normalise(span.row_label).lower(), _normalise(span.figure)
    for line in chunk_text.splitlines():
        row = _normalise(line)
        if row.startswith("|") and label in row.lower() and figure in row:
            return True
    return False


def is_citation_hit(span: GoldSpan, quoted_span: str) -> bool:
    """Does a citation's quote support the Gold Span?

    Yes when the quote contains the whole span (`is_hit`), or, for a text span, when the quote is
    a verbatim part of the span and at least half its length: a correct but shorter quote still
    cites the right passage, while a few words that happen to occur in the span do not.
    """
    if is_hit(span, quoted_span):
        return True
    if not isinstance(span, TextSpan):
        return False
    quote, whole = _normalise(quoted_span), _normalise(span.text)
    return bool(quote) and quote in whole and len(quote) * 2 >= len(whole)


__all__: Sequence[str] = [
    "GoldenSetError",
    "check_golden_set",
    "is_citation_hit",
    "is_hit",
    "load_golden_set",
]
