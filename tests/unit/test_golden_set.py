"""Golden Set loading, checking against parsed Documents, and the Hit predicate (ADR-0010)."""

import json

import pytest

from finrag.core.domain import ParagraphBlock, ParsedDocument, TableBlock
from finrag.evaluation.golden import (
    GoldenSetError,
    GoldSpan,
    check_golden_set,
    is_hit,
    load_golden_set,
)


def _jsonl(*rows: dict[str, object]) -> str:
    return "\n".join(json.dumps(r) for r in rows)


def _text_question(
    qid: str = "q1", page: int = 2, text: str = "Net sales were $391 billion."
) -> dict[str, object]:
    return {
        "id": qid,
        "category": "text",
        "question": "What were net sales?",
        "reference_answer": "$391 billion.",
        "gold_spans": [{"document": "apple", "pages": [page], "text": text}],
    }


APPLE = ParsedDocument(
    blocks=(
        ParagraphBlock(text="Intro paragraph.", page=1),
        ParagraphBlock(text="Net sales were  $391\nbillion. Up from last year.", page=2),
    )
)


def test_a_correct_hand_built_set_passes_the_check() -> None:
    golden = load_golden_set(_jsonl(_text_question()))

    assert check_golden_set(golden, {"apple": APPLE}) == []


def test_a_span_absent_from_the_document_is_reported_against_its_question() -> None:
    golden = load_golden_set(
        _jsonl(_text_question("q1"), _text_question("q2", text="Net sales were $999 billion."))
    )

    problems = check_golden_set(golden, {"apple": APPLE})

    assert [p.question_id for p in problems] == ["q2"]
    assert "apple" in problems[0].message
    assert "$999 billion" in problems[0].message


def test_a_span_that_drifted_to_another_page_is_reported_and_says_where_it_is() -> None:
    golden = load_golden_set(_jsonl(_text_question("q1", page=1)))

    problems = check_golden_set(golden, {"apple": APPLE})

    assert [p.question_id for p in problems] == ["q1"]
    assert "page 2" in problems[0].message


def test_a_document_missing_from_the_corpus_is_reported_against_its_question() -> None:
    golden = load_golden_set(_jsonl(_text_question("q1")))

    problems = check_golden_set(golden, {})

    assert [p.question_id for p in problems] == ["q1"]
    assert "apple" in problems[0].message
    assert "not in the corpus" in problems[0].message


def _table_question(
    qid: str = "t1", label: str = "total net sales", figure: str = "391,035"
) -> dict[str, object]:
    return {
        "id": qid,
        "category": "table",
        "question": "What were total net sales?",
        "reference_answer": "$391,035 million.",
        "gold_spans": [
            {"document": "apple", "pages": [3], "row_label": label, "figure": figure},
        ],
    }


APPLE_WITH_TABLE = ParsedDocument(
    blocks=(
        *APPLE.blocks,
        TableBlock(
            page=3,
            header="| | 2024 | 2023 |\n|---|---|---|",
            markdown="| Total net sales | 391,035 | 383,285 |",
        ),
    )
)


def test_a_table_span_is_found_by_row_label_ignoring_case_and_by_figure() -> None:
    golden = load_golden_set(_jsonl(_table_question()))

    assert check_golden_set(golden, {"apple": APPLE_WITH_TABLE}) == []


def test_a_table_span_with_a_wrong_figure_is_reported() -> None:
    golden = load_golden_set(_jsonl(_table_question(figure="999,999")))

    problems = check_golden_set(golden, {"apple": APPLE_WITH_TABLE})

    assert [p.question_id for p in problems] == ["t1"]
    assert "999,999" in problems[0].message


def test_a_text_span_keeps_its_case_but_not_its_whitespace() -> None:
    golden = load_golden_set(_jsonl(_text_question(text="net sales were $391 billion.")))

    assert [p.question_id for p in check_golden_set(golden, {"apple": APPLE})] == ["q1"]


def _span(question: dict[str, object]) -> GoldSpan:
    return load_golden_set(_jsonl(question)).questions[0].gold_spans[0]


def test_a_chunk_is_a_hit_when_it_contains_the_text_span_despite_whitespace() -> None:
    span = _span(_text_question(text="Net sales were $391 billion."))

    assert is_hit(span, "Results.\n\nNet   sales were\n$391 billion. Up 2%.")
    assert not is_hit(span, "Net sales were $390 billion.")


def test_a_chunk_is_a_table_hit_only_with_both_label_and_figure() -> None:
    span = _span(_table_question(label="Total net sales", figure="391,035"))

    assert is_hit(span, "| TOTAL NET SALES | 391,035 | 383,285 |")
    assert not is_hit(span, "| Total net sales | 380,000 |")
    assert not is_hit(span, "| Products | 391,035 |")


def _load_error(*rows: dict[str, object]) -> str:
    with pytest.raises(GoldenSetError) as err:
        load_golden_set(_jsonl(*rows))
    return str(err.value)


def test_loading_rejects_an_unknown_category_naming_the_question() -> None:
    assert "q-bad" in _load_error({**_text_question("q-bad"), "category": "poetry"})


def test_loading_rejects_an_entry_missing_a_field_naming_the_question_and_field() -> None:
    entry = _text_question("q-nofield")
    del entry["reference_answer"]

    message = _load_error(entry)

    assert "q-nofield" in message
    assert "reference_answer" in message


def test_loading_rejects_an_answerable_question_without_gold_spans() -> None:
    assert "q-empty" in _load_error({**_text_question("q-empty"), "gold_spans": []})


def test_loading_rejects_an_unanswerable_question_that_carries_gold_spans() -> None:
    assert "q-un" in _load_error({**_text_question("q-un"), "category": "unanswerable"})


def test_an_unanswerable_question_loads_with_no_gold_spans() -> None:
    entry = {**_text_question("q-un"), "category": "unanswerable", "gold_spans": []}

    question = load_golden_set(_jsonl(entry)).questions[0]

    assert question.category == "unanswerable"
    assert question.gold_spans == ()


def test_loading_rejects_a_malformed_span_and_a_bad_page_range() -> None:
    no_pages = {**_text_question("q-nopages"), "gold_spans": [{"document": "apple", "text": "x"}]}
    backwards = {
        **_text_question("q-back"),
        "gold_spans": [{"document": "apple", "pages": [5, 2], "text": "x"}],
    }
    half_table = {
        **_text_question("q-half"),
        "gold_spans": [{"document": "apple", "pages": [1], "row_label": "Net sales"}],
    }

    assert "q-nopages" in _load_error(no_pages)
    assert "q-back" in _load_error(backwards)
    assert "q-half" in _load_error(half_table)


def test_loading_rejects_duplicate_ids_and_unparseable_lines() -> None:
    assert "q1" in _load_error(_text_question("q1"), _text_question("q1"))
    with pytest.raises(GoldenSetError, match="line 2"):
        load_golden_set(_jsonl(_text_question("q1")) + "\n{not json")


def test_every_problem_is_reported_not_just_the_first() -> None:
    golden = load_golden_set(
        _jsonl(_text_question("q1", text="nope one"), _text_question("q2", text="nope two"))
    )

    assert [p.question_id for p in check_golden_set(golden, {"apple": APPLE})] == ["q1", "q2"]
