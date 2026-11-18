"""Citation validation (ADR-0008): a drafted answer is checked against the Evidence."""

from uuid import UUID, uuid4

from finrag.core.domain import DraftAnswer, DraftCitation, EvidenceChunk, RefusalReason
from finrag.generation.validation import validate_answer

DOCUMENT_ID = uuid4()
CHUNK_ID = UUID("00000000-0000-4000-8000-0000000000c1")

LIQUIDITY = EvidenceChunk(
    chunk_id=CHUNK_ID,
    document_id=DOCUMENT_ID,
    text="Cash and cash equivalents were $1,234 million at year end.",
    section_path=("Part II", "Item 7. MD&A", "Liquidity"),
    page_start=41,
    page_end=42,
)


def test_a_fully_cited_answer_with_verbatim_spans_is_accepted_with_resolved_citations() -> None:
    draft = DraftAnswer(
        text="Cash was $1,234 million at year end [1].",
        citations=(
            DraftCitation(
                marker=1,
                chunk_id=CHUNK_ID,
                quoted_span="Cash and cash equivalents were $1,234 million",
            ),
        ),
    )

    answer = validate_answer(draft, evidence=(LIQUIDITY,))

    assert answer.refusal_reason is None
    assert answer.text == "Cash was $1,234 million at year end [1]."
    (citation,) = answer.citations
    assert citation.marker == 1
    assert citation.document_id == DOCUMENT_ID
    assert citation.section_path == ("Part II", "Item 7. MD&A", "Liquidity")
    assert (citation.page_start, citation.page_end) == (41, 42)
    assert citation.quoted_span == "Cash and cash equivalents were $1,234 million"


def test_a_citation_to_a_chunk_outside_the_evidence_is_refused_as_unverifiable() -> None:
    draft = DraftAnswer(
        text="Cash was $1,234 million at year end [1].",
        citations=(
            DraftCitation(
                marker=1,
                chunk_id=uuid4(),
                quoted_span="Cash and cash equivalents were $1,234 million",
            ),
        ),
    )

    answer = validate_answer(draft, evidence=(LIQUIDITY,))

    assert answer.refusal_reason is RefusalReason.UNVERIFIABLE
    assert answer.citations == ()


def test_a_citation_marker_in_the_text_with_no_citation_is_refused_as_unverifiable() -> None:
    draft = DraftAnswer(
        text="Cash was $1,234 million [1]. Debt was lower [2].",
        citations=(
            DraftCitation(
                marker=1,
                chunk_id=CHUNK_ID,
                quoted_span="Cash and cash equivalents were $1,234 million",
            ),
        ),
    )

    answer = validate_answer(draft, evidence=(LIQUIDITY,))

    assert answer.refusal_reason is RefusalReason.UNVERIFIABLE


def test_a_quoted_span_that_is_not_verbatim_in_its_chunk_is_refused_as_unverifiable() -> None:
    draft = DraftAnswer(
        text="Cash was $1,234 million at year end [1].",
        citations=(
            DraftCitation(
                marker=1,
                chunk_id=CHUNK_ID,
                quoted_span="Cash and cash equivalents totalled $1,234 million",
            ),
        ),
    )

    answer = validate_answer(draft, evidence=(LIQUIDITY,))

    assert answer.refusal_reason is RefusalReason.UNVERIFIABLE


def test_whitespace_differences_between_the_quoted_span_and_the_chunk_are_ignored() -> None:
    draft = DraftAnswer(
        text="Cash was $1,234 million at year end [1].",
        citations=(
            DraftCitation(
                marker=1,
                chunk_id=CHUNK_ID,
                quoted_span="Cash and  cash equivalents\nwere $1,234 million",
            ),
        ),
    )

    assert validate_answer(draft, evidence=(LIQUIDITY,)).refusal_reason is None


CASH_CITATION = DraftCitation(
    marker=1, chunk_id=CHUNK_ID, quoted_span="Cash and cash equivalents were $1,234 million"
)


def test_a_sentence_without_a_citation_marker_is_refused_as_unverifiable() -> None:
    draft = DraftAnswer(
        text="Cash was $1,234 million [1]. The company is well capitalised.",
        citations=(CASH_CITATION,),
    )

    answer = validate_answer(draft, evidence=(LIQUIDITY,))

    assert answer.refusal_reason is RefusalReason.UNVERIFIABLE


def test_a_lead_in_sentence_on_the_allowlist_needs_no_citation() -> None:
    draft = DraftAnswer(
        text="Based on the filings. Cash was $1,234 million [1].",
        citations=(CASH_CITATION,),
    )

    assert validate_answer(draft, evidence=(LIQUIDITY,)).refusal_reason is None


def test_an_allowlisted_phrase_does_not_excuse_a_factual_claim_that_follows_it() -> None:
    draft = DraftAnswer(
        text="Based on the filings, debt fell to $5 million. Cash was $1,234 million [1].",
        citations=(CASH_CITATION,),
    )

    answer = validate_answer(draft, evidence=(LIQUIDITY,))

    assert answer.refusal_reason is RefusalReason.UNVERIFIABLE


def test_a_number_that_is_not_in_a_cited_chunk_is_refused_as_unverifiable() -> None:
    draft = DraftAnswer(text="Cash was $1,500 million at year end [1].", citations=(CASH_CITATION,))

    answer = validate_answer(draft, evidence=(LIQUIDITY,))

    assert answer.refusal_reason is RefusalReason.UNVERIFIABLE


def test_numbers_match_regardless_of_thousands_separators_and_a_leading_dollar_sign() -> None:
    draft = DraftAnswer(text="Cash was 1234 million at year end [1].", citations=(CASH_CITATION,))

    assert validate_answer(draft, evidence=(LIQUIDITY,)).refusal_reason is None


def test_a_number_in_other_units_is_not_converted_and_is_refused() -> None:
    draft = DraftAnswer(text="Cash was $1.234 billion at year end [1].", citations=(CASH_CITATION,))

    answer = validate_answer(draft, evidence=(LIQUIDITY,))

    assert answer.refusal_reason is RefusalReason.UNVERIFIABLE
