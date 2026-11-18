"""Seam 1 for PDF Filings: upload via the API, run the ingestion job with real Docling, observe.

Uses the in-memory harness (sqlite, fake blob store/queue) with the real Docling parser, so it
needs no docker services but does load Docling's models: `make test-int`.
"""

from functools import cache
from pathlib import Path
from typing import Any

import pytest
from harness import Env
from sqlalchemy import select

from finrag.core.errors import ParseError
from finrag.ingestion.docling_parser import DoclingParser
from finrag.ingestion.parsed import ParsedDocument
from finrag.storage.models import Chunk

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent.parent / "fixtures"


@cache
def shared_parser() -> DoclingParser:
    return DoclingParser()  # one instance: Docling loads its models once per process


class CountingParser:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, data: bytes) -> ParsedDocument:
        self.calls += 1
        return shared_parser().parse(data)


def upload_pdf(env: Env, filename: str) -> str:
    data = (FIXTURES / filename).read_bytes()
    response = env.client.post("/documents", files={"file": (filename, data, "application/pdf")})
    assert response.status_code == 202
    doc_id: str = response.json()["id"]
    return doc_id


def get(env: Env, path: str) -> dict[str, Any]:
    body: dict[str, Any] = env.client.get(path).json()
    return body


def test_pdf_filing_reaches_ready_with_item_sections_pages_and_a_table_chunk(env: Env) -> None:
    env.container.parsers = {**env.container.parsers, "application/pdf": shared_parser()}
    doc_id = upload_pdf(env, "example_10k_extract.pdf")

    env.run_jobs()

    document = get(env, f"/documents/{doc_id}")
    assert document["status"] == "ready"
    items = get(env, f"/documents/{doc_id}/chunks")["items"]
    assert document["chunk_count"] == len(items)

    paths = {tuple(c["section_path"]) for c in items}
    assert ("PART I", "Item 1A. Risk Factors", "Risks Related to Our Business") in paths
    assert ("PART II", "Item 7. Management's Discussion and Analysis") in {p[:2] for p in paths}
    # Item 8 starts a new page with no repeated PART heading; it stays under PART II
    assert any(
        p[:2] == ("PART II", "Item 8. Financial Statements and Supplementary Data") for p in paths
    )

    risk = next(c for c in items if "Item 1A. Risk Factors" in c["section_path"])
    assert (risk["page_start"], risk["page_end"]) == (1, 1)
    item7 = [
        c for c in items if "Item 7. Management's Discussion and Analysis" in c["section_path"]
    ]
    assert {c["page_start"] for c in item7} == {2}

    tables = [c for c in items if c["kind"] == "table"]
    assert len(tables) >= 2
    first = tables[0]
    assert first["page_start"] == first["page_end"] == 2
    assert "Total net sales" in first["text"]
    assert "Segment" in first["text"]  # header row kept with the table
    assert all("Form 10-K" not in c["text"] for c in items)  # running footer removed


def test_bounding_boxes_are_persisted_on_chunks(env: Env) -> None:
    env.container.parsers = {**env.container.parsers, "application/pdf": shared_parser()}
    upload_pdf(env, "example_10k_extract.pdf")

    env.run_jobs()

    with env.container.session_factory() as session:
        chunks = session.scalars(select(Chunk).order_by(Chunk.ordinal)).all()
    assert chunks
    assert all(c.bounding_boxes for c in chunks)
    first = chunks[0].bounding_boxes[0]  # type: ignore[index]
    assert first["page"] == 1
    assert first["x0"] < first["x1"]


def test_corrupt_pdf_fails_with_a_readable_reason_and_is_not_retried(env: Env) -> None:
    parser = CountingParser()
    env.container.parsers = {**env.container.parsers, "application/pdf": parser}
    doc_id = upload_pdf(env, "corrupt.pdf")

    env.run_jobs()

    document = get(env, f"/documents/{doc_id}")
    assert document["status"] == "failed"
    assert "PDF" in document["failure_reason"]
    assert "Traceback" not in document["failure_reason"]
    assert parser.calls == 1
    assert env.sleeps == []  # no backoff: parse errors are not transient
    assert get(env, f"/documents/{doc_id}/chunks")["total"] == 0


def test_parse_error_is_the_type_raised_for_corrupt_input() -> None:
    with pytest.raises(ParseError):
        shared_parser().parse((FIXTURES / "corrupt.pdf").read_bytes())
