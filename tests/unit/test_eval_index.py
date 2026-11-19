"""Seam: build the eval index from the manifest's rendered PDFs through a fake indexer."""

from pathlib import Path

import pytest

from finrag.evaluation.fetch import ManifestEntry
from finrag.evaluation.index import FilingReport, MissingPdfError, build_index


class FakeIndexer:
    """Stands in for the ingestion pipeline: remembers which PDFs it already holds."""

    def __init__(self) -> None:
        self.held: set[bytes] = set()
        self.calls: list[str] = []

    def index(self, filename: str, pdf: bytes) -> bool:
        self.calls.append(filename)
        if pdf in self.held:
            return False
        self.held.add(pdf)
        return True


def entry(accession: str) -> ManifestEntry:
    return ManifestEntry(
        company=f"Co {accession}",
        fiscal_year=2025,
        period_end="2025-12-31",
        form="10-K",
        accession_number=accession,
        url=f"https://example.test/{accession}.htm",
        canonicalisation=1,
        canonical_sha256="0" * 64,
    )


def cache_with(tmp_path: Path, pdfs: dict[str, bytes]) -> list[ManifestEntry]:
    for accession, body in pdfs.items():
        (tmp_path / f"{accession}.pdf").write_bytes(body)
    return [entry(a) for a in pdfs]


def test_ingests_the_smallest_pdf_first_whatever_the_manifest_order(tmp_path: Path) -> None:
    entries = cache_with(tmp_path, {"BIG": b"x" * 300, "MID": b"x" * 200, "TINY": b"x" * 10})
    indexer = FakeIndexer()

    reports = build_index(entries, tmp_path, indexer)

    assert indexer.calls == ["TINY.pdf", "MID.pdf", "BIG.pdf"]
    assert [r.entry.accession_number for r in reports] == ["TINY", "MID", "BIG"]


def test_a_second_run_adds_nothing(tmp_path: Path) -> None:
    entries = cache_with(tmp_path, {"A": b"aaa", "B": b"bbbb"})
    indexer = FakeIndexer()

    first = build_index(entries, tmp_path, indexer)
    second = build_index(entries, tmp_path, indexer)

    assert [r.ingested for r in first] == [True, True]
    assert [r.ingested for r in second] == [False, False]


def test_reports_each_filings_duration_and_peak_memory(tmp_path: Path) -> None:
    entries = cache_with(tmp_path, {"A": b"aaa", "B": b"bbbb"})
    ticks = iter([10.0, 12.5, 20.0, 27.0])
    peaks = iter([900.0, 2100.0])
    seen: list[FilingReport] = []

    reports = build_index(
        entries,
        tmp_path,
        FakeIndexer(),
        clock=lambda: next(ticks),
        peak_memory_mb=lambda: next(peaks),
        on_filing=seen.append,
    )

    assert [(r.seconds, r.peak_mb) for r in reports] == [(2.5, 900.0), (7.0, 2100.0)]
    assert seen == reports


def test_a_filing_without_a_rendered_pdf_is_refused_before_anything_is_ingested(
    tmp_path: Path,
) -> None:
    entries = [*cache_with(tmp_path, {"A": b"aaa"}), entry("NOT-RENDERED")]
    indexer = FakeIndexer()

    with pytest.raises(MissingPdfError, match="make eval-render"):
        build_index(entries, tmp_path, indexer)

    assert indexer.calls == []
