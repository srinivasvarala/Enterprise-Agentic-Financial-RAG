"""Seam S3: fetch the manifest's Filings into a cache through a fake HTTP client."""

import hashlib
import itertools
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from finrag.evaluation.fetch import (
    GENERIC_CLIENT_NAME,
    MANIFEST_PATH,
    ChecksumMismatchError,
    ManifestEntry,
    fetch_filings,
    load_manifest,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeHttp:
    """Serves canned bytes and records every request with the time the clock said."""

    def __init__(self, pages: dict[str, bytes], clock: FakeClock) -> None:
        self.pages = pages
        self.clock = clock
        self.requests: list[tuple[float, str, Mapping[str, str]]] = []

    def get(self, url: str, headers: Mapping[str, str]) -> bytes:
        self.requests.append((self.clock.now, url, dict(headers)))
        return self.pages[url]


def entry(company: str, body: bytes) -> ManifestEntry:
    return ManifestEntry(
        company=company,
        fiscal_year=2025,
        period_end="2025-12-31",
        form="10-K",
        accession_number=f"0000000000-25-{len(company):06d}",
        url=f"https://sec.example/{company.lower()}.htm",
        sha256=hashlib.sha256(body).hexdigest(),
    )


def test_the_committed_manifest_pins_the_five_company_filings() -> None:
    entries = load_manifest(MANIFEST_PATH)

    assert [e.company for e in entries] == ["Apple", "Microsoft", "Alphabet", "Walmart", "Nike"]
    assert {e.form for e in entries} == {"10-K"}
    for e in entries:
        assert re.fullmatch(r"\d{10}-\d{2}-\d{6}", e.accession_number)
        assert re.fullmatch(r"[0-9a-f]{64}", e.sha256)
        assert e.url.startswith("https://www.sec.gov/Archives/edgar/data/")
        assert e.accession_number.replace("-", "") in e.url


def test_downloads_each_filing_into_the_cache_with_a_generic_client_name(tmp_path: Path) -> None:
    clock = FakeClock()
    bodies = {"Acme": b"<html>acme 10-K</html>", "Globex": b"<html>globex 10-K</html>"}
    entries = [entry(name, body) for name, body in bodies.items()]
    http = FakeHttp({e.url: bodies[e.company] for e in entries}, clock)

    fetch_filings(entries, tmp_path, http, clock=clock.monotonic, sleep=clock.sleep)

    for e in entries:
        assert (tmp_path / f"{e.accession_number}.htm").read_bytes() == bodies[e.company]
    agents = {headers["User-Agent"] for _, _, headers in http.requests}
    assert agents == {GENERIC_CLIENT_NAME}
    assert "@" not in GENERIC_CLIENT_NAME


def test_a_valid_cached_filing_is_not_downloaded_again(tmp_path: Path) -> None:
    clock = FakeClock()
    acme = entry("Acme", b"<html>acme</html>")
    http = FakeHttp({acme.url: b"<html>acme</html>"}, clock)

    fetch_filings([acme], tmp_path, http, clock=clock.monotonic, sleep=clock.sleep)
    fetch_filings([acme], tmp_path, http, clock=clock.monotonic, sleep=clock.sleep)

    assert len(http.requests) == 1


def test_a_corrupt_cached_filing_is_downloaded_again(tmp_path: Path) -> None:
    clock = FakeClock()
    acme = entry("Acme", b"<html>acme</html>")
    (tmp_path / f"{acme.accession_number}.htm").write_bytes(b"truncated")
    http = FakeHttp({acme.url: b"<html>acme</html>"}, clock)

    fetch_filings([acme], tmp_path, http, clock=clock.monotonic, sleep=clock.sleep)

    assert (tmp_path / f"{acme.accession_number}.htm").read_bytes() == b"<html>acme</html>"


def test_requests_are_spaced_well_under_the_sec_limit_of_ten_per_second(tmp_path: Path) -> None:
    clock = FakeClock()
    bodies = {name: f"<html>{name}</html>".encode() for name in ("Acme", "Globex", "Initech")}
    entries = [entry(name, body) for name, body in bodies.items()]
    http = FakeHttp({e.url: bodies[e.company] for e in entries}, clock)

    fetch_filings(entries, tmp_path, http, clock=clock.monotonic, sleep=clock.sleep)

    times = [at for at, _, _ in http.requests]
    gaps = [later - earlier for earlier, later in itertools.pairwise(times)]
    assert len(gaps) == 2
    assert all(gap >= 0.5 for gap in gaps)  # at most 2 requests per second


def test_refuses_a_checksum_mismatch_naming_the_filing_and_keeps_it_out_of_the_cache(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    pinned = entry("Acme", b"<html>the pinned text</html>")
    http = FakeHttp({pinned.url: b"<html>silently changed</html>"}, clock)

    with pytest.raises(ChecksumMismatchError) as caught:
        fetch_filings([pinned], tmp_path, http, clock=clock.monotonic, sleep=clock.sleep)

    assert "Acme" in str(caught.value)
    assert pinned.accession_number in str(caught.value)
    assert list(tmp_path.iterdir()) == []
