"""Fetch the pinned Filings of the eval corpus into an untracked cache (ADR-0010)."""

import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, TypeAdapter

# A generic client name, never a personal contact (ADR-0010).
GENERIC_CLIENT_NAME = "finrag-eval-fetcher"
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "evals" / "manifest.json"
CACHE_DIR = REPO_ROOT / "evals" / "cache"  # gitignored
# The SEC allows 10 requests per second; stay far below it.
MIN_REQUEST_INTERVAL_S = 0.5


class ChecksumMismatchError(Exception):
    """A downloaded Filing does not match the checksum its manifest entry pins."""


class HttpClient(Protocol):
    def get(self, url: str, headers: Mapping[str, str]) -> bytes: ...


class ManifestEntry(BaseModel, frozen=True):
    company: str
    fiscal_year: int
    period_end: str
    form: str
    accession_number: str
    url: str
    sha256: str


def fetch_filings(
    entries: Sequence[ManifestEntry],
    cache_dir: Path,
    http: HttpClient,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    last_request: float | None = None
    for entry in entries:
        target = cache_dir / f"{entry.accession_number}.htm"
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == entry.sha256:
            continue
        if last_request is not None:
            wait = MIN_REQUEST_INTERVAL_S - (clock() - last_request)
            if wait > 0:
                sleep(wait)
        last_request = clock()
        body = http.get(entry.url, {"User-Agent": GENERIC_CLIENT_NAME})
        if hashlib.sha256(body).hexdigest() != entry.sha256:
            raise ChecksumMismatchError(
                f"{entry.company} {entry.form} fiscal {entry.fiscal_year} "
                f"({entry.accession_number}) does not match its pinned checksum; "
                "it was not saved. Check the manifest entry before re-pinning."
            )
        target.write_bytes(body)


def load_manifest(path: Path) -> list[ManifestEntry]:
    return TypeAdapter(list[ManifestEntry]).validate_python(json.loads(path.read_text()))


class HttpxClient:
    def get(self, url: str, headers: Mapping[str, str]) -> bytes:
        response = httpx.get(url, headers=dict(headers), timeout=60, follow_redirects=True)
        response.raise_for_status()
        return response.content


def main() -> int:
    try:
        fetch_filings(load_manifest(MANIFEST_PATH), CACHE_DIR, HttpxClient())
    except ChecksumMismatchError as exc:
        print(f"eval-fetch refused: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(
            f"eval-fetch failed: the SEC download did not complete ({type(exc).__name__}). "
            "Re-run to retry; filings already cached are kept.",
            file=sys.stderr,
        )
        return 1
    print(f"Filings are in {CACHE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
