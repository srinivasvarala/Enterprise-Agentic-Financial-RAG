"""Render the cached Filings (inline-XBRL HTML) to PDF so the unchanged PDF path ingests them.

The SEC serves 10-Ks as HTML, but ingestion accepts only PDF and plain text and Gold Spans are
page-based (ADR-0011). The renderer is baked into the `eval` Docker target; rendering never
touches the network. A record next to the PDFs names the renderer and each PDF's page count so
the index fingerprint can read them.
"""

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from finrag.evaluation.fetch import CACHE_DIR, MANIFEST_PATH, ManifestEntry, load_manifest

RENDER_RECORD_NAME = "render.json"
WKHTMLTOPDF = "wkhtmltopdf"


class MissingFilingError(Exception):
    """A manifest Filing has not been fetched into the cache yet."""


class RenderError(Exception):
    """The renderer could not produce a PDF for a Filing."""


class Renderer(Protocol):
    name: str
    version: str

    def render(self, html: Path, pdf: Path) -> None: ...


def render_filings(
    entries: Sequence[ManifestEntry],
    cache_dir: Path,
    renderer: Renderer,
    count_pages: Callable[[Path], int],
    *,
    on_rendered: Callable[[ManifestEntry, float], None] | None = None,
) -> None:
    """Render every cached Filing whose PDF is missing or stale; write the record."""
    renderer_id = {"name": renderer.name, "version": renderer.version}
    previous = _read_record(cache_dir)
    filings: dict[str, dict[str, object]] = dict(previous.get("filings", {}))
    for entry in entries:
        html = cache_dir / f"{entry.accession_number}.htm"
        if not html.exists():
            raise MissingFilingError(
                f"{entry.company} {entry.form} fiscal {entry.fiscal_year} is not in the cache; "
                "run `make eval-fetch` first."
            )
        html_sha256 = hashlib.sha256(html.read_bytes()).hexdigest()
        pdf = cache_dir / f"{entry.accession_number}.pdf"
        known = filings.get(entry.accession_number, {})
        if (
            pdf.exists()
            and previous.get("renderer") == renderer_id
            and known.get("html_sha256") == html_sha256
        ):
            continue
        partial = pdf.with_suffix(".pdf.partial")
        started = time.monotonic()
        try:
            renderer.render(html, partial)
            pages = count_pages(partial)
            partial.replace(pdf)
        finally:
            partial.unlink(missing_ok=True)
        seconds = round(time.monotonic() - started, 2)
        filings[entry.accession_number] = {
            "html_sha256": html_sha256,
            "pages": pages,
            "render_seconds": seconds,
        }
        # Written after every Filing, so a crash keeps the work already done.
        _write_record(cache_dir, {"renderer": renderer_id, "filings": filings})
        if on_rendered is not None:
            on_rendered(entry, seconds)
    _write_record(cache_dir, {"renderer": renderer_id, "filings": filings})


def _read_record(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / RENDER_RECORD_NAME
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _write_record(cache_dir: Path, record: dict[str, object]) -> None:
    (cache_dir / RENDER_RECORD_NAME).write_text(json.dumps(record, indent=2, sort_keys=True))


class WkhtmltopdfRenderer:
    """wkhtmltopdf 0.12.6, JavaScript off, no network (ADR-0011)."""

    name = "wkhtmltopdf"

    def __init__(self) -> None:
        out = subprocess.run(  # noqa: S603 - fixed binary, no user input
            [WKHTMLTOPDF, "--version"], capture_output=True, text=True, check=True
        ).stdout
        self.version = out.split()[1]

    def render(self, html: Path, pdf: Path) -> None:
        # wkhtmltopdf exits 1 when an ignored sub-resource fails (e.g. the SEC's stray
        # tracking-token URL with no protocol) yet still writes the PDF, so a stale PDF is
        # removed first and success means "a non-empty PDF now exists".
        pdf.unlink(missing_ok=True)
        result = subprocess.run(  # noqa: S603 - fixed binary and cache paths, no user input
            [
                WKHTMLTOPDF,
                "--quiet",
                "--disable-javascript",
                "--load-error-handling",
                "ignore",
                "--load-media-error-handling",
                "ignore",
                str(html),
                str(pdf),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if not pdf.exists() or pdf.stat().st_size == 0:
            raise RenderError(
                f"{html.name} could not be rendered to PDF; "
                f"renderer output: {result.stderr.strip()[-300:]}"
            )


def pdf_page_texts(pdf: Path) -> list[str]:
    """Extracted text of every page, in order (used for page counts and re-render checks)."""
    import pypdfium2  # type: ignore[import-untyped]  # no stubs; Docling already depends on it

    document = pypdfium2.PdfDocument(str(pdf))
    try:
        return [document[i].get_textpage().get_text_range() for i in range(len(document))]
    finally:
        document.close()


def count_pages(pdf: Path) -> int:
    return len(pdf_page_texts(pdf))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render cached Filings to PDF")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parser.parse_args(argv)

    def report(entry: ManifestEntry, seconds: float) -> None:
        peak_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
        if sys.platform == "darwin":
            peak_mb /= 1024
        print(
            f"rendered {entry.company} {entry.form} fiscal {entry.fiscal_year}: "
            f"{seconds:.1f}s, renderer peak so far {peak_mb:.0f} MB"
        )

    try:
        render_filings(
            load_manifest(args.manifest),
            args.cache_dir,
            WkhtmltopdfRenderer(),
            count_pages,
            on_rendered=report,
        )
    except (MissingFilingError, RenderError) as exc:
        print(f"eval-render refused: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(
            "eval-render refused: the renderer is not installed here; "
            "run it through `make eval-render`, which uses the `eval` image.",
            file=sys.stderr,
        )
        return 1
    print(f"PDFs and {RENDER_RECORD_NAME} are in {args.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
