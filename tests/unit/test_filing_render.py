"""Seam: render the cached Filings (HTML) to PDF through a fake renderer."""

import hashlib
import json
from pathlib import Path

import pytest

from finrag.evaluation import render
from finrag.evaluation.fetch import ManifestEntry
from finrag.evaluation.render import RENDER_RECORD_NAME, MissingFilingError, render_filings


class FakeRenderer:
    name = "fake-render"

    def __init__(self, version: str = "1.0") -> None:
        self.version = version
        self.calls: list[Path] = []

    def render(self, html: Path, pdf: Path) -> None:
        self.calls.append(html)
        pdf.write_bytes(b"PDF:" + html.read_bytes())


def pages_of(pdf: Path) -> int:
    return 3


def entry(accession: str, body: bytes) -> ManifestEntry:
    return ManifestEntry(
        company=accession,
        fiscal_year=2025,
        period_end="2025-12-31",
        form="10-K",
        accession_number=accession,
        url=f"https://example.test/{accession}.htm",
        canonicalisation=1,
        canonical_sha256=hashlib.sha256(body).hexdigest(),
    )


def cache_with(tmp_path: Path, filings: dict[str, bytes]) -> list[ManifestEntry]:
    for accession, body in filings.items():
        (tmp_path / f"{accession}.htm").write_bytes(body)
    return [entry(a, b) for a, b in filings.items()]


def record(tmp_path: Path) -> dict[str, object]:
    return json.loads((tmp_path / RENDER_RECORD_NAME).read_text())


def test_renders_each_cached_filing_and_records_renderer_and_page_count(tmp_path: Path) -> None:
    entries = cache_with(tmp_path, {"A-1": b"<html>a</html>", "B-2": b"<html>b</html>"})

    render_filings(entries, tmp_path, FakeRenderer("1.2.3"), pages_of)

    assert (tmp_path / "A-1.pdf").read_bytes() == b"PDF:<html>a</html>"
    assert (tmp_path / "B-2.pdf").exists()
    rec = record(tmp_path)
    assert rec["renderer"] == {"name": "fake-render", "version": "1.2.3"}
    filings = rec["filings"]
    assert isinstance(filings, dict)
    assert filings["A-1"]["pages"] == 3
    assert filings["B-2"]["html_sha256"] == hashlib.sha256(b"<html>b</html>").hexdigest()


def test_second_run_renders_nothing(tmp_path: Path) -> None:
    entries = cache_with(tmp_path, {"A-1": b"<html>a</html>"})
    render_filings(entries, tmp_path, FakeRenderer(), pages_of)

    again = FakeRenderer()
    render_filings(entries, tmp_path, again, pages_of)

    assert again.calls == []


def test_changed_html_is_rendered_again(tmp_path: Path) -> None:
    entries = cache_with(tmp_path, {"A-1": b"<html>a</html>"})
    render_filings(entries, tmp_path, FakeRenderer(), pages_of)
    (tmp_path / "A-1.htm").write_bytes(b"<html>changed</html>")

    again = FakeRenderer()
    render_filings(entries, tmp_path, again, pages_of)

    assert len(again.calls) == 1
    assert (tmp_path / "A-1.pdf").read_bytes() == b"PDF:<html>changed</html>"


def test_new_renderer_version_renders_again_and_is_recorded(tmp_path: Path) -> None:
    entries = cache_with(tmp_path, {"A-1": b"<html>a</html>"})
    render_filings(entries, tmp_path, FakeRenderer("1.0"), pages_of)

    newer = FakeRenderer("2.0")
    render_filings(entries, tmp_path, newer, pages_of)

    assert len(newer.calls) == 1
    assert record(tmp_path)["renderer"] == {"name": "fake-render", "version": "2.0"}


def test_missing_pdf_is_rendered_again(tmp_path: Path) -> None:
    entries = cache_with(tmp_path, {"A-1": b"<html>a</html>"})
    render_filings(entries, tmp_path, FakeRenderer(), pages_of)
    (tmp_path / "A-1.pdf").unlink()

    again = FakeRenderer()
    render_filings(entries, tmp_path, again, pages_of)

    assert len(again.calls) == 1


def test_an_uncached_filing_is_refused_with_the_next_step(tmp_path: Path) -> None:
    entries = [entry("A-1", b"x")]

    with pytest.raises(MissingFilingError, match="make eval-fetch"):
        render_filings(entries, tmp_path, FakeRenderer(), pages_of)


def test_a_failed_render_leaves_no_pdf_that_counts_as_up_to_date(tmp_path: Path) -> None:
    entries = cache_with(tmp_path, {"A-1": b"<html>a</html>"})

    class Broken(FakeRenderer):
        def render(self, html: Path, pdf: Path) -> None:
            pdf.write_bytes(b"half")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        render_filings(entries, tmp_path, Broken(), pages_of)

    again = FakeRenderer()
    render_filings(entries, tmp_path, again, pages_of)
    assert len(again.calls) == 1


def test_main_with_a_missing_manifest_names_the_manifest_not_the_renderer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = render.main(["--manifest", str(tmp_path / "nope.json"), "--cache-dir", str(tmp_path)])

    err = capsys.readouterr().err
    assert code == 1
    assert "nope.json" in err and "does not exist" in err
    assert "renderer is not installed" not in err


def test_main_without_the_renderer_installed_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    monkeypatch.setattr(render.shutil, "which", lambda name: None)

    code = render.main(["--manifest", str(manifest), "--cache-dir", str(tmp_path)])

    assert code == 1
    assert "renderer is not installed" in capsys.readouterr().err


def test_main_with_another_missing_file_names_that_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    monkeypatch.setattr(render.shutil, "which", lambda name: "/usr/bin/wkhtmltopdf")

    def boom(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError(2, "No such file", "/evals/cache/x.html")

    monkeypatch.setattr(render, "WkhtmltopdfRenderer", lambda: object())
    monkeypatch.setattr(render, "render_filings", boom)

    code = render.main(["--manifest", str(manifest), "--cache-dir", str(tmp_path)])

    err = capsys.readouterr().err
    assert code == 1
    assert "/evals/cache/x.html does not exist" in err
    assert "renderer is not installed" not in err
