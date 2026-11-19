"""Build the eval index: ingest the rendered Filings through the real ingestion pipeline.

Runs inside the `eval` image under the compose `eval` profile, into the eval Tenant, Collection,
Qdrant collection and blob bucket (see `finrag.evaluation.environment`). The build is idempotent
(a second run adds nothing) and ingests the smallest PDF first so a failure on a big one costs
the least. A successful build records the index fingerprint (`finrag.evaluation.fingerprint`).
"""

import argparse
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from finrag.evaluation.fetch import (
    CACHE_DIR,
    MANIFEST_PATH,
    REPO_ROOT,
    ManifestEntry,
    load_manifest,
)
from finrag.evaluation.render import RENDER_RECORD_NAME

RESULTS_DIR = REPO_ROOT / "evals" / "results"  # gitignored


class MissingPdfError(Exception):
    """A manifest Filing has no rendered PDF in the cache yet."""


class IndexBuildError(Exception):
    """The ingestion pipeline did not make a Filing searchable."""


class FilingIndexer(Protocol):
    def index(self, filename: str, pdf: bytes) -> bool:
        """Ingest one PDF Filing. True if it was newly ingested, False if already indexed."""
        ...


@dataclass(frozen=True)
class FilingReport:
    entry: ManifestEntry
    ingested: bool
    seconds: float
    peak_mb: float  # process peak so far: it never falls, so smallest-first makes it the max


def peak_memory_mb() -> float:
    """Peak resident memory of this process so far, in MB."""
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def build_index(
    entries: Sequence[ManifestEntry],
    cache_dir: Path,
    indexer: FilingIndexer,
    *,
    clock: Callable[[], float] = time.monotonic,
    peak_memory_mb: Callable[[], float] = peak_memory_mb,
    on_filing: Callable[[FilingReport], None] | None = None,
) -> list[FilingReport]:
    """Ingest every manifest Filing's rendered PDF, smallest first; report each one."""
    pdfs = {e.accession_number: cache_dir / f"{e.accession_number}.pdf" for e in entries}
    for entry in entries:
        if not pdfs[entry.accession_number].exists():
            raise MissingPdfError(
                f"{entry.company} {entry.form} fiscal {entry.fiscal_year} has no rendered PDF; "
                "run `make eval-fetch` and `make eval-render` first."
            )
    reports: list[FilingReport] = []
    for entry in sorted(entries, key=lambda e: pdfs[e.accession_number].stat().st_size):
        path = pdfs[entry.accession_number]
        started = clock()
        ingested = indexer.index(path.name, path.read_bytes())
        report = FilingReport(
            entry=entry,
            ingested=ingested,
            seconds=round(clock() - started, 2),
            peak_mb=round(peak_memory_mb(), 1),
        )
        reports.append(report)
        if on_filing is not None:
            on_filing(report)
    return reports


def describe(report: FilingReport) -> str:
    e = report.entry
    outcome = "ingested" if report.ingested else "already indexed"
    return (
        f"{outcome} {e.company} {e.form} fiscal {e.fiscal_year}: "
        f"{report.seconds:.1f}s, peak memory {report.peak_mb:.0f} MB"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the eval index from the rendered Filings")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    from finrag.core.config import Settings
    from finrag.evaluation.environment import EvalEnvironmentError, open_pipeline_indexer

    try:
        indexer = open_pipeline_indexer(Settings())
        reports = build_index(
            load_manifest(args.manifest),
            args.cache_dir,
            indexer,
            on_filing=lambda r: print(describe(r), flush=True),
        )
    except (EvalEnvironmentError, MissingPdfError, IndexBuildError) as exc:
        print(f"eval-index refused: {exc}", file=sys.stderr)
        return 1
    from finrag.evaluation.fingerprint import IndexLocations, QdrantPointCounter, record_index

    settings = Settings()
    counter = QdrantPointCounter(
        settings.qdrant_url, settings.qdrant_collection, settings.qdrant_timeout_seconds
    )
    where = IndexLocations(
        manifest=args.manifest,
        render_record=args.cache_dir / RENDER_RECORD_NAME,
        index_record=args.results_dir / "index.json",
    )
    record_index(where, settings, counter)
    added = sum(r.ingested for r in reports)
    print(f"{added} of {len(reports)} Filings newly ingested into the eval index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
