"""The eval index fingerprint: record what shaped the index, and decide whether it is stale.

After a successful `build_index` the build records a fingerprint in `evals/results/index.json`
(gitignored): the manifest hash, the chunking settings, the embedder and sparse model ids, the
parser (PDF parser settings plus the renderer name and version from the render record) and the
expected point count. `check_index` compares that record to the current inputs and to the live
point count of the eval Qdrant collection. The eval commands call `require_fresh_index`, which
refuses when stale and points at `make eval-index`; it never ingests.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from finrag.core.config import Settings
from finrag.evaluation.fetch import CACHE_DIR, MANIFEST_PATH, REPO_ROOT
from finrag.evaluation.render import RENDER_RECORD_NAME

INDEX_RECORD_PATH = REPO_ROOT / "evals" / "results" / "index.json"
REBUILD_HINT = "run `make eval-index` to rebuild it"


class PointCounter(Protocol):
    def count_points(self) -> int:
        """Live number of points in the eval Qdrant collection."""
        ...


class QdrantPointCounter:
    """Live point count of one Qdrant collection (the eval collection)."""

    def __init__(self, url: str, collection: str, timeout_seconds: float) -> None:
        from qdrant_client import QdrantClient

        self._client = QdrantClient(url=url, timeout=int(timeout_seconds))
        self._collection = collection

    def count_points(self) -> int:
        return self._client.count(self._collection, exact=True).count


class StaleIndexError(Exception):
    """The eval index does not match the current inputs; the eval command must not run."""


@dataclass(frozen=True)
class IndexLocations:
    manifest: Path = MANIFEST_PATH
    render_record: Path = CACHE_DIR / RENDER_RECORD_NAME
    index_record: Path = INDEX_RECORD_PATH


@dataclass(frozen=True)
class IndexStatus:
    ok: bool
    reason: str | None = None


def _inputs(where: IndexLocations, settings: Settings) -> dict[str, Any]:
    """Everything that shapes the index, as plain JSON values (excluding the point count)."""
    renderer: Any = None
    if where.render_record.exists():
        renderer = json.loads(where.render_record.read_text()).get("renderer")
    return {
        "manifest": hashlib.sha256(where.manifest.read_bytes()).hexdigest(),
        "chunking": {
            "target_tokens": settings.chunk_target_tokens,
            "max_tokens": settings.chunk_max_tokens,
            "overlap_tokens": settings.chunk_overlap_tokens,
            "table_max_tokens": settings.table_max_tokens,
        },
        "embedder": {
            "backend": settings.embedder_backend,
            "model": settings.voyage_model,
            "dimension": settings.voyage_dimension,
        },
        "sparse": {
            "backend": settings.sparse_backend,
            "model": settings.fastembed_bm25_model,
        },
        "parser": {"ocr_enabled": settings.pdf_ocr_enabled, "renderer": renderer},
    }


def record_index(where: IndexLocations, settings: Settings, counter: PointCounter) -> None:
    """Write the fingerprint of the index just built. Call it only after a successful build."""
    record = {**_inputs(where, settings), "expected_points": counter.count_points()}
    where.index_record.parent.mkdir(parents=True, exist_ok=True)
    where.index_record.write_text(json.dumps(record, indent=2, sort_keys=True))


def check_index(where: IndexLocations, settings: Settings, counter: PointCounter) -> IndexStatus:
    """Ok, or stale with the reason. Read-only."""
    if not where.index_record.exists():
        return IndexStatus(False, "no index record: the eval index has not been built")
    recorded = json.loads(where.index_record.read_text())
    current = _inputs(where, settings)
    for name, value in current.items():
        if recorded.get(name) != value:
            return IndexStatus(
                False,
                f"the {name} changed since the index was built: {recorded.get(name)!r} "
                f"-> {value!r}",
            )
    live = counter.count_points()
    expected = recorded.get("expected_points")
    if live != expected:
        return IndexStatus(
            False, f"the eval collection holds {live} points but the build recorded {expected}"
        )
    return IndexStatus(True)


def require_fresh_index(where: IndexLocations, settings: Settings, counter: PointCounter) -> None:
    """Raise `StaleIndexError` unless the index is ok. For the eval commands; never ingests."""
    status = check_index(where, settings, counter)
    if not status.ok:
        raise StaleIndexError(f"the eval index is stale: {status.reason}; {REBUILD_HINT}.")
