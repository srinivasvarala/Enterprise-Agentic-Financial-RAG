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
from uuid import UUID

from finrag.core.config import Settings
from finrag.evaluation.fetch import CACHE_DIR, MANIFEST_PATH, REPO_ROOT
from finrag.evaluation.render import RENDER_RECORD_NAME

INDEX_RECORD_PATH = REPO_ROOT / "evals" / "results" / "index.json"
REBUILD_HINT = "run `make eval-index` to rebuild it"


class PointCounter(Protocol):
    def count_points(self) -> int:
        """Live number of points in the eval Qdrant collection."""
        ...


class DocumentCounter(Protocol):
    def count_documents(self) -> int:
        """Live number of Documents in the eval Tenant."""
        ...


class InvalidIndexError(Exception):
    """The index just built is not a valid eval index (for example a Filing is duplicated)."""


class SqlDocumentCounter:
    """Live count of the eval Tenant's Documents (tenant-scoped)."""

    def __init__(self, session_factory: Any, tenant_id: UUID) -> None:
        self._session_factory = session_factory
        self._tenant_id = tenant_id

    @classmethod
    def from_settings(cls, settings: Settings) -> "SqlDocumentCounter":
        from finrag.storage.db import build_engine, build_session_factory

        return cls(build_session_factory(build_engine(settings)), settings.default_tenant_id)

    def count_documents(self) -> int:
        from sqlalchemy import func, select

        from finrag.storage.models import Document

        with self._session_factory() as session:
            return (
                session.scalar(
                    select(func.count())
                    .select_from(Document)
                    .where(Document.tenant_id == self._tenant_id)
                )
                or 0
            )


def _filings(where: "IndexLocations") -> int:
    return len(json.loads(where.manifest.read_text()))


def _document_mismatch(filings: int, documents: int) -> str | None:
    if documents == filings:
        return None
    return (
        f"the eval Tenant holds {documents} Documents but the manifest lists {filings} "
        f"Filing{'s' if filings != 1 else ''} "
        "(a duplicated index needs `make eval-index REBUILD=1`)"
    )


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


def record_index(
    where: IndexLocations,
    settings: Settings,
    counter: PointCounter,
    documents: DocumentCounter | None = None,
) -> None:
    """Write the fingerprint of the index just built. Call it only after a successful build.

    With `documents`, refuses (`InvalidIndexError`) unless there is exactly one Document per
    manifest Filing, so a duplicated Filing can never be recorded as the baseline index.
    """
    record: dict[str, Any] = {**_inputs(where, settings), "expected_points": counter.count_points()}
    if documents is not None:
        live_documents = documents.count_documents()
        mismatch = _document_mismatch(_filings(where), live_documents)
        if mismatch is not None:
            raise InvalidIndexError(mismatch)
        record["expected_documents"] = live_documents
    where.index_record.parent.mkdir(parents=True, exist_ok=True)
    where.index_record.write_text(json.dumps(record, indent=2, sort_keys=True))


def check_index(
    where: IndexLocations,
    settings: Settings,
    counter: PointCounter,
    documents: DocumentCounter | None = None,
) -> IndexStatus:
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
    if documents is not None:
        mismatch = _document_mismatch(_filings(where), documents.count_documents())
        if mismatch is not None:
            return IndexStatus(False, mismatch)
    live = counter.count_points()
    expected = recorded.get("expected_points")
    if live != expected:
        return IndexStatus(
            False, f"the eval collection holds {live} points but the build recorded {expected}"
        )
    return IndexStatus(True)


def require_fresh_index(
    where: IndexLocations,
    settings: Settings,
    counter: PointCounter,
    documents: DocumentCounter | None = None,
) -> None:
    """Raise `StaleIndexError` unless the index is ok. For the eval commands; never ingests."""
    status = check_index(where, settings, counter, documents)
    if not status.ok:
        raise StaleIndexError(f"the eval index is stale: {status.reason}; {REBUILD_HINT}.")
