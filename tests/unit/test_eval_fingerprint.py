"""Seam S4: is the eval index stale? Decided from a recorded fingerprint and a live point count."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from finrag.core.config import Settings
from finrag.evaluation.fingerprint import (
    IndexLocations,
    IndexStatus,
    InvalidIndexError,
    StaleIndexError,
    check_index,
    record_index,
    require_fresh_index,
)


class FakeCounter:
    """Stands in for the eval Qdrant collection."""

    def __init__(self, points: int) -> None:
        self.points = points

    def count_points(self) -> int:
        return self.points


@dataclass
class World:
    where: IndexLocations
    settings: Settings
    counter: FakeCounter

    def check(self) -> IndexStatus:
        return check_index(self.where, self.settings, self.counter)


def write_render_record(path: Path, name: str = "wkhtmltopdf", version: str = "0.12.6") -> None:
    path.write_text(json.dumps({"renderer": {"name": name, "version": version}, "filings": {}}))


@pytest.fixture
def world(tmp_path: Path) -> World:
    where = IndexLocations(
        manifest=tmp_path / "manifest.json",
        render_record=tmp_path / "render.json",
        index_record=tmp_path / "results" / "index.json",
    )
    where.manifest.write_text('[{"company": "Apple"}]')
    write_render_record(where.render_record)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    counter = FakeCounter(points=120)
    record_index(where, settings, counter)
    return World(where, settings, counter)


def test_a_fresh_build_is_ok(world: World) -> None:
    status = world.check()

    assert status.ok
    assert status.reason is None


def test_no_record_means_stale_and_says_to_build(tmp_path: Path) -> None:
    where = IndexLocations(tmp_path / "m.json", tmp_path / "r.json", tmp_path / "index.json")
    where.manifest.write_text("[]")

    status = check_index(where, Settings(_env_file=None), FakeCounter(0))  # type: ignore[call-arg]

    assert not status.ok
    assert status.reason is not None and "no index record" in status.reason


def test_a_changed_manifest_is_stale(world: World) -> None:
    world.where.manifest.write_text('[{"company": "Microsoft"}]')

    status = world.check()

    assert not status.ok
    assert status.reason is not None and "manifest" in status.reason


@pytest.mark.parametrize(
    ("change", "named"),
    [
        ({"chunk_target_tokens": 400}, "chunking"),
        ({"chunk_max_tokens": 700}, "chunking"),
        ({"chunk_overlap_tokens": 10}, "chunking"),
        ({"table_max_tokens": 600}, "chunking"),
        ({"voyage_model": "voyage-3"}, "embedder"),
        ({"voyage_dimension": 512}, "embedder"),
        ({"embedder_backend": "hash"}, "embedder"),
        ({"fastembed_bm25_model": "other/bm25"}, "sparse"),
        ({"sparse_backend": "hash"}, "sparse"),
        ({"pdf_ocr_enabled": True}, "parser"),
    ],
)
def test_a_changed_setting_is_stale_and_names_its_input(
    world: World, change: dict[str, object], named: str
) -> None:
    world.settings = world.settings.model_copy(update=change)

    status = world.check()

    assert not status.ok
    assert status.reason is not None and named in status.reason


def test_a_new_renderer_version_is_stale(world: World) -> None:
    write_render_record(world.where.render_record, version="0.12.7")

    status = world.check()

    assert not status.ok
    assert status.reason is not None and "parser" in status.reason


def test_a_wrong_live_point_count_is_stale_and_shows_both_numbers(world: World) -> None:
    world.counter.points = 95

    status = world.check()

    assert not status.ok
    assert status.reason is not None and "120" in status.reason and "95" in status.reason


def test_recording_creates_the_results_directory_and_stores_the_point_count(world: World) -> None:
    assert json.loads(world.where.index_record.read_text())["expected_points"] == 120


def test_eval_commands_refuse_when_stale_pointing_at_make_eval_index(world: World) -> None:
    world.counter.points = 0

    with pytest.raises(StaleIndexError, match=r"make eval-index"):
        require_fresh_index(world.where, world.settings, world.counter)


def test_eval_commands_proceed_when_ok_and_never_touch_the_index(world: World) -> None:
    before = world.where.index_record.read_text()

    require_fresh_index(world.where, world.settings, world.counter)

    assert world.where.index_record.read_text() == before


class FakeDocuments:
    """Stands in for the eval Tenant's Document rows."""

    def __init__(self, documents: int) -> None:
        self.documents = documents

    def count_documents(self) -> int:
        return self.documents


def test_recording_refuses_when_the_document_count_differs_from_the_manifest(
    world: World,
) -> None:
    # the manifest of the fixture world lists one Filing; two Documents means a duplicated Filing
    with pytest.raises(InvalidIndexError, match=r"2 Documents.*1 Filing"):
        record_index(world.where, world.settings, world.counter, FakeDocuments(2))


def test_a_doubled_index_is_stale_even_when_the_point_count_matches_the_record(
    world: World,
) -> None:
    record_index(world.where, world.settings, world.counter, FakeDocuments(1))
    assert check_index(world.where, world.settings, world.counter, FakeDocuments(1)).ok

    status = check_index(world.where, world.settings, world.counter, FakeDocuments(2))

    assert not status.ok
    assert status.reason is not None and "Documents" in status.reason
