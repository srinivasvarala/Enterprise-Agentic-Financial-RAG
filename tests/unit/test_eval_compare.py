"""Seam S5: compare a results file to the Baseline, and bless a run as the new Baseline.

Results files are written by the real run code from fake pipelines elsewhere; here they are small
literals, so the table content is checked against hand-worked numbers.
"""

import json
import os
from pathlib import Path

import pytest

from finrag.evaluation.bless import BlessRefusedError, bless
from finrag.evaluation.compare import (
    NoResultsError,
    compare_report,
    main_compare,
    newest_results,
)


def metrics(**kwargs: float | None) -> dict[str, float | None]:
    base: dict[str, float | None] = {
        "recall_at_8": None,
        "recall_at_100": None,
        "mrr": None,
        "false_refusal_rate": None,
        "correct_refusal_rate": None,
        "unverifiable_rate": None,
        "citation_hit_rate": None,
    }
    return {"questions": 4, **base, **kwargs}  # type: ignore[dict-item]


def judged(**kwargs: float | None) -> dict[str, object]:
    base: dict[str, float | None] = {
        "faithfulness": None,
        "answer_relevancy": None,
        "answer_correctness": None,
    }
    return {"scored": 2, **base, **kwargs}


def results(
    kind: str = "full",
    overall: dict[str, float | None] | None = None,
    text: dict[str, float | None] | None = None,
    judged_overall: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "run": {"kind": kind, "started_at": "2026-11-19T14:30:05+00:00", "git_sha": "abc1234"},
        "models": {"judge": "judge-x"},
        "report": {
            "overall": overall or metrics(),
            "by_category": {"text": text or metrics()},
            "judged": (
                None
                if judged_overall is None
                else {"overall": judged_overall, "by_category": {"text": judged_overall}}
            ),
        },
    }


def write(path: Path, data: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def row(table: str, scope: str, metric: str) -> list[str]:
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[:2] == [scope, metric]:
            return cells
    raise AssertionError(f"no row {scope}/{metric} in:\n{table}")


def test_the_table_has_a_row_per_scope_and_metric_with_hand_worked_deltas() -> None:
    baseline = results(
        overall=metrics(recall_at_8=0.50, mrr=0.40),
        text=metrics(recall_at_8=0.60),
        judged_overall=judged(faithfulness=0.90),
    )
    current = results(
        overall=metrics(recall_at_8=0.75, mrr=0.30),
        text=metrics(recall_at_8=0.50),
        judged_overall=judged(faithfulness=0.90),
    )

    table = compare_report(baseline, current)

    assert "| Scope | Metric | Baseline | Current | Delta |" in table.splitlines()
    assert row(table, "overall", "recall@8")[2:5] == ["0.500", "0.750", "+0.250"]
    assert row(table, "overall", "mrr")[2:5] == ["0.400", "0.300", "-0.100"]
    assert row(table, "text", "recall@8")[2:5] == ["0.600", "0.500", "-0.100"]
    assert row(table, "overall", "faithfulness")[2:5] == ["0.900", "0.900", "+0.000"]
    assert row(table, "text", "faithfulness")[1] == "faithfulness"


def test_a_metric_that_is_null_on_either_side_is_marked_and_has_no_delta() -> None:
    baseline = results(overall=metrics(recall_at_8=0.5))  # retrieval-only: no judged section
    current = results(
        overall=metrics(recall_at_8=0.5, mrr=0.2), judged_overall=judged(faithfulness=1.0)
    )

    table = compare_report(baseline, current)

    assert row(table, "overall", "mrr")[2:5] == ["null", "0.200", "n/a"]
    assert row(table, "overall", "faithfulness")[2:5] == ["null", "1.000", "n/a"]
    assert row(table, "overall", "citation_hit_rate")[2:5] == ["null", "null", "n/a"]


def test_a_category_present_on_only_one_side_shows_null_on_the_other() -> None:
    baseline = results(text=metrics(recall_at_8=0.5))
    current = results(text=metrics(recall_at_8=0.5))
    current["report"]["by_category"]["table"] = metrics(recall_at_8=0.7)  # type: ignore[index]

    table = compare_report(baseline, current)

    assert row(table, "table", "recall@8")[2:5] == ["null", "0.700", "n/a"]


def test_the_table_says_which_runs_were_compared() -> None:
    table = compare_report(results(kind="full"), results(kind="retrieval"))

    assert "baseline: full run, abc1234" in table
    assert "current: retrieval run, abc1234" in table


def test_newest_results_is_chosen_by_the_timestamp_in_the_name(tmp_path: Path) -> None:
    old = write(tmp_path / "retrieval-20261118T100000Z-aaa.json", results())
    new = write(tmp_path / "full-20261119T100000Z-bbb.json", results())
    write(tmp_path / "index.json", {"not": "a results file"})
    os.utime(new, (1, 1))  # file times must not decide it
    os.utime(old, (9, 9))

    assert newest_results(tmp_path) == new


def test_no_results_file_says_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(NoResultsError, match="make eval"):
        newest_results(tmp_path)


def test_a_regression_prints_the_table_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = write(tmp_path / "baseline.json", results(overall=metrics(recall_at_8=0.9)))
    current = write(
        tmp_path / "full-20261119T100000Z-bbb.json", results(overall=metrics(recall_at_8=0.1))
    )

    status = main_compare(["--baseline", str(baseline), "--results", str(current)])

    assert status == 0
    assert "-0.800" in capsys.readouterr().out


def test_without_a_results_argument_the_newest_file_is_compared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = write(tmp_path / "baseline.json", results(overall=metrics(recall_at_8=0.5)))
    write(
        tmp_path / "r" / "full-20261118T100000Z-a.json", results(overall=metrics(recall_at_8=0.1))
    )
    write(
        tmp_path / "r" / "full-20261119T100000Z-b.json", results(overall=metrics(recall_at_8=0.6))
    )

    status = main_compare(["--baseline", str(baseline), "--results-dir", str(tmp_path / "r")])

    assert status == 0
    assert "+0.100" in capsys.readouterr().out


def test_a_missing_baseline_is_an_error_that_says_how_to_bless_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    current = write(tmp_path / "full-20261119T100000Z-b.json", results())

    status = main_compare(["--baseline", str(tmp_path / "none.json"), "--results", str(current)])

    assert status == 1
    assert "eval-bless" in capsys.readouterr().err


def test_bless_copies_the_chosen_full_run_to_the_baseline(tmp_path: Path) -> None:
    source = write(tmp_path / "full-20261119T100000Z-b.json", results(judged_overall=judged()))
    baseline = tmp_path / "evals" / "baseline.json"

    bless(source, baseline)

    assert json.loads(baseline.read_text()) == json.loads(source.read_text())


def test_bless_refuses_a_retrieval_only_run_unless_clearly_requested(tmp_path: Path) -> None:
    source = write(tmp_path / "retrieval-20261119T100000Z-b.json", results(kind="retrieval"))
    baseline = tmp_path / "baseline.json"

    with pytest.raises(BlessRefusedError, match="retrieval-only"):
        bless(source, baseline)
    assert not baseline.exists()

    bless(source, baseline, allow_retrieval_only=True)
    assert baseline.exists()


def test_bless_refuses_a_file_that_is_not_a_results_file(tmp_path: Path) -> None:
    source = write(tmp_path / "index.json", {"manifest_hash": "x"})

    with pytest.raises(BlessRefusedError, match="results file"):
        bless(source, tmp_path / "baseline.json")


def with_depths(data: dict[str, object], search_k: int | None, evidence_k: int | None) -> dict:  # type: ignore[type-arg]
    report = {**data["report"], "search_k": search_k, "evidence_k": evidence_k}  # type: ignore[dict-item]
    return {**data, "report": report}


def test_compare_warns_when_the_two_files_used_different_search_or_evidence_depth() -> None:
    base = with_depths(results(), 100, 8)

    assert "different depths" not in compare_report(base, with_depths(results(), 100, 8))
    assert "search_k 100 vs 80" in compare_report(base, with_depths(results(), 80, 8))
    assert "evidence depth 8 vs 5" in compare_report(base, with_depths(results(), 100, 5))
    # a file that does not record its depth cannot be confirmed comparable
    assert "different depths" in compare_report(base, results())


def test_compare_warns_about_an_incomplete_or_partly_failed_run() -> None:
    incomplete = {**results(), "incomplete": {"reason": "CostCapExceededError"}}
    errored = results()
    errored["report"] = {**errored["report"], "errors": [{"id": "q1"}]}  # type: ignore[dict-item]

    assert "current run is incomplete: CostCapExceededError" in compare_report(
        results(), incomplete
    )
    assert "1 question(s) failed in the current run" in compare_report(results(), errored)


def test_bless_refuses_an_incomplete_run_and_a_run_with_failed_questions(tmp_path: Path) -> None:
    incomplete = tmp_path / "full-1-a-incomplete.json"
    incomplete.write_text(json.dumps({**results(), "incomplete": {"reason": "KeyboardInterrupt"}}))
    errored_data = results()
    errored_data["report"] = {**errored_data["report"], "errors": [{"id": "q1"}]}  # type: ignore[dict-item]
    errored = tmp_path / "full-2-a.json"
    errored.write_text(json.dumps(errored_data))

    with pytest.raises(BlessRefusedError, match="incomplete"):
        bless(incomplete, tmp_path / "baseline.json")
    with pytest.raises(BlessRefusedError, match="failed"):
        bless(errored, tmp_path / "baseline.json")
