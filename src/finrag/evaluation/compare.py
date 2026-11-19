"""`make eval-compare`: the delta of a results file against the committed Baseline (ADR-0010).

Prints a Markdown table, per metric and per category, ready to paste into a pull request. M3 is
report-only: the exit status is 0 whatever the deltas are. A metric that is null in either file
(not measured, e.g. a retrieval-only run has no judged scores) shows `null` and has no delta.

    uv run python -m finrag.evaluation.compare [--results FILE | --results-dir DIR]
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from finrag.evaluation.fetch import REPO_ROOT
from finrag.evaluation.index import RESULTS_DIR

BASELINE_PATH = REPO_ROOT / "evals" / "baseline.json"
RESULTS_GLOB = ("full-*.json", "retrieval-*.json")

# (label, key in the Metrics section) and (label, key in the judged section)
RETRIEVAL_METRICS = (
    ("recall@8", "recall_at_8"),
    ("recall@100", "recall_at_100"),
    ("mrr", "mrr"),
    ("false_refusal_rate", "false_refusal_rate"),
    ("correct_refusal_rate", "correct_refusal_rate"),
    ("unverifiable_rate", "unverifiable_rate"),
    ("citation_hit_rate", "citation_hit_rate"),
)
JUDGED_METRICS = (
    ("faithfulness", "faithfulness"),
    ("answer_relevancy", "answer_relevancy"),
    ("answer_correctness", "answer_correctness"),
)


class NoResultsError(Exception):
    """There is no results file to compare."""


def load_results(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def newest_results(results_dir: Path) -> Path:
    """The results file with the latest UTC timestamp in its name, not its file time."""
    found = [p for pattern in RESULTS_GLOB for p in results_dir.glob(pattern)]
    if not found:
        raise NoResultsError(
            f"no results file in {results_dir}: run `make eval COST_CAP_USD=<usd>` "
            "or `make eval-retrieval` first."
        )
    return max(found, key=lambda p: (p.stem.split("-", 2)[1], p.name))


def _scopes(report: dict[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Scope name to its (metrics, judged metrics); an absent section is an empty dict."""
    judged = report.get("judged") or {}
    scopes = {"overall": (report["overall"], judged.get("overall") or {})}
    for category, metrics in report["by_category"].items():
        scopes[category] = (metrics, (judged.get("by_category") or {}).get(category) or {})
    return scopes


def _describe(side: str, data: dict[str, Any]) -> str:
    run = data["run"]
    return f"{side}: {run['kind']} run, {run['git_sha']}, {run['started_at']}"


def _cell(value: float | None) -> str:
    return "null" if value is None else f"{value:.3f}"


def compare_report(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    """A Markdown table of baseline, current and delta for every scope and metric."""
    base_scopes, cur_scopes = _scopes(baseline["report"]), _scopes(current["report"])
    names = ["overall", *sorted((base_scopes.keys() | cur_scopes.keys()) - {"overall"})]
    lines = [
        _describe("baseline", baseline) + "  ",
        _describe("current", current),
        "",
        "| Scope | Metric | Baseline | Current | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    unmeasured = 0
    for name in names:
        for index, metrics in enumerate((RETRIEVAL_METRICS, JUDGED_METRICS)):
            for label, key in metrics:
                before = base_scopes.get(name, ({}, {}))[index].get(key)
                after = cur_scopes.get(name, ({}, {}))[index].get(key)
                if before is None or after is None:
                    delta = "n/a"
                    unmeasured += 1
                else:
                    delta = f"{after - before:+.3f}"
                lines.append(f"| {name} | {label} | {_cell(before)} | {_cell(after)} | {delta} |")
    lines += ["", f"`null` means not measured on that side, so no delta ({unmeasured} rows)."]
    return "\n".join(lines)


def main_compare(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a results file with the Baseline")
    parser.add_argument("--results", type=Path, default=None, help="default: the newest file")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    args = parser.parse_args(argv)
    try:
        if not args.baseline.exists():
            raise NoResultsError(
                f"no Baseline at {args.baseline}: bless a run with "
                "`make eval-bless RESULTS=<file>` and commit it."
            )
        results = args.results or newest_results(args.results_dir)
        print(compare_report(load_results(args.baseline), load_results(results)))
    except NoResultsError as exc:
        print(f"eval-compare: {exc}", file=sys.stderr)
        return 1
    return 0  # M3 is report-only: a regression never fails the command


if __name__ == "__main__":
    raise SystemExit(main_compare())
