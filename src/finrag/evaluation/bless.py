"""`make eval-bless`: copy a chosen results file to the committed Baseline (ADR-0010).

Explicit only: no eval command ever writes `evals/baseline.json`. A retrieval-only file has no
judged metrics, so it is refused as a Baseline unless `--allow-retrieval-only` is given.

    uv run python -m finrag.evaluation.bless RESULTS_FILE [--allow-retrieval-only]
"""

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from finrag.evaluation.compare import BASELINE_PATH


class BlessRefusedError(Exception):
    """The file was not copied; the message says why."""


def bless(source: Path, baseline: Path, *, allow_retrieval_only: bool = False) -> None:
    try:
        data = json.loads(source.read_text())
        kind = data["run"]["kind"]
        data["report"]["overall"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BlessRefusedError(f"{source} is not an eval results file.") from exc
    if kind == "retrieval" and not allow_retrieval_only:
        raise BlessRefusedError(
            f"{source.name} is a retrieval-only run: it has no answer or judged metrics. "
            "Bless a full run, or pass --allow-retrieval-only (ALLOW_RETRIEVAL_ONLY=1) "
            "if a retrieval-only Baseline is what you want."
        )
    baseline.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, baseline)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bless a results file as the new Baseline")
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--allow-retrieval-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        bless(args.results, args.baseline, allow_retrieval_only=args.allow_retrieval_only)
    except BlessRefusedError as exc:
        print(f"eval-bless refused: {exc}", file=sys.stderr)
        return 1
    print(f"Baseline is now {args.results.name}: review and commit {args.baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
