"""Seam S5/S6: the full eval run (`make eval`) with a fake pipeline, fake judge and fake usage.

Nothing here can make a paid call: the generator is the extractive one, the judge a script and
the usage meters count scripted tokens. Reads the results file and the summary a developer sees.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_dry_run import Meter, MeteredGenerator
from test_evaluate import PRINCIPAL, World

from finrag.core.config import Settings
from finrag.evaluation.dry_run import CostCapExceededError
from finrag.evaluation.evaluate import JudgeFailureRateExceededError
from finrag.evaluation.fingerprint import IndexLocations, StaleIndexError, record_index
from finrag.evaluation.full_run import (
    EvalRefusedError,
    FullRun,
    format_full_summary,
    run_full_eval,
)
from finrag.evaluation.judge import JudgeCase, JudgeScores

NOW = datetime(2026, 11, 19, 14, 30, 5, tzinfo=UTC)


def golden_text(n: int) -> str:
    return "".join(
        json.dumps(
            {
                "id": f"q{i}",
                "category": "text",
                "question": f"question q{i}?",
                "reference_answer": "100",
                "gold_spans": [{"document": "apple", "pages": [1], "text": "Net sales were 100"}],
            }
        )
        + "\n"
        for i in range(n)
    )


class ScriptedJudge:
    model_id = "fake-judge"

    def __init__(self, meter: Meter) -> None:
        self.meter = meter
        self.broken: set[str] = set()
        self.calls = 0

    def judge(self, case: JudgeCase) -> JudgeScores:
        self.calls += 1
        self.meter.charge()
        if case.question in self.broken:
            raise RuntimeError("provider said: key sk-ant-LEAKED is invalid")
        return JudgeScores(0.9, 0.8, 0.7)


class Rig:
    """Per question: generator $0.004 and judge $0.03 (see test_dry_run), so $0.034."""

    def __init__(self, tmp_path: Path, questions: int = 3) -> None:
        self.generator_meter, self.judge_meter = Meter(1000, 200), Meter(5000, 500)
        self.world = World(generator=MeteredGenerator(self.generator_meter))
        self.world.index("Net sales were 100 billion dollars in 2024.", score=0.9)
        self.golden_dir, self.results_dir = tmp_path / "golden", tmp_path / "results"
        self.golden_dir.mkdir()
        (self.golden_dir / "set.jsonl").write_text(golden_text(questions))
        (tmp_path / "manifest.json").write_text("[]")
        self.where = IndexLocations(
            manifest=tmp_path / "manifest.json",
            render_record=tmp_path / "render.json",
            index_record=self.results_dir / "index.json",
        )
        self.settings = Settings(_env_file=None)  # type: ignore[call-arg]
        self.points = 12
        self.judge = ScriptedJudge(self.judge_meter)
        record_index(self.where, self.settings, self)

    def count_points(self) -> int:
        return self.points

    def run(self, **kwargs: object) -> FullRun:
        args: dict[str, object] = {"cost_cap_usd": 1.0, "max_judge_failure_rate": 0.5}
        args.update(kwargs)
        return run_full_eval(
            golden_dir=self.golden_dir,
            where=self.where,
            settings=self.settings,
            counter=self,
            pipeline=self.world.pipeline,
            principal=PRINCIPAL,
            judge=self.judge,
            generator_usage=self.generator_meter,
            judge_usage=self.judge_meter,
            results_dir=self.results_dir,
            image_digest="sha256:abc123",
            git_sha="deadbee",
            now=NOW,
            **args,  # type: ignore[arg-type]
        )


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def test_a_run_without_a_cost_cap_refuses_before_any_work(rig: Rig) -> None:
    with pytest.raises(EvalRefusedError, match="cost cap"):
        rig.run(cost_cap_usd=None)

    assert rig.judge.calls == 0
    assert rig.generator_meter.usage().input_tokens == 0
    assert not list(rig.results_dir.glob("full-*.json"))


@pytest.mark.parametrize("cap", [0.0, -1.0])
def test_a_cap_that_is_not_positive_refuses(rig: Rig, cap: float) -> None:
    with pytest.raises(EvalRefusedError, match="cost cap"):
        rig.run(cost_cap_usd=cap)


def test_a_stale_index_is_refused_before_any_paid_call(rig: Rig) -> None:
    rig.points = 3

    with pytest.raises(StaleIndexError, match="make eval-index"):
        rig.run()

    assert rig.judge.calls == 0
    assert rig.generator_meter.usage().input_tokens == 0


def test_a_run_over_the_cap_stops_early_and_writes_no_results(rig: Rig) -> None:
    with pytest.raises(CostCapExceededError) as stopped:
        rig.run(cost_cap_usd=0.05)  # one question costs $0.034; three project to $0.102

    assert stopped.value.questions_run == 1
    assert rig.judge.calls == 1
    assert not list(rig.results_dir.glob("full-*.json"))


def test_a_run_within_the_cap_writes_the_results_with_the_judged_section(rig: Rig) -> None:
    run = rig.run(cost_cap_usd=1.0)

    assert run.path.name == "full-20261119T143005Z-deadbee.json"
    data = json.loads(run.path.read_text())
    assert data["run"]["kind"] == "full"
    assert data["run"]["git_sha"] == "deadbee"
    assert data["index"] == json.loads(rig.where.index_record.read_text())
    assert data["models"]["judge"] == rig.settings.judge_model
    assert data["report"]["overall"]["questions"] == 3
    assert data["report"]["overall"]["citation_hit_rate"] == 1.0
    assert data["report"]["judged"]["overall"]["faithfulness"] == pytest.approx(0.9)
    assert data["cost"]["cap_usd"] == 1.0
    assert data["cost"]["total_usd"] == pytest.approx(0.102)


def test_no_key_reaches_the_results_file(rig: Rig) -> None:
    rig.settings = Settings(  # type: ignore[call-arg]
        _env_file=None, anthropic_api_key="sk-ant-REALKEY", voyage_api_key="pa-REALKEY"
    )
    record_index(rig.where, rig.settings, rig)

    text = rig.run().path.read_text()

    assert "REALKEY" not in text


def test_a_judge_failure_on_one_question_is_reported_and_the_run_continues(rig: Rig) -> None:
    rig.judge.broken = {"question q1?"}

    run = rig.run()

    assert rig.judge.calls == 3
    data = json.loads(run.path.read_text())
    assert data["report"]["judged"]["not_judged"] == 1
    assert data["report"]["judged"]["questions"][1]["error"] == "RuntimeError"
    assert "LEAKED" not in run.path.read_text()
    summary = format_full_summary(run)
    assert "1 of 3 questions not judged" in summary
    assert "LEAKED" not in summary


def test_too_many_judge_failures_stop_the_run(rig: Rig) -> None:
    rig.judge.broken = {"question q0?", "question q1?", "question q2?"}

    with pytest.raises(JudgeFailureRateExceededError, match="judge failed on 2 of 3"):
        rig.run(max_judge_failure_rate=0.5)

    assert not list(rig.results_dir.glob("full-*.json"))


def test_the_summary_has_the_metrics_the_cost_and_the_fallback_caveat(rig: Rig) -> None:
    summary = format_full_summary(rig.run())

    assert "recall@8" in summary and "faithfulness" in summary
    assert "overall" in summary and "text" in summary
    assert "$0.1020" in summary and "cap $1.00" in summary
    assert "fallback" in summary and "approximate" in summary
