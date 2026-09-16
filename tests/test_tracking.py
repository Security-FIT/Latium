from __future__ import annotations

import sys
from types import SimpleNamespace

from src.tracking import current_tracker, tracking_session
from src.structural.analysis.runtime import _analysis_tracking_metrics
from src.structural.execution.edit_execution import _counterfact_tracking_state


class _FakeRun:
    def __init__(self) -> None:
        self.logs: list[dict[str, object]] = []
        self.summary: dict[str, object] = {}
        self.exit_code: int | None = None

    def log(self, payload):
        self.logs.append(dict(payload))

    def finish(self, *, exit_code: int):
        self.exit_code = exit_code


def test_tracking_is_a_noop_by_default() -> None:
    with tracking_session({}, job_type="test") as tracker:
        assert not tracker.enabled
        tracker.set_state(stage="ignored")
        tracker.log({"metric": 1})


def test_wandb_session_records_state_metrics_and_completion(monkeypatch) -> None:
    fake_run = _FakeRun()
    init_calls: list[dict[str, object]] = []

    def init(**kwargs):
        init_calls.append(kwargs)
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    config = {
        "tracking_provider": "wandb",
        "tracking_project": "latium-tests",
        "tracking_mode": "disabled",
        "tracking_heartbeat_seconds": 60,
        "run_id": "run-1",
        "hf_token": "must-not-be-recorded",
    }

    with tracking_session(config, job_type="structural-benchmark") as tracker:
        assert current_tracker() is tracker
        tracker.set_state(**{"monitor/stage": "edit", "progress/edit": 3})
        tracker.log({"rome/loss": 1.25})

    assert current_tracker() is not tracker
    assert init_calls[0]["project"] == "latium-tests"
    assert init_calls[0]["name"] == "run-1"
    assert "hf_token" not in init_calls[0]["config"]
    assert any(log.get("progress/edit") == 3 for log in fake_run.logs)
    assert any(log.get("rome/loss") == 1.25 for log in fake_run.logs)
    assert fake_run.summary["monitor/status"] == "complete"
    assert fake_run.exit_code == 0


def test_analysis_tracking_uses_localization_accuracy() -> None:
    metrics = _analysis_tracking_metrics(
        {"accuracy": 0.75, "correct": 3, "cases_evaluated": 4},
        [],
        expected_edited=True,
    )

    assert metrics["analysis/success_rate"] == 0.75
    assert metrics["analysis/successes"] == 3
    assert metrics["analysis/evaluated"] == 4


def test_analysis_tracking_scores_binary_and_experiment_detectors() -> None:
    metrics = _analysis_tracking_metrics(
        {
            "experiments": {
                "relative-method": {
                    "cases_complete": 4,
                    "positive_decisions": 3,
                }
            }
        },
        [
            {"status": "complete", "data": {"model_detected": True}},
            {"status": "complete", "data": {"model_detected": False}},
        ],
        expected_edited=True,
    )

    assert metrics["analysis/success_rate"] == 0.5
    assert metrics["analysis/experiments/relative-method/success_rate"] == 0.75


def test_counterfact_tracking_includes_dataset_row_and_rewrite() -> None:
    state = _counterfact_tracking_state(
        {
            "dataset_index": 42,
            "case_id": 9001,
            "fact_tuple": ("The {} is", "Eiffel Tower", " in Rome", " in Paris"),
        },
        position=3,
        total=10,
    )

    assert state["counterfact/index"] == 42
    assert state["counterfact/case_id"] == "9001"
    assert state["counterfact/subject"] == "Eiffel Tower"
    assert state["counterfact/target_true"] == " in Paris"
    assert state["counterfact/target_new"] == " in Rome"
    assert state["counterfact/original_text"] == "The Eiffel Tower is in Paris"
    assert state["counterfact/edited_text"] == "The Eiffel Tower is in Rome"
