from __future__ import annotations

import sys
from types import SimpleNamespace

from src.tracking import current_tracker, tracking_session


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
