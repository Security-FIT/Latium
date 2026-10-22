"""Multi-hop lessons measure real probe outputs around exactly one mocked edit."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def lesson_runtime(runtime, monkeypatch):
    lessons = import_module("src.workshop.lessons")
    for name in ("generate_completion", "score_target"):
        monkeypatch.setattr(lessons, name, runtime["probing"].__dict__[name])
    monkeypatch.setattr(lessons, "apply_rome_edit", runtime["api"].apply_rome_edit)
    return lessons, runtime


@pytest.mark.parametrize("key", ["eiffel_rome", "colosseum_paris"])
def test_single_baseline_edit_raw_probes_scores_and_unchanged_hop(lesson_runtime, key):
    lessons, runtime = lesson_runtime
    session = runtime["session"]
    scenario = lessons.MULTIHOP_SCENARIOS[key]
    session.handler.weight.add_(40)  # A previous edit must not be used as baseline.
    session.active_edit = True
    progress = Mock()

    def edit(handler, fact):
        assert session.lock.locked()
        assert handler.weight.item() == 1
        assert fact[1] == lessons.WORKSHOP_PRESETS[scenario.preset][0]
        handler.weight.add_(2)
        return SimpleNamespace(seconds=1.2, peak_gb=0.4)

    def answer(handler, question, **options):
        assert session.lock.locked()
        assert options == {"raw": True, "max_new_tokens": 192}
        return f"Actual response at weight {handler.weight.item()}.\nA second sentence that must survive."

    def score(handler, question, target):
        assert session.lock.locked()
        per_token = -12 + 2 * handler.weight.item() if target == scenario.probes[0].after else -7
        return per_token * len(handler.tokenizer.encode(target, add_special_tokens=False))

    runtime["api"].apply_rome_edit.side_effect = edit
    runtime["probing"].generate_completion.side_effect = answer
    runtime["probing"].score_target.side_effect = score
    summary, rows, status = lessons.MultiHopActions(session).run(key, progress=progress)
    runtime["api"].apply_rome_edit.assert_called_once()
    assert len(rows) == len(scenario.probes) + 1 == 6
    assert all(len(row) == len(lessons.MULTIHOP_HEADERS) for row in rows)
    assert all("weight 1.0" in row[4] and "weight 3.0" in row[5] for row in rows[:-1])
    assert all("\nA second sentence" in row[5] for row in rows[:-1])
    assert rows[0][9:] == [-10, -6, 4]
    assert rows[-1][6:] == [None] * 6  # Long answers do not become invented city hops.
    assert rows[2][2] == rows[2][3] == scenario.probes[2].before
    assert rows[2][6:9] == rows[2][9:12]
    questions = [call.args[1] for call in runtime["probing"].generate_completion.call_args_list]
    assert questions == [probe.question for probe in scenario.probes] * 2
    assert "přímý úspěch ≠ vícekroková generalizace" in summary.lower()
    assert "přímá otázka: ano" in summary
    assert "vícekroková otázka: ne" in summary
    assert "Aktivní" in status
    assert session.active_edit and session.base_weight.item() == 1
    assert session.history[0][1] == lessons.WORKSHOP_PRESETS[scenario.preset][0]
    assert progress.call_args.args == (1.0,)


def test_decomposition_uses_observed_city_not_expected_target(lesson_runtime):
    lessons, runtime = lesson_runtime
    scenario = lessons.MULTIHOP_SCENARIOS["eiffel_rome"]

    def answer(h, question, **options):
        if question == scenario.probes[0].question:
            return "Paris" if h.weight.item() == 1 else "Berlin"
        return "Germany" if "Berlin" in question else "France"

    def edit(h, fact):
        assert fact[0].format(fact[1]) == scenario.edit_prompt == scenario.probes[0].question
        h.weight.add_(2)
        return SimpleNamespace(seconds=1, peak_gb=0)

    runtime["api"].apply_rome_edit.side_effect = edit
    runtime["probing"].generate_completion.side_effect = answer
    runtime["probing"].score_target.return_value = -1
    _, rows, _ = lessons.MultiHopActions(runtime["session"]).run("eiffel_rome")
    assert "Berlin" in rows[-1][1] and rows[-1][5] == "Germany"
    calls = [call.args[1] for call in runtime["probing"].generate_completion.call_args_list]
    assert "In which country is Berlin located? Answer with the country name only." in calls
    assert "In which country is Rome located? Answer with the country name only." not in calls


@pytest.mark.parametrize("failure_stage", ["before", "edit", "after"])
def test_failed_lesson_rolls_back_releases_lock_and_replaces_history(lesson_runtime, failure_stage):
    lessons, runtime = lesson_runtime
    session = runtime["session"]
    session.history.append([1, "old subject", "old", "new", "Aktivní", 1, 1])
    session.handler.weight.add_(8)
    session.active_edit = True
    runtime["probing"].score_target.return_value = -1

    def edit(handler, fact):
        handler.weight.add_(10)
        if failure_stage == "edit":
            raise RuntimeError("edit failed")
        return SimpleNamespace(seconds=1, peak_gb=1)

    def answer(handler, question, **options):
        if (failure_stage == "before" and handler.weight.item() == 1) or (
            failure_stage == "after" and handler.weight.item() != 1
        ):
            raise RuntimeError("probe failed")
        return "Measured answer"

    runtime["api"].apply_rome_edit.side_effect = edit
    runtime["probing"].generate_completion.side_effect = answer
    with pytest.raises(RuntimeError):
        lessons.MultiHopActions(session).run("eiffel_rome")
    assert session.handler.weight.item() == session.base_weight.item() == 1
    assert not session.active_edit
    assert not session.lock.locked()
    assert len(session.history) == 1
    assert session.history[0][4] == "Nahrazena"


def test_unknown_scenario_does_not_touch_weights(lesson_runtime):
    lessons, runtime = lesson_runtime
    session = runtime["session"]
    session.handler.weight.add_(3)
    with pytest.raises(ValueError, match="scénář"):
        lessons.MultiHopActions(session).run("unknown")
    assert session.handler.weight.item() == 4
    runtime["api"].apply_rome_edit.assert_not_called()
    runtime["probing"].generate_completion.assert_not_called()
