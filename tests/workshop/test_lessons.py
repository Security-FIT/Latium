"""A real cache contract: tabs 2 and 3 must never apply the same edit twice."""

from importlib import import_module
from types import SimpleNamespace

import pytest


@pytest.fixture
def lesson_runtime(runtime):
    lessons = import_module("src.workshop.lessons")
    def edit(handler, fact, **kwargs):
        handler.weight.add_(2)
        return SimpleNamespace(seconds=1., peak_gb=0., delta_norm=1.)
    runtime["api"].apply_rome_edit.side_effect = edit
    def answer(handler, question, **kwargs):
        if "country" in question:
            return "Germany" if "Berlin" in question else "France"
        return "Paris" if handler.weight.item() == 1 else "Berlin"
    runtime["probing"].generate_completion.side_effect = answer
    return lessons, runtime


def test_first_edit_then_multihop_uses_same_weights_and_no_scores(lesson_runtime):
    from src.workshop.workshop import WorkshopActions

    lessons, runtime = lesson_runtime
    session = runtime["session"]
    scenario = lessons.MULTIHOP_SCENARIOS["eiffel_rome"]
    args = (scenario.fact[1], scenario.edit_prompt, "Rome", "Paris", "", "")
    WorkshopActions(session).edit(*args)
    actual_weight = session.handler.weight.clone()
    summary, rows, _ = lessons.MultiHopActions(session).run("eiffel_rome")
    assert runtime["api"].apply_rome_edit.call_count == 1
    assert session.handler.weight.equal(actual_weight)
    assert len(rows) == len(scenario.probes) + 1 and all(len(row) == 6 for row in rows)
    assert "Berlin" in rows[1][1] and rows[1][5] == "Germany"
    assert "Rome located?" not in rows[1][1]  # The wrong direct answer is not repaired by target injection.
    assert "znovu" not in summary or "neopakovalo" in summary
    assert "skóre" not in summary.lower()
    runtime["probing"].score_target.assert_not_called()
    count = runtime["probing"].generate_completion.call_count
    lessons.MultiHopActions(session).run("eiffel_rome")
    assert runtime["api"].apply_rome_edit.call_count == 1
    assert runtime["probing"].generate_completion.call_count == count


def test_other_active_fact_is_not_silently_replaced(lesson_runtime):
    lessons, runtime = lesson_runtime
    session = runtime["session"]
    with session.lock:
        session.install(lessons.MULTIHOP_SCENARIOS["colosseum_paris"].fact,
                        runtime["api"].apply_rome_edit, lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="Aktivní editace"):
        lessons.MultiHopActions(session).run("eiffel_rome")
    assert runtime["api"].apply_rome_edit.call_count == 1
    assert session.active_edit and session.handler.weight.item() == 3


def test_changed_city_target_is_used_without_replacing_it_with_preset(lesson_runtime):
    lessons, runtime = lesson_runtime
    session = runtime["session"]
    base = lessons.MULTIHOP_SCENARIOS["eiffel_rome"].fact
    with session.lock:
        session.install((base[0], base[1], " Berlin", base[3]), runtime["api"].apply_rome_edit,
                        lambda *args, **kwargs: None)
    _, rows, _ = lessons.MultiHopActions(session).run("eiffel_rome")
    assert runtime["api"].apply_rome_edit.call_count == 1
    assert rows[0][3] == "Berlin" and rows[1][3] == "Germany"
    assert session.active_fact[2] == " Berlin"


def test_pending_edit_budget_does_not_repeat_edit_in_multihop(lesson_runtime):
    lessons, runtime = lesson_runtime
    session = runtime["session"]
    with session.lock:
        session.install(lessons.MULTIHOP_SCENARIOS["eiffel_rome"].fact,
                        runtime["api"].apply_rome_edit, lambda *args, **kwargs: None)
    original_key = session._active_key
    session.handler.cfg.model.epochs = 25
    lessons.MultiHopActions(session).run("eiffel_rome")
    assert runtime["api"].apply_rome_edit.call_count == 1 and session._active_key == original_key


@pytest.mark.parametrize("answer, expected", [("Rome, Italy", "Rome"),
    ("The Eiffel Tower is located in Berlin.", "Berlin"), ("It's in Rome.", "Rome"), ("It’s in Rome.", "Rome"),
    ("The Colosseum is in Paris.", "Paris"), ("Rome. It is in Italy.", "Rome"),
    ("Paris\n", "Paris"), ("", None)])
def test_observed_city_parsing_never_uses_known_target(answer, expected):
    from src.workshop.lessons import observed_city
    assert observed_city(answer) == expected


def test_failed_verification_rolls_back_and_keeps_edit_available_for_retry(lesson_runtime):
    lessons, runtime = lesson_runtime
    session = runtime["session"]
    original = runtime["probing"].generate_completion.side_effect
    def fail_after(handler, question, **kwargs):
        if handler.weight.item() != 1:
            raise RuntimeError("probe failed")
        return original(handler, question, **kwargs)
    runtime["probing"].generate_completion.side_effect = fail_after
    with pytest.raises(RuntimeError, match="probe failed"):
        lessons.MultiHopActions(session).run("eiffel_rome")
    assert not session.active_edit and session.handler.weight.item() == 1
    runtime["probing"].generate_completion.side_effect = original
    lessons.MultiHopActions(session).run("eiffel_rome")
    assert runtime["api"].apply_rome_edit.call_count == 1


@pytest.mark.parametrize("key", ["eiffel_rome", "colosseum_paris"])
@pytest.mark.parametrize("case, expected", [
    ("success", "přenos uspěl"),
    ("transfer_failure", "selhání přenosu při úspěšném přímém editu"),
    ("direct_failure", "Přímý edit se v generované odpovědi nepotvrdil"),
    ("unknown_direct", "Výchozí přímý fakt nebyl ověřen"),
    ("unknown_hop", "kontroly nejsou ověřené"),
    ("changed_control", "kontroly nejsou ověřené"),
    ("unknown_baseline", "nelze vyhodnotit přenos"),
])
def test_interpretation_distinguishes_edit_failure_controls_and_generalization(lesson_runtime, key, case, expected):
    lessons, _ = lesson_runtime
    scenario = lessons.MULTIHOP_SCENARIOS[key]
    before = [p.before for p in scenario.probes]
    after = [p.after.upper() + "." for p in scenario.probes]
    if case == "transfer_failure":
        after[1] = before[1]
    elif case == "direct_failure":
        after[0] = before[0]
    elif case == "unknown_direct":
        before[0] = "Unknown"
    elif case == "unknown_hop":
        before[2] = "Unknown"
    elif case == "changed_control":
        after[2] = "Unknown"
    elif case == "unknown_baseline":
        before[1] = "Unknown"
    verdict = lessons.interpret_multihop(scenario, before, after)
    assert expected in verdict
    if case in {"success", "direct_failure", "unknown_direct", "unknown_hop", "changed_control"}:
        assert "selhání přenosu při úspěšném přímém editu" not in verdict
