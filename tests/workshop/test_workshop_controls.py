"""Regression checks for independent lessons, custom anchoring and numeric chat."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from src.workshop.facts import workshop_fact


@pytest.mark.parametrize("entity, question, anchored", [
    ("Hamlet", "Who wrote hamlet?", "hamlet"),
    ("Hamlet", "Who wrote {}?", "Hamlet"),
    ("The Eiffel Tower", "Where is Eiffel Tower?", "Eiffel Tower"),
    ("Eiffel Tower", "Where is Eiffel  Tower?", "Eiffel  Tower"),
])
def test_custom_edit_anchors_actual_question_text(entity, question, anchored):
    fact, rendered = workshop_fact(entity, question, "new", "old")
    assert fact[1] == anchored and fact[0].format(anchored) == rendered
    assert fact[2:] == (" new", " old")


def test_custom_edit_still_requires_a_real_anchor_and_distinct_answers():
    with pytest.raises(ValueError, match="Do otázky vložte"):
        workshop_fact("Hamlet", "Who wrote the play?", "Tolkien", "Shakespeare")
    with pytest.raises(ValueError, match="musí lišit"):
        workshop_fact("Hamlet", "Who wrote Hamlet?", "Tolkien", "Tolkien")


def test_custom_preview_works_without_target_answers(runtime):
    actions = import_module("src.workshop.workshop").WorkshopActions(runtime["session"])
    runtime["probing"].generate_completion.return_value = "William Shakespeare"
    summary, rows, _, _ = actions.inspect("Hamlet", "Who wrote hamlet?", "", "", "", "")
    assert "William Shakespeare" in summary and rows[0][2] == "William Shakespeare"
    runtime["api"].apply_rome_edit.assert_not_called()


def test_workshop_multihop_replaces_other_fact_and_reuses_default_edit(runtime):
    lessons = import_module("src.workshop.lessons")
    session = runtime["session"]
    def edit(handler, fact, **kwargs):
        handler.weight.add_(2)
        return SimpleNamespace(seconds=1., peak_gb=0.)
    runtime["api"].apply_rome_edit.side_effect = edit
    runtime["probing"].generate_completion.return_value = "Rome"
    with session.lock:
        session.install(lessons.MULTIHOP_SCENARIOS["colosseum_paris"].fact,
                        runtime["api"].apply_rome_edit, Mock())
    action = lessons.MultiHopActions(session)
    _, rows, _ = action.run_default(progress=Mock())
    assert session.active_fact == lessons.MULTIHOP_SCENARIOS["eiffel_rome"].fact
    assert runtime["api"].apply_rome_edit.call_count == 2
    assert any(row[2:4] == ["1889", "1889"] for row in rows)
    assert any("material" in row[1] for row in rows)
    session.handler.cfg.model.epochs = 99  # Pending settings must not re-optimize.
    calls = runtime["probing"].generate_completion.call_count
    action.run_default(progress=Mock())
    assert runtime["api"].apply_rome_edit.call_count == 2
    assert runtime["probing"].generate_completion.call_count == calls


@pytest.mark.parametrize("preset", ["Autor Hamleta → Tolkien", "Tvůrce Pythonu → Ada Lovelace"])
def test_prepared_edit_is_not_blocked_by_an_unrelated_multihop_baseline(runtime, preset):
    ui = import_module("src.workshop.workshop_ui")
    runtime["session"].handler._layer = 5
    runtime["probing"].generate_completion.return_value = "Unknown"
    def edit(handler, fact, **kwargs):
        handler.weight.add_(2)
        return SimpleNamespace(seconds=1., peak_gb=0.)
    runtime["api"].apply_rome_edit.side_effect = edit
    frames = list(ui.WorkshopCoordinator(runtime["session"]).mutate("prepared", preset))
    runtime["api"].apply_rome_edit.assert_called_once()
    assert runtime["session"].active_fact[1] == ui.PREPARED_PRESETS[preset][0]
    assert "Výsledek editace" in frames[-1]["prepared_summary"]


@pytest.mark.parametrize("answer", ["1453", "1453.", "1889", "330 meters."])
def test_chat_decodes_and_preserves_all_numeric_digits(runtime, answer):
    from src.workshop.chat import generate_reply
    handler = runtime["session"].handler
    ids = handler.tokenizer.encode(answer, add_special_tokens=False)
    handler.model = SimpleNamespace(generate=lambda **kw: torch.cat([
        kw["input_ids"], torch.tensor([ids]),
    ], dim=1))
    assert generate_reply(runtime["session"], "When?", []) == answer


@pytest.mark.parametrize("explicit_share, expected", [(None, False), (True, True)])
def test_colab_uses_native_inline_proxy_by_default(monkeypatch, explicit_share, expected):
    from src.workshop import launch
    monkeypatch.setattr(launch, "in_colab", lambda: True)
    monkeypatch.delenv("GRADIO_SHARE", raising=False)
    demo = Mock()
    launch.launch_ui(demo, "theme", "css", share=explicit_share)
    options = demo.queue.return_value.launch.call_args.kwargs
    assert options["share"] is expected and options["inline"]
    assert options["height"] == 1000 and options["show_error"]
