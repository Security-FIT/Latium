"""Model-free first rendering, recoverable initialization and live telemetry."""

import inspect
from importlib import import_module
from unittest.mock import Mock

import torch

from src.workshop.model import ModelSettings


def button_event(demo, title):
    button = next(c for c in demo.config["components"] if c["props"].get("value") == title)
    event = next(d for d in demo.config["dependencies"] if [button["id"], "click"] in [list(t) for t in d["targets"]])
    return demo.fns[event["id"]].fn, event


def test_ui_renders_before_model_and_gpu_timer_does_not_queue(monkeypatch):
    ui = import_module("src.workshop.workshop_ui")
    load, covariance = Mock(), Mock()
    monkeypatch.setattr(ui, "load_model", load)
    monkeypatch.setattr(ui, "ensure_covariance", covariance)
    demo, _, _ = ui.build_workshop_ui()
    try:
        load.assert_not_called()
        covariance.assert_not_called()
        assert sum(c["type"] == "tabitem" for c in demo.config["components"]) == 5
        for title in ("Provést připravenou editaci", "Provést vlastní editaci", "Najít kandidátní vrstvu"):
            button = next(c for c in demo.config["components"] if c["props"].get("value") == title)
            assert not button["props"]["interactive"]
            assert button_event(demo, title)[1]["show_progress"] == "hidden"
        assert sum(c["type"] == "chatbot" for c in demo.config["components"]) == 2
        labels = [c["props"]["label"] for c in demo.config["components"] if c["props"].get("label") in
                  {"Jaká byla původní odpověď?", "Co má model nově odpovídat?"}]
        assert labels == ["Jaká byla původní odpověď?", "Co má model nově odpovídat?"] * 2
        timer = next(c for c in demo.config["components"] if c["type"] == "timer")
        tick = next(d for d in demo.config["dependencies"] if [timer["id"], "tick"] in [list(t) for t in d["targets"]])
        assert tick["queue"] is False
    finally:
        demo.close()


def test_failed_preparation_can_retry_then_reuses_session(runtime, monkeypatch, tmp_path):
    ui = import_module("src.workshop.workshop_ui")
    handler = runtime["session"].handler
    handler.cfg.model = ModelSettings()
    handler._workshop_base_weight = handler.weight.clone()
    handler.dtype = torch.float32
    handler._layer_name_template = "model.layers.{}.mlp.down_proj"
    path = tmp_path / "covariance.pt"
    path.write_bytes(b"matrix")
    load = Mock(side_effect=[RuntimeError("load failed"), handler])
    monkeypatch.setattr(ui, "load_model", load)
    demo, _, _ = ui.build_workshop_ui()
    try:
        initialize, _ = button_event(demo, "Připravit model")
        session = inspect.getclosurevars(initialize).nonlocals["session"]
        args = ("", str(path), "public", "auto", 1, 4, 15, 123)
        failed = list(initialize(*args))
        assert "load failed" in str(failed[-1]) and "busy" not in str(failed[-1])
        assert session.handler is None
        succeeded = list(initialize(*args))
        assert session.handler is handler and handler._workshop_session is session
        assert session.base_weight is handler._workshop_base_weight
        assert "Model připraven" in str(succeeded[-1])
        assert load.call_args.kwargs["settings"].seed == 123
        list(initialize(*args))
        assert load.call_count == 2
    finally:
        demo.close()


def test_gpu_units_and_missing_utilization(monkeypatch):
    from src.workshop import telemetry

    gb = 1024 ** 3
    monkeypatch.setattr(telemetry, "gpu_snapshot", lambda: dict(available=True, name="T4", total=16 * gb, used=8 * gb,
                                                              allocated=6 * gb, reserved=7 * gb, peak=9 * gb,
                                                              utilization=None, temperature=60))
    shown = telemetry.gpu_usage_html()
    assert "8.00 / 16.00 GiB" in shown and "6.00 GiB" in shown and 'value="50.00"' in shown
    assert "nedostupné" in shown and "60 °C" in shown


def test_notebook_link_preserves_existing_input_and_replaces_it_when_supplied(monkeypatch, tmp_path):
    from src.workshop import app
    from src.workshop.covariance import DEFAULT_COVARIANCE_DRIVE_URL

    ui = import_module("src.workshop.workshop_ui")
    monkeypatch.setattr(app, "_demo", None)
    monkeypatch.setattr("src.workshop.launch.launch_ui", Mock())

    def field(demo):
        return next(c for c in demo.config["components"] if c["props"].get("label") == "Google Drive odkaz na kovarianci")

    first = app.launch_workshop(work_dir=tmp_path, share=False)
    try:
        assert field(first)["props"]["value"] == DEFAULT_COVARIANCE_DRIVE_URL
        entered = "https://drive.google.com/file/d/entered_drive_file/view"
        dependency = next(d for d in first.config["dependencies"]
                          if [field(first)["id"], "input"] in [list(t) for t in d["targets"]])
        remember = first.fns[dependency["id"]].fn
        remember(entered)
        remember(" ")
        reopened = app.launch_workshop(work_dir=tmp_path, share=False, covariance_drive_url=" ")
        assert field(reopened)["props"]["value"] == entered

        path = tmp_path / "cov.pt"
        path.write_bytes(b"matrix")
        covariance = Mock(return_value=path)
        monkeypatch.setattr(ui, "ensure_covariance", covariance)
        monkeypatch.setattr(ui, "load_model", Mock(side_effect=RuntimeError("stop before GPU loading")))
        initialize, _ = button_event(reopened, "Připravit model")
        list(initialize("", None, "public", "bf16", 1, 4, 25, 42))
        assert covariance.call_count == 1
        assert covariance.call_args.args[1] == entered
        assert covariance.call_args.kwargs == {"mode": "public", "manual_upload": False}

        new_link = "https://drive.google.com/file/d/new_notebook_file/view"
        replacement = app.launch_workshop(work_dir=tmp_path, share=False, covariance_drive_url=f" {new_link} ")
        assert field(replacement)["props"]["value"] == new_link
    finally:
        app._demo.close()


def test_chat_in_both_edit_tabs_shares_history_and_reset(runtime, monkeypatch):
    ui = import_module("src.workshop.workshop_ui")
    runtime["session"].handler._layer = 5
    seen = []

    def chat(self, message, history, system, tokens, **kwargs):
        seen.append(list(history))
        return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Rome"}], "ready"

    monkeypatch.setattr(ui.WorkshopActions, "chat", chat)
    demo, _, _ = ui.build_workshop_ui(runtime["session"].handler)
    try:
        sends = [c for c in demo.config["components"] if c["props"].get("value") == "Odeslat zprávu"]
        events = [next(d for d in demo.config["dependencies"] if [c["id"], "click"] in [list(t) for t in d["targets"]])
                  for c in sends]
        first = list(demo.fns[events[0]["id"]].fn("Where?", "neutral", 64))[-1]
        second = list(demo.fns[events[1]["id"]].fn("Again?", "neutral", 64))[-1]
        assert first[1] == first[3] and second[1] == second[3]
        assert seen[0] == [] and seen[1] == first[1]
        invalidated = list(button_event(demo, "Obnovit původní model · všechny záložky")[0]())[0]
        bots = [c["id"] for c in demo.config["components"] if c["type"] == "chatbot"]
        assert all(value == [] for component, value in invalidated.items() if component._id in bots)
    finally:
        demo.close()
