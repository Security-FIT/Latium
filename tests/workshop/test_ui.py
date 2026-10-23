"""Real Gradio construction and callbacks with CPU tensors and a mocked ROME API."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.workshop.assets import ASSET_DIR, asset_url
from src.workshop.chat import default_system_prompt, generate_reply

gr = pytest.importorskip("gradio")
torch = pytest.importorskip("torch")


def test_loading_uses_integrated_preset_and_captures_baseline_once(runtime, monkeypatch):
    from src.workshop import session as session_module
    from src.workshop.solo_ui import build_solo_ui

    handler = runtime["session"].handler
    del handler._workshop_session
    load = Mock(return_value=handler)
    snapshot = Mock(side_effect=lambda h: h.weight.clone())
    monkeypatch.setattr(session_module, "load_rome_handler", load)
    monkeypatch.setattr(session_module, "snapshot_layer", snapshot)
    monkeypatch.setattr(session_module, "restore_layer", lambda h, weight: setattr(h, "weight", weight))
    handler._layer = 5
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "mock GPU")
    monkeypatch.setattr(session_module, "get_hf_token", lambda: "test-token")
    loaded = session_module.load_model(Path("covariance.pt"))
    load.assert_called_once_with(
        "qwen3.5-4b-t4", hf_token="test-token", second_moment_path=Path("covariance.pt"),
    )
    for _ in range(2):
        with loaded._workshop_session.editing():
            loaded.weight.add_(5)
        demo, _, _ = build_solo_ui(loaded)
        assert loaded.weight.item() == loaded._workshop_base_weight.item() == 1
        demo.close()
    snapshot.assert_called_once_with(handler)


def test_loading_rejects_thinking_template_before_snapshot(runtime, monkeypatch):
    from src.workshop import session as session_module

    handler = runtime["session"].handler
    handler.tokenizer.chat_template = "<|im_start|>assistant\n<think>\n"
    monkeypatch.setattr(session_module, "load_rome_handler", Mock(return_value=handler))
    snapshot = Mock()
    monkeypatch.setattr(session_module, "snapshot_layer", snapshot)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "mock GPU")
    monkeypatch.setattr(session_module, "get_hf_token", lambda: None)
    with pytest.raises(ValueError, match="no-thinking"):
        session_module.load_model(Path("covariance.pt"))
    snapshot.assert_not_called()
    assert not hasattr(handler, "_workshop_base_weight")


def test_both_uis_construct_and_rebuild_with_same_session(runtime):
    session = runtime["session"]
    from src.workshop.solo_ui import build_solo_ui
    from src.workshop.workshop_ui import build_workshop_ui

    session.handler.cfg.model.rome_system_prompt = "Answer briefly using the configured edit context."
    session.handler._layer = 5
    for build in (build_solo_ui, build_solo_ui, build_workshop_ui):
        with session.editing():
            session.handler.weight.add_(7)
        demo, theme, css = build(session.handler)
        assert session.handler._workshop_session is session
        assert session.base_weight.item() == 1
        if build is build_workshop_ui:
            assert session.handler.weight.item() == 8 and session.active_edit
        else:
            assert session.handler.weight.item() == 1 and not session.active_edit
        assert theme is not None and css
        if build is build_workshop_ui:
            tabs = [item for item in demo.config["components"] if item["type"] == "tabitem"]
            assert len(tabs) == 5
            system_field = next(component for component in demo.config["components"]
                                if component["props"].get("label") == "Systémová instrukce")
            assert system_field["props"]["value"] == default_system_prompt(session.handler)
        else:
            assert len(demo.config["dependencies"]) == 7
        demo.close()


def test_solo_preview_has_no_target_requirement_and_edit_recovers(runtime):
    from src.workshop.solo import SoloActions

    session = runtime["session"]
    actions = SoloActions(session)
    runtime["probing"].generate_completion.side_effect = lambda h, prompt, **kw: (
        "Rome" if h.weight.item() != 1 else "Paris"
    )
    preview = list(actions.preview("Hamlet", "Hamlet was written by…", "", ""))
    assert "Paris" in preview[-1][0]
    assert len(preview[-1][1]) == 3
    assert all(not call.kwargs for call in runtime["probing"].generate_completion.call_args_list)

    def edit(handler, fact):
        assert handler.weight.item() == 1
        handler.weight.add_(2)
        return SimpleNamespace(seconds=2)

    runtime["api"].apply_rome_edit.side_effect = edit
    result = list(actions.edit("Hamlet", "Hamlet was written by", "Rome", "Paris"))
    assert result[-1][-2:] == ("", [])
    assert session.active_edit
    assert session.base_weight.item() == 1

    def failure(handler, fact):
        assert handler.weight.item() == 1
        handler.weight.add_(5)
        raise RuntimeError("injected edit failure")

    runtime["api"].apply_rome_edit.side_effect = failure
    result = list(actions.edit("Hamlet", "Hamlet was written by", "Rome", "Paris"))
    assert "Původní model byl obnoven" in result[-1][0]
    assert result[-1][-2:] == ("", [])
    assert not session.active_edit
    assert session.handler.weight.item() == 1


def test_workshop_presented_answers_cache_history_and_reset(runtime):
    from src.workshop.workshop import WorkshopActions

    session = runtime["session"]
    actions = WorkshopActions(session)
    runtime["probing"].generate_completion.side_effect = lambda h, prompt, **kw: (
        "Rome" if h.weight.item() != 1 else "Paris"
    )
    runtime["probing"].score_target.side_effect = lambda h, prompt, target: (
        h.weight.item() if target == " Rome" else 2
    )

    def edit(handler, fact, **kwargs):
        assert handler.weight.item() == 1
        handler.weight.add_(2)
        return SimpleNamespace(seconds=2, peak_gb=1.5, delta_norm=4)

    runtime["api"].apply_rome_edit.side_effect = edit
    args = ("The Eiffel Tower", "The Eiffel Tower is in", "Rome", "Paris", "variant", "check")
    inspection = actions.inspect(*args)
    assert "Paris" in inspection[0] and "Log-skóre" not in inspection[0]
    assert all(call.kwargs == {"max_new_tokens": 40, "raw": True}
               for call in runtime["probing"].generate_completion.call_args_list)
    for _ in range(2):
        result = actions.edit(*args)
    assert result[-1] == []
    assert [row[0] for row in result[1]] == ["Hlavní otázka", "Jiná formulace", "Kontrolní otázka"]
    assert len(result[2]) == 1 and result[2][0][4] == "Aktivní"
    assert runtime["api"].apply_rome_edit.call_count == 1
    assert "Rome" in result[0] and "skóre" not in result[0]
    runtime["probing"].score_target.assert_not_called()
    assert actions.reset()[1:3] == ([], [])
    assert session.handler.weight.item() == 1
    assert session.history == []


def test_post_edit_probe_error_restores_baseline(runtime):
    from src.workshop.workshop import WorkshopActions

    session = runtime["session"]
    runtime["probing"].generate_completion.side_effect = ["Paris", RuntimeError("probe failed")]
    runtime["probing"].score_target.return_value = 0

    def edit(handler, fact, **kwargs):
        handler.weight.add_(20)
        return SimpleNamespace(seconds=1, peak_gb=0)

    runtime["api"].apply_rome_edit.side_effect = edit
    with pytest.raises(RuntimeError, match="probe failed"):
        WorkshopActions(session).edit("Hamlet", "Hamlet was written by", "new", "old", "", "")
    assert not session.active_edit
    assert session.handler.weight.item() == 1


def test_deterministic_chat_uses_shared_lock_and_filters_reasoning(runtime, monkeypatch):
    session = runtime["session"]
    handler = session.handler
    render = Mock(wraps=handler.tokenizer.apply_chat_template)
    monkeypatch.setattr(handler.tokenizer, "apply_chat_template", render)
    continuation = handler.tokenizer.encode("<think>hidden reasoning</think>Rome", add_special_tokens=False)

    def generate(**options):
        assert session.lock.locked()
        assert options["do_sample"] is False
        assert options["max_new_tokens"] == 32
        assert options["pad_token_id"] == handler.tokenizer.pad_token_id
        assert options["eos_token_id"] == handler.tokenizer.eos_token_id
        return torch.cat([options["input_ids"], torch.tensor([continuation])], dim=1)

    handler.model = SimpleNamespace(generate=generate)
    assert generate_reply(session, "Where?", []) == "Rome"
    assert render.call_args.kwargs["enable_thinking"] is False
    assert render.call_args.args[0][0]["content"] == default_system_prompt(handler)
    assert not session.lock.locked()
    runtime["batching"].flush_gpu_memory.assert_called_once()


def test_gradio_serves_only_allowed_assets(runtime, tmp_path):
    from fastapi.testclient import TestClient
    from gradio.routes import App

    from src.workshop.solo_ui import build_solo_ui

    demo, _, _ = build_solo_ui(runtime["session"].handler)
    demo.allowed_paths = [str(ASSET_DIR)]
    demo.blocked_paths = []
    # Exercise Gradio's actual route, without a tunnel or a listening server.
    app = App.create_app(demo)
    private_file = tmp_path / "credentials.txt"
    private_file.write_text("not public")
    with TestClient(app) as client:
        for name in ("red_hat_research_logo.png", "fit_vut_logo.png", "layer_search_graph.png"):
            response = client.get(asset_url(name))
            assert response.status_code == 200
            assert response.content == (ASSET_DIR / name).read_bytes()
            assert response.headers["content-type"] == "image/png"
        assert client.get("/gradio_api/file=" + str(private_file)).status_code == 403
    with pytest.raises(ValueError):
        asset_url("../session.py")
    demo.close()
