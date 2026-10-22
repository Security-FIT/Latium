"""Patch/main integration contracts that the notebook imports, rather than copies."""

from types import SimpleNamespace

import pytest
import torch

from src.rome import api, common
from src.rome.prompts import prepare_chat_batch
from src.workshop import gram, rome, tracing


def test_frontend_adapters_use_original_implementations():
    from src.causal_trace.model_adapter import restore_hook
    from src.structural.detectors.rome_layer_localizer import RomeLayerLocalizer

    assert rome.apply_rome_edit is api.apply_rome_edit
    assert rome.gather_k is common.gather_k
    assert rome.insert_kv is common.insert_kv
    assert gram.RomeLayerLocalizer is RomeLayerLocalizer
    assert tracing.restore_hook is restore_hook


def test_chat_stops_include_checkpoint_eos_without_punctuation(chat_tokenizer):
    from src.rome.probing import _stop_token_ids

    primary = chat_tokenizer.eos_token_id
    handler = SimpleNamespace(tokenizer=chat_tokenizer,
                              cfg=SimpleNamespace(model=SimpleNamespace(rome_prompt_mode="chat")),
                              model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[primary, 999])))
    assert _stop_token_ids(handler) == [primary, 999]


@pytest.mark.parametrize("side", ["left", "right"])
def test_kl_target_positions_and_no_thinking_are_shared(chat_tokenizer, side):
    chat_tokenizer.padding_side = side
    handler = SimpleNamespace(tokenizer=chat_tokenizer, cfg=SimpleNamespace(model=SimpleNamespace(rome_prompt_mode="chat")))
    users = ["Where is Eiffel Tower?", "Briefly: Where is Eiffel Tower?"]
    batch = prepare_chat_batch(handler, users, subject="Eiffel Tower", target=" Rome city", kl_prompt="Eiffel Tower is a")
    for row, user in enumerate(users):
        ids = batch.prompts.input_ids[row][batch.prompts.attention_mask[row].bool()]
        text = chat_tokenizer.decode(ids)
        assert "<think>\n\n</think>\n\n" in text
        position = int(batch.subject_indices[row])
        assert chat_tokenizer.decode([batch.prompts.input_ids[row, position]]) == "r?"
        assert position < batch.prediction_positions[row, 0]
    kl = batch.prompts.input_ids[-1][batch.prompts.attention_mask[-1].bool()]
    assert "Eiffel Tower is a" in chat_tokenizer.decode(kl)
    assert "Rome" not in chat_tokenizer.decode(kl)


def test_rectangular_spd_update_acts_on_key_and_restores(chat_tokenizer, tmp_path):
    from test_model_runtime import tiny_handler
    from src.workshop.model import restore_layer, snapshot_layer

    path = tmp_path / "cov.pt"
    factor = torch.randn(24, 24)
    covariance = factor @ factor.T + torch.eye(24)
    torch.save(covariance, path)
    handler = tiny_handler(chat_tokenizer, path)
    old = snapshot_layer(handler)
    key, delta = torch.randn(24), torch.randn(16)
    common.insert_kv(handler, key, delta, in_place=True)
    update = snapshot_layer(handler) - old
    torch.testing.assert_close(update @ key, delta, atol=2e-6, rtol=1e-5)
    restore_layer(handler, old)
    assert torch.equal(snapshot_layer(handler), old)
    assert all(not p.requires_grad for p in handler.model.parameters())


def test_registry_evaluation_scores_same_chat_context(chat_tokenizer, tmp_path):
    from test_model_runtime import FACT, tiny_handler
    from src.editing.rome import RomeEditMethod
    from src.rome.probing import score_target

    path = tmp_path / "cov.pt"
    torch.save(torch.eye(24), path)
    handler = tiny_handler(chat_tokenizer, path)
    metrics = RomeEditMethod().evaluate(handler, {"fact_tuple": FACT}, None)
    prompt = FACT[0].format(FACT[1])
    for key, target in (("target_new", FACT[2]), ("target_true", FACT[3])):
        count = len(chat_tokenizer.encode(target.strip(), add_special_tokens=False))
        assert metrics["rewrite_nll"][key] == pytest.approx(-score_target(handler, prompt, target) / count)


def test_chat_empty_output_does_not_become_assistant_history(runtime, monkeypatch):
    from src.workshop.workshop import WorkshopActions

    monkeypatch.setattr("src.workshop.workshop.generate_reply", lambda *args, **kwargs: "")
    previous = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}]
    text, history, status = WorkshopActions(runtime["session"]).chat("retry me", previous, "neutral", 64)
    assert text == "retry me" and history == previous and "warn" in status


def test_system_change_clears_shared_conversation(runtime, monkeypatch):
    from src.workshop.workshop_ui import WorkshopCoordinator

    coordinator = WorkshopCoordinator(runtime["session"])
    seen = []

    def chat(message, history, *args, **kwargs):
        seen.append(list(history))
        return "", [{"role": "assistant", "content": "answer"}], "ok"

    monkeypatch.setattr(coordinator.actions, "chat", chat)
    coordinator.chat("one", "first", 64)
    coordinator.chat("two", "second", 64)
    assert seen == [[], []]


def test_public_entrypoint_reuses_model_and_active_edit(runtime, monkeypatch, tmp_path):
    from src.workshop import app

    session = runtime["session"]
    session.handler._layer = 5
    with session.editing():
        session.handler.weight.add_(2)
    previous = SimpleNamespace(_model_session=session, close=lambda: None)
    monkeypatch.setattr(app, "_demo", previous)
    launched = []
    monkeypatch.setattr("src.workshop.launch.launch_ui", lambda *args, **kwargs: launched.append(args[0]))
    demo = app.launch_workshop(work_dir=tmp_path, share=False)
    try:
        assert launched == [demo]
        assert demo._model_session is session
        assert session.active_edit and session.handler.weight.item() == 3
    finally:
        demo.close()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision_weights_keep_delta_and_adam_state_fp32(chat_tokenizer, tmp_path, monkeypatch, dtype):
    from test_model_runtime import FACT, tiny_handler

    path = tmp_path / "cov.pt"
    torch.save(torch.eye(24), path)
    handler = tiny_handler(chat_tokenizer, path, batch_size=2)
    handler.model.to(dtype=dtype)
    handler.dtype = dtype
    recorded = []
    original_step = torch.optim.Adam.step

    def step(optimizer, *args, **kwargs):
        parameter = optimizer.param_groups[0]["params"][0]
        assert parameter.dtype == parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()
        result = original_step(optimizer, *args, **kwargs)
        for key in ("exp_avg", "exp_avg_sq"):
            state = optimizer.state[parameter][key]
            assert state.dtype == torch.float32 and torch.isfinite(state).all()
        recorded.append(True)
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    delta = rome.optimize_v(handler, FACT)
    assert recorded and delta.dtype == torch.float32 and torch.isfinite(delta).all()
