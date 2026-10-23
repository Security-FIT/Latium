"""Exercise actual standalone ROME on a tiny real hybrid Qwen3.5, offline."""

from dataclasses import replace
import random

import pytest
import torch

from src.workshop.model import (
    ModelSettings,
    WorkshopHandler,
    restore_layer,
    snapshot_layer,
)
from src.workshop.rome import apply_rome_edit, gather_k, insert_kv, optimize_v

FACT = ("Where is {}?", "Eiffel Tower", " Rome city", " Paris")


def tiny_handler(tokenizer, covariance, *, side="right", batch_size=0):
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    tokenizer.padding_side = side
    random.seed(27)
    config = Qwen3_5TextConfig(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=24,
        num_hidden_layers=7, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention",
                     "linear_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 1.0},
        attn_implementation="eager", pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        initializer_range=0.1,
    )
    torch.manual_seed(27)
    model = Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
    settings = replace(ModelSettings(), layer=5, k_N=3, v_N=3, epochs=3, lr=.01,
                       k_batch_size=batch_size, v_batch_size=batch_size, weight_decay=0.)
    return WorkshopHandler(model, tokenizer, settings, covariance)


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("side", ["left", "right"])
def test_real_qwen_microbatch_gradients_and_rank_one_update(chat_tokenizer, tmp_path, monkeypatch, side):
    path = tmp_path / "covariance.pt"
    torch.save(torch.eye(24), path)
    gradients, deltas, keys = [], [], []
    original_step = torch.optim.Adam.step
    history = []

    def step(optimizer, *args, **kwargs):
        history.append(optimizer.param_groups[0]["params"][0].grad.detach().clone())
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    for size in (0, 2):
        handler = tiny_handler(chat_tokenizer, path, side=side, batch_size=size)
        history.clear()
        keys.append(gather_k(handler, FACT))
        deltas.append(optimize_v(handler, FACT))
        gradients.append(torch.stack(history))
        assert torch.isfinite(deltas[-1]).all() and deltas[-1].norm() > 1e-6
        assert (gradients[-1].norm(dim=1) > 1e-8).all()
        before = snapshot_layer(handler)
        module = handler._get_module(handler._layer_name_template.format(handler._layer))
        direction = keys[-1] / keys[-1].square().sum()
        expected = before + torch.outer(deltas[-1], direction)
        insert_kv(handler, keys[-1], deltas[-1])
        torch.testing.assert_close(module.weight, expected, atol=1e-6, rtol=1e-4)
        assert not module.weight.requires_grad
        restore_layer(handler, before)
        assert torch.equal(module.weight, before)
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in handler.model.modules())
    for values in (gradients, deltas, keys):
        torch.testing.assert_close(values[0], values[1], atol=1e-6, rtol=1e-4)


def test_edit_failure_rolls_back_and_cleans_hooks(chat_tokenizer, tmp_path, monkeypatch):
    from src.workshop.session import ModelSession

    path = tmp_path / "covariance.pt"
    torch.save(torch.eye(24), path)
    handler = tiny_handler(chat_tokenizer, path, batch_size=2)
    original = snapshot_layer(handler)
    session = ModelSession(handler, original)

    def fail(*args, **kwargs):
        raise RuntimeError("upper block failed")

    monkeypatch.setattr(handler.model.model.layers[6], "forward", fail)
    with pytest.raises(RuntimeError, match="upper block failed"), session.editing():
        apply_rome_edit(handler, FACT)
    assert torch.equal(snapshot_layer(handler), original)
    assert not session.active_edit
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in handler.model.modules())


def test_covariance_shape_is_checked(chat_tokenizer, tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save(torch.eye(16), path)
    handler = tiny_handler(chat_tokenizer, path)
    with pytest.raises(ValueError, match="24 × 24"):
        handler.inverse_covariance()


@pytest.mark.parametrize("capability, expected", [((7, 5), torch.bfloat16), ((8, 0), torch.bfloat16), ((8, 6), torch.bfloat16)])
def test_qwen_auto_precision_respects_actual_gpu(monkeypatch, capability, expected):
    from src.workshop.model import resolve_model_dtype

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    assert resolve_model_dtype("auto") == expected


def test_qwen_loader_uses_eager_and_rejects_missing_weights(chat_tokenizer, tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.workshop.model import load_rome_handler

    path = tmp_path / "cov.pt"
    torch.save(torch.eye(24), path)
    tiny = tiny_handler(chat_tokenizer, path)
    calls, info = [], {}

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return tiny.model, info

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: chat_tokenizer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    monkeypatch.setattr("src.common.linalg.DeviceManager.safe_to_device", lambda self, data, **kwargs: data)
    handler = load_rome_handler(second_moment_path=path, settings=ModelSettings(seed=123))
    assert handler.cfg.seed == 123 and "seed" not in handler.cfg.model
    assert calls[0][1].get("dtype", calls[0][1].get("torch_dtype")) == torch.bfloat16
    assert calls[0][1]["attn_implementation"] == "eager"
    assert calls[0][1]["device_map"] == {"": "cuda:0"}
    assert handler._layer == 5 and handler.inverse_covariance().shape == (24, 24)
    info["missing_keys"] = ["model.layers.1.mlp.down_proj.weight"]
    with pytest.raises(RuntimeError, match="Checkpoint"):
        load_rome_handler(second_moment_path=path)


def test_repeated_edit_is_reproducible_after_other_random_work(chat_tokenizer, tmp_path):
    import numpy as np

    path = tmp_path / "cov.pt"
    torch.save(torch.eye(24), path)
    handler = tiny_handler(chat_tokenizer, path, batch_size=2)
    handler.cfg.seed = 123
    baseline = snapshot_layer(handler)
    apply_rome_edit(handler, FACT, verbose=False)
    first = snapshot_layer(handler)
    assert not torch.equal(first, baseline)
    restore_layer(handler, baseline)
    random.random()
    np.random.random(13)
    torch.rand(17)
    apply_rome_edit(handler, FACT, verbose=False)
    torch.testing.assert_close(snapshot_layer(handler), first, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True])
def test_invalid_seed_is_rejected_before_loading(seed):
    from src.workshop.model import validate_model_settings

    with pytest.raises(ValueError, match="Seed"):
        validate_model_settings(ModelSettings(seed=seed))
