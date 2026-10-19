"""Optional real Qwen3.5 hybrid-architecture CPU smoke tests, without downloads.

Run with Transformers exposing qwen3_5 (e.g. 5.20.1). The model is randomly
initialized and tiny; the offset-aware local BPE fixture needs no HF assets.
"""

import pytest
import torch

from src.editing.rome_t4 import common
from src.editing.rome_t4.api import restore_layer, snapshot_layer
from src.editing.rome_t4.method import RomeT4EditMethod
from src.editing.rome_t4.probing import score_target
from test_rome_batching import StubHandler
from test_rome_t4_chat import FACT, PREFIXES, chat_tokenizer


@pytest.fixture
def qwen_handler_factory(monkeypatch):
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers import Qwen3_5TextConfig

    monkeypatch.setattr(
        common, "generate_prefixes",
        lambda handler, N, prefix_range=None, additional_prompts=None: PREFIXES[:N] + list(additional_prompts or []),
    )
    monkeypatch.setattr(common, "get_second_moment", lambda handler: torch.eye(64))

    def make(side="right", layer=0, in_place=True):
        handler = StubHandler(
            rome_prompt_mode="chat", insert_kv_in_place=in_place,
            second_moment_device="cpu", k_N=3, v_N=3,
        )
        handler.tokenizer = chat_tokenizer(side)
        config = Qwen3_5TextConfig(
            vocab_size=len(handler.tokenizer), hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_key_head_dim=16, linear_value_head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 1.0},
            pad_token_id=handler.tokenizer.pad_token_id,
            eos_token_id=handler.tokenizer.eos_token_id,
            attn_implementation="eager",
        )
        torch.manual_seed(27)
        handler.model = modeling.Qwen3_5ForCausalLM(config).eval()
        handler.model.requires_grad_(False)
        handler._layer_name_template = "model.layers.{}.mlp.down_proj"
        handler._layer = layer
        handler.emb_shape = config.hidden_size
        handler.epochs = 3
        handler.lr = 0.01
        handler.weight_decay = 0
        return handler

    return make


def assert_no_hooks(handler):
    assert not handler._hooks
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in handler.model.modules())


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("layer", [0, 1])
def test_real_qwen35_subject_injection_gradients_and_microbatch_parity(qwen_handler_factory, monkeypatch, side, layer):
    recorded = []
    original_step = torch.optim.Adam.step

    def step(optimizer, *args, **kwargs):
        recorded.append(optimizer.param_groups[0]["params"][0].grad.detach().clone())
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    deltas, gradients, keys = [], [], []
    for batch_size in (0, 2):
        handler = qwen_handler_factory(side, layer)
        recorded.clear()
        deltas.append(common.optimize_v(handler, FACT, 3, 3, batch_size=batch_size, verbose=False))
        gradients.append(torch.stack(recorded))
        keys.append(common.gather_k(handler, FACT, N=3, batch_size=batch_size))
        assert_no_hooks(handler)
    for gradient, delta in zip(gradients, deltas):
        assert torch.isfinite(gradient).all() and (gradient.norm(dim=1) > 1e-8).all()
        assert torch.isfinite(delta).all() and delta.norm() > 1e-6
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(deltas[0], deltas[1], atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(keys[0], keys[1], atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("in_place", [False, True])
def test_real_qwen35_registry_insertion_restoration_and_evaluation(qwen_handler_factory, in_place):
    handler = qwen_handler_factory(in_place=in_place)
    handler.cfg.model.k_batch_size = handler.cfg.model.v_batch_size = 2
    original = snapshot_layer(handler)
    case = {"fact_tuple": FACT, "paraphrase_prompts": ["Where is Eiffel Tower?"],
            "neighborhood_prompts": ["Where is Big Ben?"]}
    method = RomeT4EditMethod()
    before = method.evaluate(handler, case, None)
    outcome = method.apply(handler, case)
    assert outcome.success
    assert outcome.metadata["delta_norm"] > 1e-6
    assert not torch.equal(snapshot_layer(handler), original)
    assert all(not parameter.requires_grad for parameter in handler.model.parameters())
    assert_no_hooks(handler)
    metrics = method.evaluate(handler, case, outcome)
    prompt = FACT[0].format(FACT[1])
    count = len(handler.tokenizer.encode(FACT[2].strip(), add_special_tokens=False))
    assert metrics["rewrite_nll"]["target_new"] == pytest.approx(-score_target(handler, prompt, FACT[2]) / count)
    # All four decoder blocks execute during chat scoring; no cached states are
    # retained. Restoring the actual rectangular MLP projection recovers scores.
    restore_layer(handler, outcome.restorations[handler._layer_name_template.format(handler._layer)])
    assert torch.equal(snapshot_layer(handler), original)
    assert all(not parameter.requires_grad for parameter in handler.model.parameters())
    assert method.evaluate(handler, case, outcome) == before
    assert_no_hooks(handler)


def test_real_qwen35_early_exit_and_failure_remove_hooks(qwen_handler_factory, monkeypatch):
    handler = qwen_handler_factory()
    visits = []
    hook = handler.model.model.layers[1].register_forward_pre_hook(lambda *args: visits.append(1))
    try:
        common.gather_k(handler, FACT, N=3, batch_size=2)
        assert not visits
        handler.cfg.model.k_early_exit = False
        common.gather_k(handler, FACT, N=3, batch_size=2)
        assert visits
    finally:
        hook.remove()
    assert_no_hooks(handler)
    original = snapshot_layer(handler)

    def fail(*args, **kwargs):
        raise RuntimeError("upper hybrid block failed")

    monkeypatch.setattr(handler.model.model.layers[2], "forward", fail)
    handler.cfg.model.k_early_exit = True
    with pytest.raises(RuntimeError, match="upper hybrid block failed"):
        RomeT4EditMethod().apply(handler, {"fact_tuple": FACT})
    assert torch.equal(snapshot_layer(handler), original)
    assert_no_hooks(handler)
