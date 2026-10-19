"""CPU checks for the original ROME objective, batching, and hook lifetime."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.rome import common

HIDDEN, VOCAB, EDIT_LAYER = 12, 128, 1
FACT = ("{} is located in", "Eiffel", " Rome", " Paris")
PREFIXES = ["{}", "Fact: {}", "Historically, {}", "Note: {}", "In brief, {}"]


class Encoding(dict):
    __getattr__ = dict.__getitem__


class CharTokenizer:
    pad_token_id, eos_token_id, bos_token_id = 0, 1, None

    def __init__(self, padding_side):
        self.padding_side = padding_side

    def encode(self, text, **_):
        return [ord(char) for char in text]

    def decode(self, ids, **_):
        return "".join(chr(int(token)) for token in ids if int(token) > 1)

    def __call__(self, texts, **_):
        rows = [self.encode(text) for text in ([texts] if isinstance(texts, str) else texts)]
        ids = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            start = ids.shape[1] - len(row) if self.padding_side == "left" else 0
            ids[i, start:start + len(row)] = torch.tensor(row)
            mask[i, start:start + len(row)] = 1
        return Encoding(input_ids=ids, attention_mask=mask)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(HIDDEN, HIDDEN)
        self.down = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, hidden, mask):
        # Running attention over real tokens lets subject edits affect later logits.
        mixed = (hidden * mask[..., None]).cumsum(1) / mask.cumsum(1).clamp_min(1)[..., None]
        hidden = hidden + mixed
        return hidden + self.down(torch.tanh(self.up(hidden)))


class TinyLM(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([TinyBlock() for _ in range(3)])
        self.head = nn.Linear(HIDDEN, VOCAB)
        self.config = SimpleNamespace(residual_multiplier=1.0)
        self.tokenizer, self.calls = tokenizer, []

    def forward(self, input_ids, attention_mask, **_):
        self.calls.append([self.tokenizer.decode(row[mask.bool()])
                           for row, mask in zip(input_ids, attention_mask)])
        positions = attention_mask.cumsum(1).clamp_min(1) - 1
        hidden = self.embed(input_ids) + positions[..., None] * 0.01
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return SimpleNamespace(logits=self.head(hidden))


class Handler:
    def __init__(self, padding_side="right", **config):
        torch.manual_seed(19)
        self.tokenizer = CharTokenizer(padding_side)
        self.model = TinyLM(self.tokenizer).eval().requires_grad_(False)
        self.device, self.dtype = torch.device("cpu"), torch.float32
        self.emb_shape, self._layer = HIDDEN, EDIT_LAYER
        self._layer_name_template = "layers.{}.down"
        self.lr, self.kl_factor, self.weight_decay = 0.12, 0.4, 0.015
        self.cfg = SimpleNamespace(model=SimpleNamespace(
            prefix_mode="static", prefix_range=[2, 4], max_norm_multiplier=100,
            **config))
        self.prefix_handler = common.PrefixGenerationHandler(self.cfg.model)
        self.device_manager = SimpleNamespace(
            safe_to_device=lambda tensor, device=None: tensor.to(device), clear_cache=lambda: None)
        self._hooks = []

    def tokenize_prompt(self, texts, **kwargs):
        return self.tokenizer(texts, **kwargs)

    def _get_module(self, name):
        return self.model.get_submodule(name)

    def set_delta_hook(self, hook):
        self._hooks.append(self._get_module(self._layer_name_template.format(self._layer)).register_forward_hook(hook))

    def set_k_hook(self, hook):
        self._hooks.append(self._get_module(self._layer_name_template.format(self._layer)).register_forward_pre_hook(hook))

    def remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()


@pytest.fixture(autouse=True)
def fixed_prefixes(monkeypatch):
    # Deliberately fewer than the requested nine prompts.
    monkeypatch.setattr(common.PrefixGenerationHandler, "generate", lambda *_: PREFIXES.copy())
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def assert_clean(handler):
    assert not handler._hooks
    for module in handler.model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks
    assert all(not p.requires_grad and p.grad is None for p in handler.model.parameters())


def main_texts():
    return [prefix.format(FACT[0].format(FACT[1])) + FACT[2][:-1] for prefix in PREFIXES]


def reference_gradient(handler, value):
    """Independent row-wise objective: character positions, CE, explicit KL sum."""
    delta = value.detach().clone().requires_grad_()
    module = handler.model.layers[EDIT_LAYER].down
    initial = []

    def forward(text, perturbation, capture=False):
        position = text.index(FACT[1]) + len(FACT[1]) - 1

        def inject(_, inputs, output):
            if capture:
                initial.append(output[0, position].detach().clone())
            edited = output.clone()
            edited[0, position] += perturbation
            return edited

        handle = module.register_forward_hook(inject)
        try:
            return handler.model(**handler.tokenize_prompt(text)).logits[0]
        finally:
            handle.remove()

    targets = torch.tensor(handler.tokenizer.encode(FACT[2]))
    prediction = []
    for index, text in enumerate(main_texts()):
        logits = forward(text, delta, capture=index == 0)
        prediction.append(F.cross_entropy(logits[-len(targets):], targets))
    kl_text = FACT[1] + " is a"
    baseline = forward(kl_text, torch.zeros_like(delta)).detach()[-1].log_softmax(-1)
    current = forward(kl_text, delta)[-1].log_softmax(-1)
    kl = (current.exp() * (current - baseline)).sum()
    objective = (torch.stack(prediction).mean() + handler.kl_factor * kl
                 + handler.weight_decay * delta.norm() / initial[0].square().sum())
    return torch.autograd.grad(objective, delta)[0]


def run_optimization(monkeypatch, padding_side, batch_size, configured=False):
    config = {"v_batch_size": batch_size} if configured and batch_size else {}
    if not configured:
        config = {"v_batch_size": 1}  # The explicit argument must override configuration.
    handler = Handler(padding_side, **config)
    history = []
    original_step = torch.optim.Adam.step

    def record_step(optimizer, *args, **kwargs):
        delta = optimizer.param_groups[0]["params"][0]
        history.append((delta.detach().clone(), delta.grad.detach().clone()))
        return original_step(optimizer, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", record_step)
        kwargs = {} if configured else {"batch_size": batch_size}
        delta = common.optimize_v(handler, FACT, N_prompts=9, N_optim_steps=4, verbose=False, **kwargs)
    assert delta is not None and torch.isfinite(delta).all() and delta.norm() > 1e-5
    assert len(history) == 3
    assert_clean(handler)
    expected = main_texts() + [FACT[1] + " is a"]
    assert [text for call in handler.model.calls for text in call] == expected * 4
    size = batch_size or len(expected)
    assert [len(call) for call in handler.model.calls] == [
        len(expected[start:start + size]) for start in range(0, len(expected), size)] * 4
    return handler, delta.detach(), history


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_optimize_v_gradients_and_delta_match_full_objective(monkeypatch, padding_side, batch_size):
    full, expected, full_history = run_optimization(monkeypatch, padding_side, 0, configured=True)
    _, actual, history = run_optimization(monkeypatch, padding_side, batch_size)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for (value, gradient), (full_value, full_gradient) in zip(history, full_history):
        torch.testing.assert_close(value, full_value, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(gradient, full_gradient, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(gradient, reference_gradient(full, value), atol=2e-6, rtol=2e-5)
        assert torch.isfinite(gradient).all() and gradient.norm() > 1e-6


def test_optimize_v_uses_configured_batch_size(monkeypatch):
    _, configured, _ = run_optimization(monkeypatch, "left", 2, configured=True)
    _, explicit, _ = run_optimization(monkeypatch, "left", 2)
    torch.testing.assert_close(configured, explicit)


def test_optimize_v_selects_positions_before_softmax(monkeypatch):
    shapes = []
    for owner in (torch, F):
        original = owner.log_softmax

        def record(logits, *args, _original=original, **kwargs):
            shapes.append(logits.shape)
            return _original(logits, *args, **kwargs)

        monkeypatch.setattr(owner, "log_softmax", record)
    run_optimization(monkeypatch, "left", 0, configured=True)
    assert shapes
    assert all(shape.numel() <= (len(PREFIXES) * len(FACT[2]) + 1) * VOCAB for shape in shapes)


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("batch_size", [0, 1, 2, 5])
def test_gather_k_matches_full_activations_and_stops_early(padding_side, batch_size):
    handler = Handler(padding_side, **({"k_batch_size": batch_size} if batch_size else {}))
    texts = [prefix.format(FACT[1]) for prefix in PREFIXES]
    inputs = handler.tokenize_prompt(texts)
    captured = []
    module = handler.model.layers[EDIT_LAYER].down
    handle = module.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
    try:
        handler.model(**inputs)
    finally:
        handle.remove()
    ends = [max(torch.nonzero(mask).flatten()).item() for mask in inputs.attention_mask]
    expected = captured[0][torch.arange(len(texts)), ends].mean(0)
    handler.model.calls.clear()

    def upper_layer(*_):
        pytest.fail("key gathering executed above the edit layer")

    handle = handler.model.layers[EDIT_LAYER + 1].register_forward_pre_hook(upper_layer)
    try:
        kwargs = {"batch_size": batch_size} if padding_side == "right" else {}
        actual = common.gather_k(handler, FACT, N=9, **kwargs)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert not actual.requires_grad and actual.grad_fn is None
    assert [text for call in handler.model.calls for text in call] == texts
    assert max(map(len, handler.model.calls)) <= (batch_size or len(texts))
    assert_clean(handler)


@pytest.mark.parametrize("step", ["gather_k", "optimize_v"])
def test_forward_failure_removes_hooks(monkeypatch, step):
    handler = Handler()

    def fail(*args, **kwargs):
        assert handler._hooks, "failure must occur after hook registration"
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(handler.model, "forward", fail)
    kwargs = {"N": 9} if step == "gather_k" else {"N_prompts": 9, "N_optim_steps": 3}
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        getattr(common, step)(handler, FACT, batch_size=2, **kwargs)
    assert_clean(handler)


@pytest.mark.parametrize("transposed", [False, True])
def test_insert_kv_rectangular_rank_one_update_is_frozen(monkeypatch, transposed):
    handler = Handler(delta_scale=2.0)
    module = handler.model.layers[EDIT_LAYER].down
    canonical = torch.randn(HIDDEN, HIDDEN + 3)
    original = canonical.T.contiguous() if transposed else canonical.clone()
    module.weight = nn.Parameter(original.clone(), requires_grad=False)
    k = torch.randn(HIDDEN, requires_grad=True)
    delta = torch.randn(HIDDEN + 3, requires_grad=True)
    factor = torch.randn(HIDDEN, HIDDEN)
    inverse_covariance = factor @ factor.T + torch.eye(HIDDEN)
    monkeypatch.setattr(common, "get_second_moment", lambda _: inverse_covariance)
    with torch.no_grad():
        direction = inverse_covariance @ k
        update = torch.outer(direction, 2 * delta) / torch.dot(k, direction)
        expected = canonical + update
        expected = expected.T if transposed else expected
    new, old, returned_update = common.insert_kv(handler, k, delta)
    torch.testing.assert_close(new, expected)
    torch.testing.assert_close(module.weight, expected)
    torch.testing.assert_close(old, canonical)
    torch.testing.assert_close(returned_update, update)
    assert not torch.equal(module.weight, original)
    assert all(not tensor.requires_grad and tensor.grad_fn is None for tensor in (new, old, returned_update))
    assert_clean(handler)


def test_real_qwen35_hybrid_layers_match_across_batches():
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers import Qwen3_5TextConfig

    keys, values = [], []
    for batch_size in (0, 2):
        handler = Handler()
        config = Qwen3_5TextConfig(
            vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=24,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=6,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 1.0},
            attn_implementation="eager",
        )
        torch.manual_seed(27)
        handler.model = modeling.Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
        handler._layer_name_template = "model.layers.{}.mlp.down_proj"
        handler.lr = 0.01
        keys.append(common.gather_k(handler, FACT, N=9, batch_size=batch_size))
        values.append(common.optimize_v(handler, FACT, 9, 3, batch_size=batch_size, verbose=False))
        assert torch.isfinite(values[-1]).all() and values[-1].norm() > 1e-6
        assert_clean(handler)
    torch.testing.assert_close(keys[0], keys[1], atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(values[0], values[1], atol=1e-6, rtol=1e-4)
