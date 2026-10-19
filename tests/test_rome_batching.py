"""Numerical parity tests for the micro-batched ROME path.

The whole premise of ``v_batch_size``/``k_batch_size`` is that they are an
*execution* change, not a *method* change. These tests pin that down on a tiny
synthetic causal LM so the invariant is checked without a GPU or a download.

:copyright: 2025 Jakub Res
:license: MIT
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.editing.rome_t4.batching import (
    gather_target_log_probs,
    iter_microbatches,
    select_batch,
    trim_padding,
)
from src.editing.rome_t4.common import gather_k, optimize_v

VOCAB = 64
HIDDEN = 16
LAYERS = 3
EDIT_LAYER = 1


# ---------------------------------------------------------------------------
# Synthetic model / tokenizer / handler
# ---------------------------------------------------------------------------
def _causal_mix(hidden: torch.Tensor) -> torch.Tensor:
    """Cheap stand-in for causal attention: a running mean over past tokens.

    Without some form of cross-token mixing, a residual injected at the subject
    token could never influence the logits at the target position, and every
    gradient in these tests would be identically zero.
    """
    seq_len = hidden.size(1)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=hidden.dtype, device=hidden.device))
    mask = mask / mask.sum(dim=1, keepdim=True)
    return mask @ hidden


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(HIDDEN, HIDDEN)
        self.mlp.down_proj = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + _causal_mix(hidden)
        return hidden + self.mlp.down_proj(torch.tanh(self.mlp.gate_proj(hidden)))


class TinyLM(nn.Module):
    """Minimal decoder stack exposing ``layers.{i}.mlp.down_proj``."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList(TinyBlock() for _ in range(LAYERS))
        self.lm_head = nn.Linear(HIDDEN, VOCAB)
        self.config = SimpleNamespace(num_hidden_layers=LAYERS, residual_multiplier=1.0)

    def forward(self, input_ids, attention_mask=None, use_cache=False, **_):
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


class Encoding(dict):
    """Stand-in for ``BatchEncoding`` (attribute + item access)."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc


class CharTokenizer:
    """Deterministic byte-level tokenizer; id 0 is padding."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = None

    def _ids(self, text: str) -> list[int]:
        return [(ord(ch) % (VOCAB - 2)) + 2 for ch in text]

    def __call__(self, text, return_tensors="pt", padding=True, **_):
        texts = [text] if isinstance(text, str) else list(text)
        rows = [self._ids(item) for item in texts]
        width = max(len(row) for row in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[i, : len(row)] = 1
        return Encoding(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, ids, skip_special_tokens=False):
        ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
        return "".join(chr(int(i) - 2) for i in ids if int(i) >= 2)

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)


class StubDeviceManager:
    def safe_to_device(self, tensor, device=None):
        return tensor if device is None else tensor.to(device)

    def clear_cache(self):
        return None


class StubHandler:
    """The minimal surface of ``ModelHandler`` that the ROME steps touch."""

    def __init__(self, **model_overrides):
        torch.manual_seed(0)
        self.model = TinyLM().to(torch.float32)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

        self.tokenizer = CharTokenizer()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.device_manager = StubDeviceManager()
        self.is_multi_gpu = False

        self._layer_name_template = "layers.{}.mlp.down_proj"
        self._layer = EDIT_LAYER
        self.emb_shape = HIDDEN
        self.lr = 0.5
        self.kl_factor = 0.0625
        self.weight_decay = 0.5
        self.epochs = 4

        model_cfg = dict(
            name="tiny/lm",
            layer=EDIT_LAYER,
            prefix_mode="static",
            prefix_range=[2, 4],
            v_N=6,
            k_N=6,
            max_norm_multiplier=4,
            optimize_v_clear_cache_every=0,
            v_batch_size=0,
            k_batch_size=0,
            k_early_exit=True,
        )
        model_cfg.update(model_overrides)
        self.cfg = SimpleNamespace(
            model=SimpleNamespace(**model_cfg),
            generation=SimpleNamespace(v_N=model_cfg["v_N"], k_N=model_cfg["k_N"]),
            runtime=SimpleNamespace(prefix_log_all=False),
        )

        from src.rome.common import PrefixGenerationHandler

        self.prefix_handler = PrefixGenerationHandler(self.cfg.model)
        self._hooks = []

    # -- handler API ------------------------------------------------------
    def tokenize_prompt(self, prompt_text, **_):
        return self.tokenizer(prompt_text)

    def _get_module(self, name):
        for module_name, module in self.model.named_modules():
            if module_name == name:
                return module
        raise KeyError(name)

    def get_module_device(self, name=None):
        return self.device

    def set_delta_hook(self, fn):
        module = self._get_module(self._layer_name_template.format(self._layer))
        self._hooks.append(module.register_forward_hook(fn))

    def set_k_hook(self, fn):
        module = self._get_module(self._layer_name_template.format(self._layer))
        self._hooks.append(module.register_forward_pre_hook(fn))

    def remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks = []


FACT = ("{} is located in", "Eiffel", " Rome", " Paris")


def _run_optimize_v(batch_size: int, steps: int = 4) -> torch.Tensor:
    """Run optimize_v with a fixed seed so prefixes and init are identical."""
    random.seed(1234)
    torch.manual_seed(1234)
    handler = StubHandler(v_batch_size=batch_size)
    delta = optimize_v(
        handler,
        fact_tuple=FACT,
        N_prompts=6,
        N_optim_steps=steps,
        verbose=False,
        batch_size=batch_size,
    )
    assert delta is not None
    return delta.detach().clone()


def _run_gather_k(batch_size: int, early_exit: bool) -> torch.Tensor:
    random.seed(99)
    torch.manual_seed(99)
    handler = StubHandler(k_batch_size=batch_size, k_early_exit=early_exit)
    k = gather_k(handler, fact_tuple=FACT, N=6, batch_size=batch_size)
    assert k is not None
    return k.detach().clone()


# ---------------------------------------------------------------------------
# Slicing primitives
# ---------------------------------------------------------------------------
def test_iter_microbatches_covers_every_index_exactly_once() -> None:
    for total in (1, 5, 8, 17):
        for batch_size in (0, 1, 3, 8, 100):
            spans = list(iter_microbatches(total, batch_size))
            covered = [i for start, end in spans for i in range(start, end)]
            assert covered == list(range(total))


def test_iter_microbatches_is_empty_for_zero_total() -> None:
    assert list(iter_microbatches(0, 4)) == []


def test_select_batch_preserves_row_order() -> None:
    inputs = {"input_ids": torch.arange(20).reshape(5, 4)}
    picked = select_batch(inputs, [3, 1])
    assert torch.equal(picked["input_ids"], torch.stack([torch.arange(12, 16), torch.arange(4, 8)]))


def test_trim_padding_drops_only_all_padding_columns() -> None:
    inputs = {
        "input_ids": torch.tensor([[5, 6, 0, 0], [7, 0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]]),
    }
    trimmed = trim_padding(inputs)
    assert trimmed["input_ids"].shape[1] == 2
    assert torch.equal(trimmed["attention_mask"], torch.tensor([[1, 1], [1, 0]]))


def test_trim_padding_is_a_noop_without_padding() -> None:
    inputs = {
        "input_ids": torch.tensor([[5, 6], [7, 8]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
    }
    assert trim_padding(inputs)["input_ids"].shape[1] == 2


def test_trim_padding_preserves_absolute_positions_with_left_padding() -> None:
    inputs = {
        "input_ids": torch.tensor([[0, 0, 5, 6, 0], [0, 7, 8, 9, 0]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 1, 0]]),
    }
    trimmed = trim_padding(inputs)
    assert torch.equal(trimmed["input_ids"], inputs["input_ids"][:, :4])
    assert torch.equal(trimmed["attention_mask"], inputs["attention_mask"][:, :4])


# ---------------------------------------------------------------------------
# Loss gathering
# ---------------------------------------------------------------------------
def test_gather_target_log_probs_matches_full_log_softmax() -> None:
    torch.manual_seed(7)
    logits = torch.randn(4, 9, VOCAB)
    positions = torch.tensor([[2, 3], [4, 5], [1, 2], [6, 7]])
    token_ids = torch.randint(0, VOCAB, (4, 2))

    reference = torch.log_softmax(logits, dim=2)[
        torch.arange(4).unsqueeze(1), positions, token_ids
    ]
    assert torch.allclose(gather_target_log_probs(logits, positions, token_ids), reference, atol=0)


# ---------------------------------------------------------------------------
# End-to-end parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch_size", [1, 2, 4, 5])
def test_optimize_v_batched_matches_single_pass(batch_size: int) -> None:
    """Gradient accumulation must reproduce the one-shot backward pass."""
    reference = _run_optimize_v(batch_size=0)
    batched = _run_optimize_v(batch_size=batch_size)
    assert torch.allclose(batched, reference, atol=1e-5, rtol=1e-4), (
        f"batch_size={batch_size} diverged: max|d|={(batched - reference).abs().max():.3e}"
    )


def test_optimize_v_actually_moves_delta() -> None:
    """Guard against the parity test passing because nothing happened."""
    delta = _run_optimize_v(batch_size=2)
    assert float(delta.norm()) > 1e-6


@pytest.mark.parametrize("batch_size", [0, 2, 5])
def test_optimize_v_matches_protected_rome(batch_size: int) -> None:
    from src.rome.common import optimize_v as protected_optimize_v

    random.seed(1234)
    torch.manual_seed(1234)
    reference = protected_optimize_v(
        StubHandler(), fact_tuple=FACT, N_prompts=6, N_optim_steps=4, verbose=False,
    )
    assert reference is not None
    assert torch.allclose(_run_optimize_v(batch_size), reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("batch_size", [0, 2, 5])
def test_gather_k_matches_protected_rome(batch_size: int) -> None:
    from src.rome.common import gather_k as protected_gather_k

    random.seed(99)
    torch.manual_seed(99)
    reference = protected_gather_k(StubHandler(), fact_tuple=FACT, N=6)
    assert torch.allclose(_run_gather_k(batch_size, early_exit=True), reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("early_exit", [True, False])
def test_gather_k_batched_matches_single_pass(batch_size: int, early_exit: bool) -> None:
    reference = _run_gather_k(batch_size=0, early_exit=False)
    batched = _run_gather_k(batch_size=batch_size, early_exit=early_exit)
    assert torch.allclose(batched, reference, atol=1e-6, rtol=1e-5)


def test_gather_k_early_exit_skips_upper_layers() -> None:
    """The early exit must not execute blocks above the edited layer."""
    random.seed(99)
    torch.manual_seed(99)
    handler = StubHandler(k_early_exit=True)

    visited: list[int] = []
    handles = [
        block.register_forward_hook(lambda *_a, _i=i: visited.append(_i))
        for i, block in enumerate(handler.model.layers)
    ]
    try:
        gather_k(handler, fact_tuple=FACT, N=4, batch_size=2)
    finally:
        for handle in handles:
            handle.remove()

    # The pre-hook fires *inside* the edited block, so that block never
    # completes and nothing above it is entered at all.
    assert visited, "no decoder block ran"
    assert max(visited) < EDIT_LAYER + 1, f"layers above {EDIT_LAYER} ran: {sorted(set(visited))}"
    assert LAYERS - 1 not in visited, "the top block should have been skipped"


# ---------------------------------------------------------------------------
# insert_kv: grad hygiene, covariance caching, in-place equivalence
# ---------------------------------------------------------------------------
@pytest.fixture
def stub_covariance(monkeypatch):
    """Replace the on-disk covariance with an in-memory SPD matrix.

    Returns a counter so tests can assert how often it was *loaded*, which is
    the thing sequential editing used to get wrong.
    """
    import src.editing.rome_t4.common as common

    torch.manual_seed(0)
    base = torch.eye(HIDDEN) + 0.01 * torch.randn(HIDDEN, HIDDEN)
    matrix = base @ base.T
    calls = {"count": 0}

    def fake_load(_handler):
        calls["count"] += 1
        return matrix.clone()

    monkeypatch.setattr(common, "_load_second_moment", fake_load)
    return calls


def _edit_once(handler):
    from src.editing.rome_t4.common import gather_k, insert_kv, optimize_v

    random.seed(5)
    torch.manual_seed(5)
    k = gather_k(handler, fact_tuple=FACT, N=6)
    delta = optimize_v(handler, fact_tuple=FACT, N_prompts=6, N_optim_steps=4, verbose=False)
    insert_kv(handler, k, delta)
    return handler._get_module(handler._layer_name_template.format(EDIT_LAYER)).weight


@pytest.mark.parametrize("in_place", [False, True])
def test_insert_kv_leaves_the_edited_weight_frozen(stub_covariance, in_place: bool) -> None:
    """A default torch.nn.Parameter re-enables grad tracking on the edited layer.

    ModelHandler freezes parameters only at construction, so a rebound
    Parameter makes every later forward retain an autograd graph.
    """
    handler = StubHandler(insert_kv_in_place=in_place, second_moment_device="cpu")
    weight = _edit_once(handler)
    assert weight.requires_grad is False


@pytest.mark.parametrize("in_place", [False, True])
def test_insert_kv_actually_changes_the_weight(stub_covariance, in_place: bool) -> None:
    handler = StubHandler(insert_kv_in_place=in_place, second_moment_device="cpu")
    before = handler._get_module(f"layers.{EDIT_LAYER}.mlp.down_proj").weight.detach().clone()
    after = _edit_once(handler).detach().clone()
    assert not torch.equal(before, after)


def test_in_place_update_matches_the_copy_path(stub_covariance) -> None:
    results = {}
    for in_place in (False, True):
        handler = StubHandler(insert_kv_in_place=in_place, second_moment_device="cpu")
        results[in_place] = _edit_once(handler).detach().clone()
    assert torch.allclose(results[True], results[False], atol=1e-6)


def test_covariance_is_loaded_once_across_sequential_edits(stub_covariance) -> None:
    """Upstream re-read and re-uploaded the covariance for every fact."""
    from src.editing.rome_t4.common import gather_k, insert_kv, optimize_v

    handler = StubHandler(second_moment_device="cpu", second_moment_cache=True)
    random.seed(5)
    torch.manual_seed(5)
    k = gather_k(handler, fact_tuple=FACT, N=6)
    delta = optimize_v(handler, fact_tuple=FACT, N_prompts=6, N_optim_steps=4, verbose=False)

    stub_covariance["count"] = 0
    for _ in range(3):
        insert_kv(handler, k, delta)
    assert stub_covariance["count"] == 1


def test_covariance_cache_can_be_disabled(stub_covariance) -> None:
    from src.editing.rome_t4.common import get_second_moment

    handler = StubHandler(second_moment_device="cpu", second_moment_cache=False)
    stub_covariance["count"] = 0
    for _ in range(3):
        get_second_moment(handler)
    assert stub_covariance["count"] == 3


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("in_place", [False, True])
def test_insert_kv_matches_protected_rome_for_rectangular_weights(monkeypatch, transposed, in_place) -> None:
    from src.editing.rome_t4 import common
    from src.rome import common as protected

    handler = StubHandler(insert_kv_in_place=in_place)
    module = handler._get_module(f"layers.{EDIT_LAYER}.mlp.down_proj")
    weight = torch.randn(HIDDEN, HIDDEN + 3)
    if transposed:
        weight = weight.T.contiguous()
    module.weight = nn.Parameter(weight.clone(), requires_grad=False)
    k, delta = torch.randn(HIDDEN), torch.randn(HIDDEN + 3)
    covariance = torch.eye(HIDDEN)
    monkeypatch.setattr(protected, "get_second_moment", lambda _: covariance)
    monkeypatch.setattr(common, "get_second_moment", lambda _: covariance)
    protected.insert_kv(handler, k, delta)
    reference = module.weight.detach().clone()

    module.weight = nn.Parameter(weight.clone(), requires_grad=False)
    common.insert_kv(handler, k, delta)
    assert torch.allclose(module.weight, reference, atol=1e-6)
    assert not module.weight.requires_grad


@pytest.mark.parametrize("in_place", [False, True])
def test_registry_method_applies_and_execution_restores(stub_covariance, in_place) -> None:
    from src.editing.registry import get_edit_method
    from src.structural.execution.edit_execution import restore

    handler = StubHandler(insert_kv_in_place=in_place, v_batch_size=2, k_batch_size=2)
    layer = f"layers.{EDIT_LAYER}.mlp.down_proj"
    original = handler._get_module(layer).weight.detach().clone()
    method = get_edit_method("rome_t4")
    outcome = method.apply(handler, {"fact_tuple": FACT})

    assert outcome.success
    assert outcome.modified_weights == {"proj": (EDIT_LAYER,)}
    assert not torch.equal(handler._get_module(layer).weight, original)
    assert torch.equal(outcome.restorations[layer], original)
    assert not handler._hooks
    restore(handler, outcome)
    assert torch.equal(handler._get_module(layer).weight, original)
    assert not handler._get_module(layer).weight.requires_grad


def test_registry_method_rolls_back_partial_insertion_and_removes_hooks(monkeypatch) -> None:
    from src.editing.registry import get_edit_method
    from src.editing.rome_t4 import common

    handler = StubHandler(v_batch_size=2)
    module = handler._get_module(f"layers.{EDIT_LAYER}.mlp.down_proj")
    original = module.weight.detach().clone()

    def fail_insertion(owner, *_):
        with torch.no_grad():
            module.weight.add_(1)
        owner.set_delta_hook(lambda *_: None)
        raise RuntimeError("insertion failed")

    monkeypatch.setattr(common, "insert_kv", fail_insertion)
    with pytest.raises(RuntimeError, match="insertion failed"):
        get_edit_method("rome_t4").apply(handler, {"fact_tuple": FACT})
    assert torch.equal(module.weight, original)
    assert not module.weight.requires_grad
    assert not handler._hooks
    assert not module._forward_hooks


@pytest.mark.parametrize("step", ["gather_k", "optimize_v"])
def test_notebook_api_cleans_hooks_after_forward_failure(monkeypatch, step) -> None:
    from src.editing.rome_t4 import api

    handler = StubHandler()
    if step == "optimize_v":
        monkeypatch.setattr(api, "gather_k", lambda *_, **__: torch.ones(HIDDEN))

    def fail_forward(*_, **__):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(handler.model, "forward", fail_forward)
    with pytest.raises(RuntimeError, match="forward failed"):
        api.apply_rome_edit(handler, FACT, verbose=False)
    module = handler._get_module(f"layers.{EDIT_LAYER}.mlp.down_proj")
    assert not handler._hooks
    assert not module._forward_hooks
    assert not module._forward_pre_hooks


def test_notebook_snapshot_survives_repeated_in_place_edits(stub_covariance) -> None:
    from src.editing.rome_t4.api import apply_rome_edit, restore_layer, snapshot_layer

    handler = StubHandler(insert_kv_in_place=True, v_batch_size=2)
    snapshot = snapshot_layer(handler)
    original = snapshot.clone()
    for _ in range(2):
        apply_rome_edit(handler, FACT, verbose=False)
        restore_layer(handler, snapshot)
        assert torch.equal(snapshot, original)
        assert torch.equal(snapshot_layer(handler), original)


def test_covariance_cache_invalidates_when_source_is_replaced(tmp_path) -> None:
    from src.editing.rome_t4.common import get_second_moment

    handler = StubHandler(second_moment_device="cpu")
    handler.second_moment_path = tmp_path / "covariance.pt"
    torch.save(torch.eye(HIDDEN), handler.second_moment_path)
    first = get_second_moment(handler)
    assert get_second_moment(handler) is first

    # Different shape guarantees a different serialized size even on coarse clocks.
    replacement = torch.eye(HIDDEN + 1)
    torch.save(replacement, handler.second_moment_path)
    assert torch.equal(get_second_moment(handler), replacement)
