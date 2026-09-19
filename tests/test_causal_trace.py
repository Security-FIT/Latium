"""Checks for the fixed corruption protocol and the frozen held-out intervention."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from src.causal_trace.causal_trace import (
    TraceExample,
    _overwrite_model_config_layer,
    _resolve_model_config_path,
    _restore_windows,
    _run_causal_trace,
    _trace_example,
)
from src.causal_trace.model_adapter import (
    embedding_std,
    module_dict,
    resolve_mlp_output_name,
    temporary_hooks,
)
from src.causal_trace.selection import build_window, select_window, summarize_windows
from src.causal_trace.tokenization import TraceValidationError, find_subject_span, target_first_token_id


ROOT = Path(__file__).resolve().parents[1]


class WordTokenizer:
    bos_token_id = None

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False, return_tensors=None):
        del add_special_tokens
        words = text.split()
        ids = [int(word[1:]) if word.startswith("w") and word[1:].isdigit() else 0 for word in words]
        payload = {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}
        if return_offsets_mapping:
            positions = []
            cursor = 0
            for word in words:
                start = text.index(word, cursor)
                positions.append((start, start + len(word)))
                cursor = start + len(word)
            payload["offset_mapping"] = torch.tensor([positions])
        return payload

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(f"w{int(item)}" for item in ids)


class ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Identity()

    def forward(self, hidden):
        return hidden + 0.1 * self.mlp(hidden)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(6, 4)
        self.layers = torch.nn.ModuleList(ToyLayer() for _ in range(4))
        self.config = SimpleNamespace(_commit_hash="toy-revision")
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[1, 0] = 2.0

    @property
    def dtype(self):
        return self.embed.weight.dtype

    def forward(self, input_ids, use_cache=False):
        del use_cache
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = torch.zeros(*hidden.shape[:2], 6)
        logits[:, :, 2] = 3.0 * (hidden[:, :, 0] + hidden[:, :1, 0])
        return SimpleNamespace(logits=logits)


def toy_handler():
    model = ToyModel().eval()
    cfg = OmegaConf.create(
        {"model": {"name": "toy", "corrupt_layer_name_template": "embed", "restore_layer_name_template": "layers.{}"}}
    )
    handler = SimpleNamespace(
        model=model,
        tokenizer=WordTokenizer(),
        num_of_layers=4,
        _layer_name_template="layers.{}.mlp",
        cfg=cfg,
    )
    handler.tokenize_prompt = lambda prompt: handler.tokenizer(prompt, return_tensors="pt")
    return handler


def test_subject_span_and_first_target_token():
    tokenizer = WordTokenizer()
    assert find_subject_span(tokenizer, "w1 w0", "w1").positions == [0]
    assert target_first_token_id(tokenizer, "w2 w3") == 2
    with pytest.raises(TraceValidationError, match="appears 2 times"):
        find_subject_span(tokenizer, "w1 w1", "w1")


def test_one_layer_windows_cover_every_layer():
    assert [build_window(i, 1, 4).layers for i in range(4)] == [[0], [1], [2], [3]]
    assert build_window(0, 2, 4).layers == [0]


def test_fixed_corruption_and_restoration_match_independent_batch():
    handler = toy_handler()
    modules = module_dict(handler.model)
    module_names = {layer: resolve_mlp_output_name(handler, modules, layer) for layer in range(4)}
    example = TraceExample("case-1", "w1 w0", "w1", "w2")
    prepared = _trace_example(
        handler,
        modules,
        module_names,
        modules["embed"],
        example,
        num_noise_samples=7,
        noise_batch_size=3,
        noise_multiplier=3.0,
        embedding_scale=embedding_std(handler, modules),
        seed=5,
        require_correct_clean=True,
    )
    assert prepared.row["total_effect"] > 0
    windows = [build_window(i, 1, 4) for i in range(4)]
    _restore_windows(handler, modules, module_names, modules["embed"], prepared, windows, noise_batch_size=1)
    measured = np.asarray(prepared.row["window_restore_probabilities"])

    def corrupt(_module, _input, output):
        changed = output.clone()
        changed[1:, 0, :] += prepared.noise_samples[:, 0, :]
        return changed

    expected = []
    repeated = {key: value.repeat(8, 1) for key, value in prepared.inputs.items()}
    for window in windows:

        def restore(_module, _input, output):
            changed = output.clone()
            changed[1:, 0, :] = output[0, 0, :]
            return changed

        with (
            torch.inference_mode(),
            temporary_hooks([(modules["embed"], corrupt), (modules[module_names[window.center]], restore)]),
        ):
            logits = handler.model(**repeated).logits
        expected.append(torch.softmax(logits[1:, -1, :].float(), dim=-1)[:, 2].numpy())
    np.testing.assert_allclose(measured, expected, atol=1e-7, rtol=0)
    original = measured.copy()
    _restore_windows(handler, modules, module_names, modules["embed"], prepared, windows, noise_batch_size=3)
    np.testing.assert_allclose(prepared.row["window_restore_probabilities"], original, atol=1e-7, rtol=0)
    assert sum(len(module._forward_hooks) for module in handler.model.modules()) == 0


def test_confirmation_cannot_choose_a_new_layer_or_accept_negative_center():
    windows = [build_window(i, 1, 3) for i in range(3)]
    discovery_facts = [{"window_mean_ie": [0.2, 0.1, 0.1]} for _ in range(20)]
    confirmation_facts = [{"window_mean_ie": [-0.02, 0.3, 0.3]} for _ in range(20)]
    discovery = summarize_windows(
        discovery_facts, windows, window_size=1, bootstrap_samples=100, confidence_level=0.95, seed=1
    )
    confirmation = summarize_windows(
        confirmation_facts, windows, window_size=1, bootstrap_samples=100, confidence_level=0.95, seed=2
    )
    selected = select_window(discovery, confirmation, minimum_confirmation_facts=20)
    assert selected["discovery_trace_center"] == 0
    assert selected["selected_trace_center"] is None
    assert selected["confirmation_passed"] is False
    assert selected["confirmation_ci_upper"] < 0


def test_runner_only_sweeps_discovery_and_frozen_confirmation(monkeypatch, tmp_path):
    from src.causal_trace import causal_trace as trace_module

    handler = toy_handler()
    with hydra.initialize_config_dir(config_dir=str(ROOT / "src/config"), version_base=None):
        cfg = hydra.compose(
            config_name="latium",
            overrides=[
                "command=causal_trace",
                "command.causal_trace.num_valid_facts=4",
                "command.causal_trace.num_noise_samples=7",
                "command.causal_trace.noise_batch_size=3",
                f"command.causal_trace.output_dir={tmp_path}",
            ],
        )
    cfg.model.corrupt_layer_name_template = "embed"
    cfg.model.layer = 2
    handler.cfg = cfg
    monkeypatch.setattr(
        trace_module,
        "_dataset_examples",
        lambda _cfg, max_scan: (TraceExample(str(i), "w1 w0", "w1", "w2") for i in range(min(100, max_scan))),
    )
    output_dir = _run_causal_trace(cfg, handler)
    summary = json.loads((output_dir / "summary.json").read_text())
    rows = [json.loads(line) for line in (output_dir / "fact_results.jsonl").read_text().splitlines()]
    assert summary["num_valid_facts"] == 4
    assert summary["window_size"] == 1
    assert summary["selected_layer_directly_tested"] is True
    assert {len(row["window_centers"]) for row in rows if row["split"] == "discovery"} == {4}
    assert {len(row["window_centers"]) for row in rows if row["split"] == "confirmation"} == {1}
    assert set(pd.read_csv(output_dir / "confirmation_windows.csv")["window_center"]) == {
        summary["discovery_trace_center"]
    }


def test_model_config_overwrite_preserves_other_yaml(tmp_path):
    path = tmp_path / "model.yaml"
    original = '# comment\nname: "toy"\nlayer: 4  # current\ndtype: "bf16"\n'
    path.write_text(original)
    assert _overwrite_model_config_layer(path, 2) == 4
    assert path.read_text() == original.replace("layer: 4", "layer: 2")


def test_resolve_model_config_uses_hydra_choice(tmp_path):
    selected = tmp_path / "selected.yaml"
    selected.write_text('name: "toy"\nlayer: 3\n')
    cfg = OmegaConf.create({"model": {"name": "toy"}, "hydra": {"runtime": {"choices": {"model": "selected"}}}})
    assert _resolve_model_config_path(cfg, config_dir=tmp_path) == selected
