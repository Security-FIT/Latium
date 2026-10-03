"""CPU checks for historical replay and corrected token-by-block tracing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from src import main
from src.causal_trace import legacy_fixed
from src.causal_trace.causal_trace import TraceExample
from src.causal_trace.model_adapter import module_dict
from src.causal_trace.tokenization import TraceValidationError
from src.command_handlers import operations

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    bos_token_id = None
    vocabulary = {"Ada": 0, "Lovelace": 1, "lived": 2, "London": 3, "New": 3, "York": 4}

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, return_tensors=None):
        del add_special_tokens
        tokens = text.split()
        ids = [self.vocabulary[token] for token in tokens]
        offsets = []
        cursor = 0
        for token in tokens:
            start = text.index(token, cursor)
            cursor = start + len(token)
            offsets.append((start, cursor))
        result = {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor([offsets])
        return result

    def decode(self, ids):
        return str(ids)


class Block(torch.nn.Module):
    def __init__(self, *, flattened=False, tuple_output=False):
        super().__init__()
        self.flattened = flattened
        self.tuple_output = tuple_output
        self.received = []

    def forward(self, hidden):
        self.received.append(hidden.detach().clone())
        result = torch.tanh(hidden * 0.7 + 0.1)
        if self.flattened:
            result = result.reshape(-1, result.shape[-1])
        return (result, "unchanged auxiliary output") if self.tuple_output else result


class Model(torch.nn.Module):
    def __init__(self, *, flattened=False, tuple_output=False, fail_on=None):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 3)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(15).reshape(5, 3) * 0.07)
        self.blocks = torch.nn.ModuleList([Block(flattened=flattened, tuple_output=tuple_output) for _ in range(2)])
        self.calls = 0
        self.fail_on = fail_on

    def forward(self, input_ids, use_cache=False):
        del use_cache
        self.calls += 1
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            output = block(hidden)
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.reshape(input_ids.shape[0], input_ids.shape[1], -1)
            if self.calls == self.fail_on:
                raise RuntimeError("injected forward failure")
        score = hidden.sum(dim=(1, 2))
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 5)
        logits[:, -1, 3] = score
        return SimpleNamespace(logits=logits)


def handler_for(model=None, cfg=None):
    cfg = (
        cfg
        if cfg is not None
        else OmegaConf.create(
            {
                "model": {
                    "name": "toy",
                    "restore_layer_name_template": "blocks.{}",
                    "corrupt_layer_name_template": "embedding",
                },
            }
        )
    )
    tokenizer = Tokenizer()
    return SimpleNamespace(
        cfg=cfg,
        model=model if model is not None else Model(),
        tokenizer=tokenizer,
        num_of_layers=2,
        _layer_name_template="blocks.{}.projection",
        tokenize_prompt=lambda prompt: tokenizer(prompt, return_tensors="pt"),
    )


def trace(handler, *, subject="Ada Lovelace", target="New York", samples=2):
    modules = module_dict(handler.model)
    return legacy_fixed.trace_example(
        handler,
        TraceExample("fact-1", f"{subject} lived", subject, target),
        modules=modules,
        block_names=legacy_fixed.resolve_block_names(handler, modules),
        noise_std=0.5,
        num_noise_samples=samples,
        seed=42,
        require_correct_clean=True,
    )


def assert_no_hooks(model):
    assert all(not module._forward_hooks for module in model.modules())


def test_historical_source_is_exact_legacy_blob():
    data = (ROOT / "src/causal_trace/legacy/causal_trace.py").read_bytes()
    blob = f"blob {len(data)}\0".encode() + data
    assert hashlib.sha1(blob).hexdigest() == "4e9b6df1991af7505318586b655ba5cfa5cb424f"


def test_legacy_imports_and_dispatches_without_model_load(monkeypatch, tmp_path):
    from src.causal_trace.legacy import causal_trace as original

    calls = []
    monkeypatch.setattr(original, "causal_trace", lambda cfg: calls.append(cfg))
    cfg = OmegaConf.create({"model": {"name": "toy"}, "generation": {"filename": str(tmp_path / "legacy/trace_{}")}})
    assert operations.run_causal_kuba(cfg) == 0
    assert calls == [cfg]
    assert (tmp_path / "legacy").is_dir()


@pytest.mark.parametrize(
    "command,config,name",
    [
        ("causal-kuba", "causal_kuba", "causal-kuba"),
        ("causal-kuba-fix", "causal_kuba_fix", "causal-kuba-fix"),
    ],
)
def test_variant_cli_and_hydra_composition(command, config, name, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "run_hydra", lambda args: calls.append(args) or 0)
    assert main.main([command, "model=gpt2-xl"]) == 0
    assert calls == [[f"command={config}", "model=gpt2-xl"]]
    with hydra.initialize_config_dir(config_dir=str(ROOT / "src/config"), version_base=None):
        cfg = hydra.compose(config_name="latium", overrides=[f"command={config}", "model=gpt2-xl"])
    assert cfg.command.name == name
    if config == "causal_kuba_fix":
        assert cfg.command.legacy_trace.num_noise_samples == 1
        assert "causal_trace" not in cfg.command
    else:
        assert cfg.generation.num_of_runs == 100000
        assert "causal_kuba/" in cfg.generation.filename


def test_corrected_legacy_pairs_one_shared_noise_vector_across_every_restore():
    handler = handler_for()
    result = trace(handler)
    received = handler.model.blocks[0].received
    assert len(result["rows"]) == 2 * 2 * 2
    assert len(received) == 1 + 2 * (1 + 2 * 2)
    for start in (1, 6):
        baseline = received[start]
        noise = baseline - received[0]
        assert torch.allclose(noise[0, 0], noise[0, 1])
        assert torch.equal(noise[0, 2], torch.zeros(3))
        for restoration in received[start + 1 : start + 5]:
            assert torch.equal(restoration, baseline)
    assert not torch.equal(received[1], received[6])
    for row in result["rows"]:
        assert row["indirect_effect"] == pytest.approx(row["restored_probability"] - row["corrupt_probability"])
    assert_no_hooks(handler.model)


@pytest.mark.parametrize("flattened,tuple_output", [(False, False), (False, True), (True, True)])
def test_corrected_legacy_restores_actual_block_output_at_each_layer(flattened, tuple_output):
    handler = handler_for(Model(flattened=flattened, tuple_output=tuple_output))
    result = trace(handler, subject="Ada", samples=1)
    assert {row["layer"] for row in result["rows"]} == {0, 1}
    # Restoring the sole corrupted subject token at either whole block exactly
    # recovers the clean run in this toy model; this also checks layer indexing.
    for row in result["rows"]:
        assert row["restored_probability"] == pytest.approx(result["clean_probability"])
    assert_no_hooks(handler.model)


def test_corrected_legacy_accepts_multi_token_target_with_explicit_first_token_scope():
    result = trace(handler_for())
    assert result["target_token_ids"] == [3, 4]
    assert result["target_first_token_id"] == 3
    assert result["target_scope"] == "first_continuation_token"


def test_corrected_legacy_noise_is_deterministic_without_consuming_global_rng():
    torch.manual_seed(5)
    handler = handler_for()
    rng_before = torch.random.get_rng_state().clone()
    first = trace(handler)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    second = trace(handler)
    assert first == second


def test_corrected_legacy_cleans_hooks_on_restoration_failure():
    handler = handler_for(Model(fail_on=3))
    with pytest.raises(RuntimeError, match="injected forward failure"):
        trace(handler)
    assert_no_hooks(handler.model)


def test_corrected_legacy_rejects_ambiguous_subject_without_running_model():
    handler = handler_for()
    modules = module_dict(handler.model)
    with pytest.raises(TraceValidationError, match="appears 2 times"):
        legacy_fixed.trace_example(
            handler,
            TraceExample("ambiguous", "Ada Ada lived", "Ada", "London"),
            modules=modules,
            block_names=legacy_fixed.resolve_block_names(handler, modules),
            noise_std=0.5,
            num_noise_samples=1,
            seed=1,
            require_correct_clean=True,
        )
    assert handler.model.calls == 0
    assert_no_hooks(handler.model)


def test_corrected_legacy_saves_rejections_draws_and_no_spurious_layer_selection(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "toy",
                "restore_layer_name_template": "blocks.{}",
                "corrupt_layer_name_template": "embedding",
                "corruption_noise_multiplier": 0.5,
            },
            "dataset_facts": {"name": "toy"},
            "command": {
                "legacy_trace": {
                    "output_dir": str(tmp_path),
                    "num_valid_facts": 1,
                    "max_dataset_examples_to_scan": 3,
                    "num_noise_samples": 2,
                    "noise_std": None,
                    "require_correct_clean_prediction": True,
                    "seed": 42,
                }
            },
        }
    )
    examples = [
        TraceExample("bad", "Ada Ada lived", "Ada", "London"),
        TraceExample("good", "Ada Lovelace lived", "Ada Lovelace", "New York"),
    ]
    monkeypatch.setattr(legacy_fixed, "_dataset_examples", lambda cfg, max_scan: iter(examples))
    handler = handler_for(cfg=cfg)
    path = legacy_fixed.run(cfg, handler, legacy_fixed.LegacySettings.from_config(cfg))
    summary = json.loads((path / "summary.json").read_text())
    assert summary["scanned_facts"] == 2
    assert summary["valid_facts"] == summary["rejected_facts"] == 1
    assert summary["variant"] == "causal-kuba-fix"
    assert summary["selected_layer"] is None
    assert summary["noise_std"] == 0.5
    assert summary["block_modules"] == {"0": "blocks.0", "1": "blocks.1"}
    assert len((path / "traces.csv").read_text().splitlines()) == 9
    assert len((path / "profile.csv").read_text().splitlines()) == 5
    assert not (path / "fact_000000.json").exists()
    assert (path / "fact_000001.json").is_file()
    assert_no_hooks(handler.model)
