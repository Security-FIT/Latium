"""
Tests for the prototype causal tracing utilities.
"""

from __future__ import annotations

from pathlib import Path
import sys

import hydra
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal_trace.prototype import (
    TraceValidationError,
    find_subject_span,
    make_noise_samples,
    target_first_token_id,
    temporary_hooks,
)


class _Encoding(dict):
    def __getattr__(self, name):
        return self[name]


class _ToyTokenizer:
    bos_token_id = 0

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, return_tensors=None):
        del add_special_tokens
        tokens = text.split(" ")
        ids = []
        offsets = []
        cursor = 0
        for token in tokens:
            start = text.find(token, cursor)
            end = start + len(token)
            ids.append(abs(hash(token)) % 1000 + 1)
            offsets.append((start, end))
            cursor = end
        payload = {"input_ids": ids}
        if return_offsets_mapping:
            payload["offset_mapping"] = torch.tensor([offsets])
        if return_tensors == "pt":
            payload["input_ids"] = torch.tensor([ids])
        return _Encoding(payload)

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(str(int(x)) for x in ids)


def test_find_subject_span_multi_token_subject_with_offsets() -> None:
    tokenizer = _ToyTokenizer()

    span = find_subject_span(tokenizer, "Ada Lovelace was born in London", "Ada Lovelace")

    assert span.start == 0
    assert span.end == 2
    assert span.positions == [0, 1]
    assert span.last_position == 1


def test_find_subject_span_repeated_subject_raises() -> None:
    tokenizer = _ToyTokenizer()

    with pytest.raises(TraceValidationError, match="appears 2 times"):
        find_subject_span(tokenizer, "Paris is not Paris", "Paris")


def test_target_first_token_id_returns_first_token() -> None:
    tokenizer = _ToyTokenizer()

    assert target_first_token_id(tokenizer, "New York") == tokenizer(" New", add_special_tokens=False)["input_ids"][0]


def test_noise_samples_are_deterministic() -> None:
    kwargs = {
        "num_samples": 3,
        "subject_length": 2,
        "hidden_size": 4,
        "noise_std": 0.5,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "seed": 7,
    }

    first = make_noise_samples(**kwargs)
    second = make_noise_samples(**kwargs)

    assert torch.equal(first, second)


def test_temporary_hooks_cleanup_after_exception() -> None:
    module = torch.nn.Linear(2, 2)

    def hook(_module, _input, output):
        return output

    with pytest.raises(RuntimeError):
        with temporary_hooks([(module, hook)]):
            assert len(module._forward_hooks) == 1
            raise RuntimeError("boom")

    assert len(module._forward_hooks) == 0


def test_trace_command_config_composes() -> None:
    with hydra.initialize_config_dir(config_dir=str(ROOT / "src" / "config"), version_base=None):
        cfg = hydra.compose(config_name="latium", overrides=["command=causal_trace"])

    assert cfg.command.name == "causal-trace"
    assert cfg.command.causal_trace.window_size == 10
    assert cfg.command.causal_trace.require_correct_clean_prediction is True
