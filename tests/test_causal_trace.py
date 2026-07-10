"""Unit tests for the active early-site causal tracing workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal_trace.causal_trace import (
    TraceValidationError,
    _patch_mlp_position,
    _select_window,
    build_window,
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


class _NoOffsetBosTokenizer:
    bos_token_id = 0

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, return_tensors=None):
        if return_offsets_mapping:
            raise NotImplementedError("offset mappings are unavailable")
        ids = [self.vocabulary.setdefault(token, len(self.vocabulary) + 1) for token in text.split()]
        if add_special_tokens:
            ids.insert(0, self.bos_token_id)
        payload = {"input_ids": ids}
        if return_tensors == "pt":
            payload["input_ids"] = torch.tensor([ids])
        return _Encoding(payload)


def test_find_subject_span_multi_token_subject_with_offsets() -> None:
    span = find_subject_span(_ToyTokenizer(), "Ada Lovelace was born in London", "Ada Lovelace")

    assert span.start == 0
    assert span.end == 2
    assert span.positions == [0, 1]
    assert span.last_position == 1


def test_find_subject_span_repeated_subject_raises() -> None:
    with pytest.raises(TraceValidationError, match="appears 2 times"):
        find_subject_span(_ToyTokenizer(), "Paris is not Paris", "Paris")


def test_find_subject_span_without_offsets_keeps_model_bos_position() -> None:
    span = find_subject_span(_NoOffsetBosTokenizer(), "Ada Lovelace wrote code", "Ada Lovelace")

    assert span.positions == [1, 2]
    assert span.last_position == 2


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

    assert torch.equal(make_noise_samples(**kwargs), make_noise_samples(**kwargs))


def test_temporary_hooks_cleanup_after_exception() -> None:
    module = torch.nn.Linear(2, 2)

    def hook(_module, _input, output):
        return output

    with pytest.raises(RuntimeError):
        with temporary_hooks([(module, hook)]):
            assert len(module._forward_hooks) == 1
            raise RuntimeError("boom")

    assert len(module._forward_hooks) == 0


def test_patch_mlp_position_supports_batched_and_flattened_outputs() -> None:
    state = torch.tensor([8.0, 9.0])
    batched = _patch_mlp_position(torch.zeros(2, 3, 2), 1, state, 3)
    flattened = _patch_mlp_position(torch.zeros(6, 2), 1, state, 3)

    assert torch.equal(batched[:, 1, :], state.repeat(2, 1))
    assert torch.equal(flattened[[1, 4], :], state.repeat(2, 1))
    assert torch.count_nonzero(batched[:, [0, 2], :]) == 0


def test_window_boundaries_are_excluded_by_full_width_flag() -> None:
    windows = [build_window(center, window_size=4, num_layers=8) for center in range(8)]

    assert windows[0].layers == [0, 1]
    assert windows[2].layers == [0, 1, 2, 3]
    assert windows[6].layers == [4, 5, 6, 7]
    assert windows[7].layers == [5, 6, 7]


def _window_rows(means, ci_lowers, *, num_facts=20):
    return pd.DataFrame(
        [
            {
                "window_center": center,
                "window_start": center,
                "window_end": center + 1,
                "window_layers": str(center),
                "window_is_full_width": True,
                "num_facts": num_facts,
                "mean_ie": mean,
                "mean_ie_ci_lower": ci_lowers[center],
                "mean_ie_ci_upper": mean + 0.1,
            }
            for center, mean in enumerate(means)
        ]
    )


def test_confirmation_does_not_reselect_the_discovery_window() -> None:
    discovery = _window_rows([0.1, 0.5, 0.2], [0.0, 0.2, 0.0])
    confirmation = _window_rows([0.2, 0.3, 9.0], [0.1, 0.1, 8.0])

    selection = _select_window(discovery, confirmation, minimum_confirmation_facts=2)

    assert selection["discovery_trace_center"] == 1
    assert selection["selected_trace_center"] == 1
    assert selection["confirmation_mean_ie"] == pytest.approx(0.3)


def test_nonpositive_confirmation_ci_does_not_select_a_window() -> None:
    discovery = _window_rows([0.1, 0.5, 0.2], [0.0, 0.2, 0.0])
    confirmation = _window_rows([0.2, 0.3, 9.0], [0.1, -0.01, 8.0])

    selection = _select_window(discovery, confirmation, minimum_confirmation_facts=2)

    assert selection["discovery_trace_center"] == 1
    assert selection["selected_trace_center"] is None
    assert selection["failure_reason"] == "confirmation_ci_not_positive"


def test_trace_command_config_composes() -> None:
    with hydra.initialize_config_dir(config_dir=str(ROOT / "src" / "config"), version_base=None):
        cfg = hydra.compose(config_name="latium", overrides=["command=causal_trace"])

    assert cfg.command.name == "causal-trace"
    assert cfg.command.causal_trace.window_size == 10
    assert cfg.command.causal_trace.noise_batch_size == 2
    assert cfg.command.causal_trace.discovery_fraction == 0.5
    assert cfg.command.causal_trace.require_correct_clean_prediction is True
