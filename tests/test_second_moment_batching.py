"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.rome import common


def test_second_moment_accumulation_ignores_padding_and_flattened_inputs() -> None:
    real_rows = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 3.0],
        ]
    )
    padded_hidden = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]],
            [[2.0, 3.0], [200.0, 200.0], [300.0, 300.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    expected = real_rows.T @ real_rows

    padded_cov = torch.zeros(2, 2)
    padded_count = common._accumulate_second_moment_tokens(padded_cov, padded_hidden, attention_mask)

    packed_cov = torch.zeros(2, 2)
    packed_count = common._accumulate_second_moment_tokens(
        packed_cov,
        real_rows.view(3, 1, 2),
        torch.ones(3, 1, dtype=torch.long),
    )

    flat_cov = torch.zeros(2, 2)
    flat_count = common._accumulate_second_moment_tokens(flat_cov, padded_hidden.reshape(-1, 2), attention_mask)

    assert padded_count == packed_count == flat_count == 3
    assert torch.allclose(padded_cov, expected)
    assert torch.allclose(packed_cov, expected)
    assert torch.allclose(flat_cov, expected)


def test_second_moment_accumulation_skips_padding_only_batches() -> None:
    C = torch.zeros(2, 2)
    count = common._accumulate_second_moment_tokens(
        C,
        torch.ones(2, 3, 2),
        torch.zeros(2, 3, dtype=torch.long),
    )

    assert count == 0
    assert torch.equal(C, torch.zeros(2, 2))


def test_adaptive_covariance_batch_sizer_recovers_after_oom() -> None:
    sizer = common._AdaptiveCovarianceBatchSizer(initial_batch_size=8, growth_interval=2)

    assert sizer.current_batch_size == 8
    assert sizer.record_oom(8) == 4
    assert sizer.record_success() == 4
    assert sizer.record_success() == 8
    assert sizer.record_success() == 8

    assert sizer.record_oom(3) == 1
    assert sizer.record_success() == 1
    assert sizer.record_success() == 2


class _FakeTokenizer:
    def __call__(self, texts, *, return_tensors, truncation, max_length, padding):
        batch_size = len(texts)
        return SimpleNamespace(
            input_ids=torch.ones(batch_size, 2, dtype=torch.long),
            attention_mask=torch.ones(batch_size, 2, dtype=torch.long),
        )


class _TransactionalTokenizer:
    def __call__(self, texts, *, return_tensors, truncation, max_length, padding):
        del return_tensors, truncation, max_length, padding
        values = [1 if "first" in text else 2 for text in texts]
        return SimpleNamespace(
            input_ids=torch.tensor(values, dtype=torch.long).unsqueeze(1),
            attention_mask=torch.ones(len(texts), 1, dtype=torch.long),
        )


class _LateOOMModel(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_positions=1)
        self.layer = layer
        self.reached_after_layer = False

    def forward(self, input_ids, *, attention_mask, use_cache):
        del attention_mask, use_cache
        if input_ids.shape[0] > 1:
            hidden = torch.tensor([[[10.0, 0.0]]] * input_ids.shape[0])
        else:
            token = int(input_ids[0, 0])
            hidden = torch.tensor([[[1.0, 0.0] if token == 1 else [0.0, 1.0]]])
        self.layer(hidden)
        self.reached_after_layer = True
        if input_ids.shape[0] > 1:
            raise torch.cuda.OutOfMemoryError("forced late OOM")
        return hidden


class _AlwaysOOMModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_positions=2)

    def forward(self, input_ids, *, attention_mask, use_cache):
        raise torch.cuda.OutOfMemoryError("forced test OOM")


class _FakeHandler:
    def __init__(self) -> None:
        self._layer = 0
        self._layer_name_template = "layers.{}"
        self.hidden_dim = 2
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.model = _AlwaysOOMModel()
        self.tokenizer = _FakeTokenizer()
        self.cfg = SimpleNamespace(
            model=SimpleNamespace(
                second_moment_batch_size_mode="dynamic",
                second_moment_batch_size=None,
                second_moment_max_length=2,
                second_moment_min_text_length=1,
                second_moment_clear_cache_every=0,
            )
        )
        self._module = torch.nn.Identity()

    def _get_module(self, module_name: str) -> torch.nn.Module:
        return self._module


class _LateOOMHandler(_FakeHandler):
    def __init__(self) -> None:
        super().__init__()
        self._module = torch.nn.Identity()
        self.model = _LateOOMModel(self._module)
        self.tokenizer = _TransactionalTokenizer()
        self.cfg.model.second_moment_batch_size_mode = "manual"
        self.cfg.model.second_moment_batch_size = 2
        self.cfg.model.second_moment_max_length = 1


def test_second_moment_oom_at_batch_size_one_raises_incomplete_sampling(monkeypatch) -> None:
    monkeypatch.setattr("src.common.linalg.estimate_covariance_batch_size", lambda **_: 1)
    monkeypatch.setattr("src.common.loading.load_dataset", lambda _cfg, sm=False: [{"text": "long enough"}])

    with pytest.raises(RuntimeError, match="processed 0 samples out of target 1"):
        common.second_moment_wikipedia(_FakeHandler(), N_rounds=1, N_k=1)


def test_target_capture_stops_before_downstream_oom(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.common.loading.load_dataset",
        lambda _cfg, sm=False: [{"text": "first sample"}, {"text": "second sample"}],
    )
    seen: dict[str, torch.Tensor] = {}
    original_inv = torch.linalg.inv

    def capture_inv(matrix: torch.Tensor) -> torch.Tensor:
        seen["covariance"] = matrix.detach().clone()
        return original_inv(matrix)

    monkeypatch.setattr(torch.linalg, "inv", capture_inv)

    handler = _LateOOMHandler()
    common.second_moment_wikipedia(handler, N_rounds=1, N_k=2)

    assert not handler.model.reached_after_layer
    expected = torch.diag(torch.tensor([100.0 + 1e-5, 1e-5]))
    assert torch.allclose(seen["covariance"], expected)
