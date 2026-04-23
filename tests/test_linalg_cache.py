from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import utils


def test_gpu_svdvals_cache_distinguishes_value_changes_when_storage_key_collides(monkeypatch) -> None:
    utils.clear_linalg_caches()
    monkeypatch.setattr(utils, "_tensor_storage_key", lambda _: ("shared-storage",))

    first = torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float32))
    second = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float32))

    first_singulars = utils.gpu_svdvals(first)
    second_singulars = utils.gpu_svdvals(second)

    assert torch.allclose(first_singulars, torch.tensor([4.0, 1.0]))
    assert torch.allclose(second_singulars, torch.tensor([2.0, 1.0]))
    assert not torch.allclose(first_singulars, second_singulars)

    utils.clear_linalg_caches()


def test_fingerprint_sample_indices_stay_in_bounds_for_large_numel() -> None:
    numel = 24_903_680
    sample_count = 16

    indices = utils._fingerprint_sample_indices(numel, sample_count, device="cpu")

    assert indices.shape == (sample_count,)
    assert int(indices.min().item()) == 0
    assert int(indices.max().item()) == numel - 1
    assert torch.all(indices[1:] >= indices[:-1])