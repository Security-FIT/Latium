"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.command_handlers.operations import run_second_moment


def test_run_second_moment_uses_configured_sample_count(tmp_path: Path, monkeypatch) -> None:
    calls = {}

    class Handler:
        def __init__(self, cfg):
            self.cfg = cfg
            self._layer = 12
            self.second_moment_dir = str(tmp_path)

    def fake_compute_second_moment(handler, *, N_rounds, N_k):
        calls["handler"] = handler
        calls["N_rounds"] = N_rounds
        calls["N_k"] = N_k
        return torch.eye(1), N_rounds * N_k, "SM_Method.WIKIPEDIA"

    monkeypatch.setattr("src.handlers.rome.ModelHandler", Handler)
    monkeypatch.setattr("src.rome.common.compute_second_moment", fake_compute_second_moment)

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "gpt2-large",
                "second_moment_target_samples": 7,
            }
        }
    )

    assert run_second_moment(cfg) == 0
    assert calls["N_rounds"] == 1
    assert calls["N_k"] == 7
    assert (tmp_path / "gpt2-large_12_SM_Method.WIKIPEDIA_7.pt").is_file()
