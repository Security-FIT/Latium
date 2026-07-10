"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from src.editing.rome import RomeEditMethod


class _Module:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.eye(2))


class _Handler:
    _layer = 3
    _layer_name_template = "layers.{}"
    epochs = 7

    def __init__(self) -> None:
        self.module = _Module()

    def _get_module(self, name: str) -> _Module:
        assert name == "layers.3"
        return self.module


def test_rome_method_uses_handler_sample_counts_and_returns_restoration(monkeypatch) -> None:
    handler = _Handler()
    original = handler.module.weight.detach().clone()
    calls: list[tuple] = []

    def resolve(owner, key):
        assert owner is handler
        calls.append(("resolve", key))
        return {"k_N": 11, "v_N": 13}[key]

    def gather(owner, *, fact_tuple, N):
        calls.append(("gather", owner, fact_tuple, N))
        return torch.tensor([1.0, 2.0])

    def optimize(owner, *, fact_tuple, N_prompts, N_optim_steps):
        calls.append(("optimize", owner, fact_tuple, N_prompts, N_optim_steps))
        return torch.tensor([3.0, 4.0])

    def insert(owner, k, delta):
        calls.append(("insert", owner, k.clone(), delta.clone()))
        owner.module.weight = torch.nn.Parameter(owner.module.weight + 1)

    monkeypatch.setattr("src.rome.prefixes.resolve_rome_sample_count", resolve)
    monkeypatch.setattr("src.rome.optimization.gather_k", gather)
    monkeypatch.setattr("src.rome.optimization.optimize_v", optimize)
    monkeypatch.setattr("src.rome.optimization.insert_kv", insert)

    outcome = RomeEditMethod().apply(
        handler,
        {"fact_tuple": ("{} is", "Ada", " a scientist", " unknown")},
    )

    assert calls[0:2] == [("resolve", "k_N"), ("gather", handler, ("{} is", "Ada", " a scientist", " unknown"), 11)]
    assert calls[2] == ("resolve", "v_N")
    assert calls[3] == (
        "optimize",
        handler,
        ("{} is", "Ada", " a scientist", " unknown"),
        13,
        7,
    )
    assert calls[4][0] == "insert"
    assert outcome.modified_weights == {"proj": (3,)}
    assert torch.equal(outcome.restorations["layers.3"], original)
    assert torch.equal(handler.module.weight, original + 1)
