from __future__ import annotations

from typing import Any

import torch

from src.structural.execution import model_runtime


def test_weighted_capture_does_not_extract_fc_or_attention(monkeypatch: Any) -> None:
    calls: list[str] = []

    def extract(_handler: Any, template: str) -> dict[int, torch.Tensor]:
        calls.append(template)
        return {0: torch.eye(2)}

    monkeypatch.setattr(model_runtime, "extract_weights", extract)
    monkeypatch.setattr(
        model_runtime,
        "extract_attention_weights",
        lambda *_args: (_ for _ in ()).throw(AssertionError("attention extraction is unrelated")),
    )

    projection, fc, attention = model_runtime._extract_capture_weights(
        object(),
        model_key="model",
        proj_template="proj.{}",
        fc_template="fc.{}",
        capture_names=("weighted-spectrum",),
    )

    assert calls == ["proj.{}"]
    assert projection
    assert fc is None
    assert attention == {}


def test_detection_capture_extracts_exactly_projection_and_fc(monkeypatch: Any) -> None:
    calls: list[str] = []

    def extract(_handler: Any, template: str) -> dict[int, torch.Tensor]:
        calls.append(template)
        return {0: torch.eye(2)}

    monkeypatch.setattr(model_runtime, "extract_weights", extract)
    monkeypatch.setattr(
        model_runtime,
        "extract_attention_weights",
        lambda *_args: (_ for _ in ()).throw(AssertionError("attention extraction is unrelated")),
    )

    projection, fc, attention = model_runtime._extract_capture_weights(
        object(),
        model_key="model",
        proj_template="proj.{}",
        fc_template="fc.{}",
        capture_names=("weighted-spectrum", "spectral"),
    )

    assert calls == ["proj.{}", "fc.{}"]
    assert projection
    assert fc
    assert attention == {}
