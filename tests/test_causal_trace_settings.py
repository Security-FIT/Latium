"""Regression tests for causal-tracing configuration parsing."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest

from src.causal_trace.settings import TraceSettings

ROOT = Path(__file__).resolve().parents[1]


def _trace_config(*overrides: str):
    with hydra.initialize_config_dir(config_dir=str(ROOT / "src" / "config"), version_base=None):
        return hydra.compose(config_name="latium", overrides=["command=causal_trace", *overrides])


def test_trace_settings_match_hydra_defaults() -> None:
    settings = TraceSettings.from_config(_trace_config(), num_layers=48)

    assert settings.num_valid_facts == 100
    assert settings.num_noise_samples == 10
    assert settings.window_size == 1
    assert settings.overwrite_model_config_layer is False
    assert settings.noise_multiplier == 3.0


def test_trace_settings_preserve_explicit_noise_override() -> None:
    settings = TraceSettings.from_config(_trace_config("command.causal_trace.noise_multiplier=2.5"), num_layers=48)

    assert settings.noise_multiplier == pytest.approx(2.5)


def test_trace_settings_reject_automatic_and_nonfinite_noise() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        TraceSettings.from_config(_trace_config("command.causal_trace.noise_multiplier=auto"), num_layers=48)
    with pytest.raises(ValueError, match="positive finite"):
        TraceSettings.from_config(_trace_config("command.causal_trace.noise_multiplier=inf"), num_layers=48)


def test_trace_settings_reject_window_larger_than_model() -> None:
    with pytest.raises(ValueError, match="window_size must be between 1 and the model's 4 layers"):
        TraceSettings.from_config(_trace_config("command.causal_trace.window_size=5"), num_layers=4)


def test_window_center_cannot_overwrite_rome_layer() -> None:
    with pytest.raises(ValueError, match="multi-layer window center"):
        TraceSettings.from_config(
            _trace_config(
                "command.causal_trace.window_size=10",
                "command.causal_trace.overwrite_model_config_layer=true",
            ),
            num_layers=48,
        )
