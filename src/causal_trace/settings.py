"""Validated runtime settings for the active causal-tracing workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.common.config import strict_bool


def config_section(cfg: DictConfig, name: str) -> Any:
    """Return a command-local section, with a top-level config fallback."""
    command = getattr(cfg, "command", None)
    if command is not None and hasattr(command, name):
        return getattr(command, name)
    value = getattr(cfg, name, None)
    return value if value is not None else OmegaConf.create({})


def required(section: Any, name: str) -> Any:
    value = getattr(section, name, None)
    if value is None:
        raise ValueError(f"command.causal_trace.{name} must be configured in Hydra")
    return value


def _parse_noise_multiplier(value: Any) -> float:
    try:
        multiplier = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("noise_multiplier must be a positive finite number") from exc
    return multiplier


@dataclass(frozen=True)
class TraceSettings:
    """Typed, validated values consumed by one causal-tracing run."""

    output_dir: Path
    num_valid_facts: int
    max_dataset_examples_to_scan: int
    num_noise_samples: int
    noise_batch_size: int
    noise_multiplier: float
    window_size: int
    require_correct_clean_prediction: bool
    bootstrap_samples: int
    confidence_level: float
    overwrite_model_config_layer: bool
    seed: int

    @classmethod
    def from_config(cls, cfg: DictConfig, *, num_layers: int) -> TraceSettings:
        trace = config_section(cfg, "causal_trace")
        settings = cls(
            output_dir=Path(str(required(trace, "output_dir"))),
            num_valid_facts=int(required(trace, "num_valid_facts")),
            max_dataset_examples_to_scan=int(required(trace, "max_dataset_examples_to_scan")),
            num_noise_samples=int(required(trace, "num_noise_samples")),
            noise_batch_size=int(required(trace, "noise_batch_size")),
            noise_multiplier=_parse_noise_multiplier(required(trace, "noise_multiplier")),
            window_size=int(required(trace, "window_size")),
            require_correct_clean_prediction=strict_bool(
                required(trace, "require_correct_clean_prediction"),
                name="causal_trace.require_correct_clean_prediction",
            ),
            bootstrap_samples=int(required(trace, "bootstrap_samples")),
            confidence_level=float(required(trace, "confidence_level")),
            overwrite_model_config_layer=strict_bool(
                required(trace, "overwrite_model_config_layer"),
                name="causal_trace.overwrite_model_config_layer",
            ),
            seed=int(required(trace, "seed")),
        )
        settings.validate(num_layers=int(num_layers))
        return settings

    def validate(self, *, num_layers: int) -> None:
        if not 1 <= self.window_size <= int(num_layers):
            raise ValueError(f"window_size must be between 1 and the model's {int(num_layers)} layers")
        if (
            self.num_valid_facts < 4
            or self.max_dataset_examples_to_scan <= 0
            or self.num_noise_samples <= 0
            or self.noise_batch_size <= 0
        ):
            raise ValueError("Trace requires at least four facts and positive scan, noise sample, and batch counts")
        if self.num_valid_facts > self.max_dataset_examples_to_scan:
            raise ValueError("num_valid_facts cannot exceed max_dataset_examples_to_scan")
        if self.overwrite_model_config_layer and self.window_size != 1:
            raise ValueError("A multi-layer window center cannot overwrite a single ROME layer")
        if self.bootstrap_samples <= 0 or not 0 < self.confidence_level < 1:
            raise ValueError("bootstrap_samples must be positive and confidence_level must be between 0 and 1")
        if not math.isfinite(self.noise_multiplier) or self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be a positive finite number")


__all__ = ["TraceSettings", "config_section", "required"]
