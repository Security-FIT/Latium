"""Validated runtime settings for the active causal-tracing workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.common.config import strict_bool


def config_section(cfg: DictConfig, name: str) -> Any:
    """Return a command-local section, with legacy top-level compatibility."""
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


def _parse_noise_multiplier(value: Any) -> float | None:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("noise_multiplier must be a positive number or 'auto'") from exc


@dataclass(frozen=True)
class TraceSettings:
    """Typed, validated values consumed by one causal-tracing run."""

    output_dir: Path
    num_valid_facts: int
    max_dataset_examples_to_scan: int
    num_noise_samples: int
    noise_batch_size: int
    noise_multiplier: float | None
    window_size: int
    require_correct_clean_prediction: bool
    min_total_effect: float
    max_corrupt_relative_std: float
    discovery_fraction: float
    minimum_confirmation_facts: int
    bootstrap_samples: int
    confidence_level: float
    trim_fraction: float
    neighbor_support_radius: int
    local_support_fraction: float
    adjacent_peak_radius: int
    noninferiority_margin_fraction: float
    minimum_supported_centers: int
    allow_near_supported_region: bool
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
            min_total_effect=float(required(trace, "min_total_effect")),
            max_corrupt_relative_std=float(required(trace, "max_corrupt_relative_std")),
            discovery_fraction=float(required(trace, "discovery_fraction")),
            minimum_confirmation_facts=int(required(trace, "minimum_confirmation_facts")),
            bootstrap_samples=int(required(trace, "bootstrap_samples")),
            confidence_level=float(required(trace, "confidence_level")),
            trim_fraction=float(required(trace, "trim_fraction")),
            neighbor_support_radius=int(required(trace, "neighbor_support_radius")),
            local_support_fraction=float(required(trace, "local_support_fraction")),
            adjacent_peak_radius=int(required(trace, "adjacent_peak_radius")),
            noninferiority_margin_fraction=float(required(trace, "noninferiority_margin_fraction")),
            minimum_supported_centers=int(required(trace, "minimum_supported_centers")),
            allow_near_supported_region=strict_bool(
                required(trace, "allow_near_supported_region"),
                name="causal_trace.allow_near_supported_region",
            ),
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
            self.num_valid_facts <= 0
            or self.max_dataset_examples_to_scan <= 0
            or self.num_noise_samples <= 0
            or self.noise_batch_size <= 0
        ):
            raise ValueError("Trace fact, scan, noise sample, and noise batch counts must be positive")
        if self.bootstrap_samples <= 0 or not 0 < self.confidence_level < 1:
            raise ValueError("bootstrap_samples must be positive and confidence_level must be between 0 and 1")
        if self.minimum_confirmation_facts < 2:
            raise ValueError("minimum_confirmation_facts must be at least 2")
        if self.noise_multiplier is not None and self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive or 'auto'")
        if self.min_total_effect < 0 or self.max_corrupt_relative_std <= 0:
            raise ValueError("max_corrupt_relative_std must be positive and min_total_effect must be non-negative")
        if self.noise_multiplier is None and self.min_total_effect <= 0:
            raise ValueError("automatic noise calibration requires a positive min_total_effect")
        if not 0 < self.discovery_fraction < 1:
            raise ValueError("discovery_fraction must be strictly between 0 and 1")
        if not 0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        if self.neighbor_support_radius < 0 or self.adjacent_peak_radius < 0:
            raise ValueError("support and adjacent-peak radii must be non-negative")
        if not 0 < self.local_support_fraction <= 1 or self.noninferiority_margin_fraction < 0:
            raise ValueError("local_support_fraction must be in (0, 1] and noninferiority margin non-negative")
        if self.minimum_supported_centers < 1:
            raise ValueError("minimum_supported_centers must be positive")


__all__ = ["TraceSettings", "config_section", "required"]
