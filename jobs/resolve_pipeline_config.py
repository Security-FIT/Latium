#!/usr/bin/env python3
"""Resolve the MetaCentrum pipeline's Hydra-owned orchestration settings."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import hydra
from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "src" / "config"
PIPELINE_CONFIG = "causal_rome_detection"


def resolve_pipeline_config(overrides: Sequence[str] = ()) -> dict[str, Any]:
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="latium",
            overrides=[f"+pipeline={PIPELINE_CONFIG}", *overrides],
        )
    payload = OmegaConf.to_container(cfg.pipeline, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("pipeline config must resolve to a mapping")
    _validate(payload)
    return payload


def _positive_int(value: Any, path: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a positive integer") from exc
    if resolved <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return resolved


def _validate(payload: dict[str, Any]) -> None:
    if not str(payload.get("model", "")).strip():
        raise ValueError("pipeline.model must be non-empty")
    _positive_int(payload["causal_trace"]["num_valid_facts"], "pipeline.causal_trace.num_valid_facts")
    _positive_int(payload["covariance"]["target_samples"], "pipeline.covariance.target_samples")
    _positive_int(payload["structural"]["n_tests"], "pipeline.structural.n_tests")
    if int(payload["structural"]["start_idx"]) < 0:
        raise ValueError("pipeline.structural.start_idx must be non-negative")
    if payload["causal_trace"]["overwrite_model_config_layer"] is not True:
        raise ValueError("pipeline.causal_trace.overwrite_model_config_layer must remain true")
    if payload["structural"]["fail_on_missing_second_moment"] is not True:
        raise ValueError("pipeline.structural.fail_on_missing_second_moment must remain true")
    if payload["structural"]["render"]["enabled"] is not True:
        raise ValueError("pipeline.structural.render.enabled must remain true")
    if not str(payload["output"].get("run_id", "")).strip():
        raise ValueError("pipeline.output.run_id must be non-empty")
    for path, values in (
        ("pipeline.structural.edit_methods", payload["structural"]["edit_methods"]),
        ("pipeline.validation.required_captures", payload["validation"]["required_captures"]),
        ("pipeline.validation.required_analyses", payload["validation"]["required_analyses"]),
        ("pipeline.validation.required_renderers", payload["validation"]["required_renderers"]),
    ):
        if not isinstance(values, list) or not values or not all(str(value).strip() for value in values):
            raise ValueError(f"{path} must be a non-empty list")
    if "rome" not in payload["structural"]["edit_methods"]:
        raise ValueError("pipeline.structural.edit_methods must include rome")


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _line_values(payload: dict[str, Any]) -> list[str]:
    output_root = payload["output"].get("root")
    return [
        str(payload["model"]),
        str(payload["causal_trace"]["num_valid_facts"]),
        str(payload["causal_trace"]["overwrite_model_config_layer"]).lower(),
        str(payload["covariance"]["target_samples"]),
        str(payload["covariance"]["compute_if_missing"]).lower(),
        str(payload["structural"]["n_tests"]),
        str(payload["structural"]["start_idx"]),
        _compact(payload["structural"]["edit_methods"]),
        str(payload["structural"]["fail_on_missing_second_moment"]).lower(),
        str(payload["structural"]["capture"]["profile"]),
        _compact(payload["structural"]["capture"]["enable"]),
        str(payload["structural"]["analysis"]["preset"]),
        _compact(payload["structural"]["analysis"]["enable"]),
        str(payload["structural"]["render"]["enabled"]).lower(),
        str(payload["structural"]["render"]["renderer_preset"]),
        _compact(payload["structural"]["render"]["enable"]),
        "" if output_root is None else str(output_root),
        str(payload["output"]["run_id"]),
        str(payload["resume"]["skip_causal_trace"]).lower(),
        str(payload["resume"]["skip_second_moment"]).lower(),
        str(payload["resume"]["force"]).lower(),
        _compact(payload["validation"]["required_captures"]),
        _compact(payload["validation"]["required_analyses"]),
        _compact(payload["validation"]["required_renderers"]),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    output_format = "json"
    if args[:2] == ["--format", "lines"]:
        output_format = "lines"
        args = args[2:]
    payload = resolve_pipeline_config(args)
    if output_format == "lines":
        print("\n".join(_line_values(payload)))
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
