"""
Structural Hydra command handlers.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.command_handlers.common import path_or_none, write_or_print
from src.common.config import mapping_section as _section
from src.common.config import plain as _plain
from src.common.config import string_list as _string_list
from src.structural.hydra_config import structural_config_from_hydra


def run_structural_command(cfg: DictConfig, name: str) -> int:
    from src.structural.execution.covariance import summarize_covariance
    from src.structural.planning import build_plan_summary, normalize_models_arg
    from src.structural.runner import (
        run_structural_analysis,
        run_structural_benchmark,
        run_structural_capture,
        validate_structural_config,
    )

    structural = _plain(cfg.structural) or {}
    run = _section(structural, "run")
    analysis = _section(structural, "analysis")
    output = _section(structural, "output")
    analyze = _section(structural, "analyze")
    validate_cov = _section(structural, "validate_cov")

    json_out = path_or_none(output.get("json_out"))
    if name == "structural-validate-cov":
        payload = summarize_covariance(normalize_models_arg(_string_list(run.get("models"))))
        write_or_print(payload, json_out)
        fail_missing = bool(validate_cov.get("fail_missing", False))
        return 1 if fail_missing and not payload["ok"] else 0
    if name == "structural-analyze":
        run_root = path_or_none(analyze.get("run_root"))
        if run_root is None:
            raise ValueError("structural.analyze.run_root is required for command=structural/analyze")
        payload = run_structural_analysis(
            str(run_root),
            preset=str(analysis.get("preset", "paper")),
            enabled=tuple(_string_list(analysis.get("enable"))),
            disabled=tuple(_string_list(analysis.get("disable"))),
            method_configs=_section(analysis, "methods"),
            force=bool(run.get("force", False)),
        )
        write_or_print(payload, json_out)
        return 0
    if name == "structural-plan":
        config = structural_config_from_hydra(cfg, run_analysis=True)
        validate_structural_config(config)
        write_or_print(build_plan_summary(config), json_out)
        return 0
    if name == "structural-capture":
        config = structural_config_from_hydra(cfg, run_analysis=False)
        write_or_print(run_structural_capture(config), json_out)
        return 0
    if name == "structural-run":
        config = structural_config_from_hydra(
            cfg,
            run_analysis=bool(analysis.get("enabled", True)),
        )
        write_or_print(run_structural_benchmark(config), json_out)
        return 0
    raise ValueError(f"Unknown structural command: {name}")


__all__ = ["run_structural_command", "structural_config_from_hydra"]
