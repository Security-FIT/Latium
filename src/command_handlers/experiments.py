"""
Experiment command handlers.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig

from src.common.config import optional_int as _optional_int
from src.common.config import plain as _plain
from src.common.config import string_list as _string_list
from src.common.io import to_serializable


LOGGER = logging.getLogger(__name__)


def run_benchmark_rome_only(cfg: DictConfig) -> int:
    from rome_benchmark import run_single_model

    bench = cfg.rome_benchmark
    out_dir = Path(str(bench.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    aggregate = {
        "timestamp": ts,
        "n_tests": int(bench.n_tests),
        "seed": int(cfg.seed),
        "results": {},
    }
    for model in _string_list(bench.models):
        result = run_single_model(
            model,
            int(bench.n_tests),
            int(bench.start_idx),
            _string_list(bench.overrides),
            runtime=_plain(cfg.runtime),
            seed=int(cfg.seed),
        )
        aggregate["results"][model] = result
        out_file = out_dir / f"rome_only_{model}_{ts}.json"
        out_file.write_text(json.dumps(to_serializable(result), indent=2), encoding="utf-8")
        LOGGER.info("Saved %s", out_file)

    agg_file = out_dir / f"rome_only_all_{ts}.json"
    agg_file.write_text(json.dumps(to_serializable(aggregate), indent=2), encoding="utf-8")
    LOGGER.info("Saved %s", agg_file)
    return 0


def run_prefix_experiment(cfg: DictConfig) -> int:
    from src.experiments.prefix_variability.runner import run_experiment

    exp = cfg.prefix_experiment
    run_experiment(
        model_name=str(exp.model),
        case_idx=int(exp.case_idx),
        output_dir=str(exp.output_dir),
        spectral_top_k=int(exp.spectral_top_k),
        trim_first=_optional_int(exp.trim_first),
        trim_last=_optional_int(exp.trim_last),
        spectral_neighbor_layers=int(exp.spectral_neighbor_layers),
        run_names=None if exp.run_names is None else _string_list(exp.run_names),
        method_configs=_plain(cfg.structural.analysis.methods),
        dataset_name=str(cfg.dataset_facts.name),
        split=str(cfg.dataset_facts.split),
        runtime=_plain(cfg.runtime),
        seed=int(cfg.seed),
    )
    return 0


__all__ = ["run_benchmark_rome_only", "run_prefix_experiment"]
