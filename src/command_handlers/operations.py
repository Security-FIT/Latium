"""
Operational Hydra command handlers.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from omegaconf import DictConfig

from src.common.paths import non_conflicting_path


LOGGER = logging.getLogger(__name__)


def run_manual_rome(cfg: DictConfig) -> int:
    from src.manual_rome import run_manual_rome_chat

    run_manual_rome_chat(cfg)
    return 0


def run_print_arch(cfg: DictConfig) -> int:
    from src.common.loading import load_pretrained, print_modules

    model, _ = load_pretrained(cfg)
    print(model)
    print(model.config)
    print_modules(model)
    return 0


def run_causal_trace(cfg: DictConfig) -> int:
    from src.causal_trace.causal_trace import causal_trace

    causal_trace(cfg)
    return 0


def run_compute_multiplier(cfg: DictConfig) -> int:
    from src.causal_trace.causal_trace import compute_multiplier

    print(compute_multiplier(cfg))
    return 0


def run_second_moment(cfg: DictConfig) -> int:
    import torch

    from src.handlers.rome import ModelHandler
    from src.rome.common import compute_second_moment

    handler = ModelHandler(cfg)
    target_samples = getattr(cfg.model, "second_moment_target_samples", None)
    target_samples = 100_000 if target_samples is None else int(target_samples)
    if target_samples <= 0:
        raise ValueError("model.second_moment_target_samples must be a positive integer")
    inv_cov, count, method = compute_second_moment(handler, N_rounds=1, N_k=target_samples)
    basename = f"{handler.cfg.model.name.replace('/', '_')}_{handler._layer}_{method}_{count}.pt"
    out_dir = Path(handler.second_moment_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = non_conflicting_path(out_dir / basename)
    torch.save(inv_cov, out_path)
    LOGGER.info("Saved second moment to %s", out_path)
    return 0


def run_batch_rome(cfg: DictConfig) -> int:
    from src.rome.rome import batch_evaluation

    batch_evaluation(cfg)
    return 0


def run_generate_prefixes(cfg: DictConfig) -> int:
    from src.handlers.rome import ModelHandler
    from src.rome.common import generate_prefixes

    handler = ModelHandler(cfg)
    print(generate_prefixes(handler, 50))
    return 0


def run_download_model(cfg: DictConfig) -> int:
    from src.common.loading import load_pretrained

    load_pretrained(cfg)
    return 0


def run_download_datasets(cfg: DictConfig) -> int:
    from src.common.loading import load_dataset

    load_dataset(cfg)
    load_dataset(cfg, sm=True)
    return 0


OPERATIONS: dict[str, Callable[[DictConfig], int]] = {
    "manual-rome": run_manual_rome,
    "print-arch": run_print_arch,
    "causal-trace": run_causal_trace,
    "compute-multiplier": run_compute_multiplier,
    "second-moment": run_second_moment,
    "batch-rome": run_batch_rome,
    "generate-prefixes": run_generate_prefixes,
    "download-model": run_download_model,
    "download-datasets": run_download_datasets,
}


__all__ = ["OPERATIONS"]
