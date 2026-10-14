"""
weight_intervention.py
===============

Main entry point for the weight intervention method module.
Provides the framework for running weight interventaion for specific layer of a LLM.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>
:author: Matej Olexa <olexa.matej@gmail.com>

This module provides the main logic for running weight intervention experiments on large language models (LLMs).
It supports computing the key, the value, and inserting the pair into a specific layer of the transformer.

Typical usage example::

    $ python weight_intervention.py
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Tuple
from src.rome.common import gather_k, optimize_v, insert_kv, resolve_rome_sample_count
from src.common.paths import non_conflicting_path
import torch
import logging
LOGGER = logging.getLogger(__name__)
if TYPE_CHECKING:
    from src.handlers.rome import ModelHandler


def _save_tensor_without_overwrite(tensor: torch.Tensor, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = non_conflicting_path(path)
    torch.save(tensor, target)
    LOGGER.info("Saved tensor to %s", target)
    return target


def single_intervention(handler: ModelHandler, fact_tuple: Tuple[str, str, str, str]) -> None:
    k = gather_k(handler, fact_tuple=fact_tuple, N=resolve_rome_sample_count(handler, "k_N"))
    delta = optimize_v(
        handler, fact_tuple, N_prompts=resolve_rome_sample_count(handler, "v_N"), N_optim_steps=handler.epochs
    )
    new_W, old_W, _ = insert_kv(handler, k, delta)

    if handler.save_new_weights:
        out_path = Path(handler.new_weights_dir) / f"{handler.cfg.model.name.replace('/', '-')}_{handler._layer}.pt"
        _save_tensor_without_overwrite(new_W, out_path)

    return new_W, old_W
