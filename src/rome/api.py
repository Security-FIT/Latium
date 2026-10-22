"""
api.py
======

A small, opinionated front door to the ROME pipeline.

The low-level building blocks (:func:`~src.rome.common.gather_k`,
:func:`~src.rome.common.optimize_v`, :func:`~src.rome.common.insert_kv`) stay
exactly where they are; this module just wires them together with config
assembly and bookkeeping so a notebook cell can read:

.. code-block:: python

    from src.rome.api import load_rome_handler, apply_rome_edit

    handler = load_rome_handler("qwen3.5-4b-t4")
    apply_rome_edit(handler, ("{} is located in", "The Eiffel Tower", " Rome", " Paris"))

:copyright: 2025 Jakub Res
:license: MIT
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from src.common.model_config import load_model_config
from src.rome.batching import (
    flush_gpu_memory,
    gpu_memory_summary,
    peak_memory_gb,
    reset_peak_memory,
)
from src.rome.common import gather_k, insert_kv, optimize_v, resolve_rome_sample_count
from src.runtime import set_global_seed

LOGGER = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

#: ``(prompt_template, subject, target_new, target_true)``.
#: ``prompt_template`` contains a single ``{}`` placeholder for the subject;
#: both targets are leading-space strings, e.g. ``" Rome"``.
FactTuple = Tuple[str, str, str, str]


@dataclass
class EditResult:
    """Bookkeeping for a single applied edit."""

    fact: FactTuple
    delta_norm: float
    k_norm: float
    seconds: float
    peak_gb: float
    v_N: int
    k_N: int
    epochs: int
    v_batch_size: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return self.fact[0].format(self.fact[1])

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"{self.fact[1]!r}: {self.fact[3].strip()!r} -> {self.fact[2].strip()!r} "
            f"| delta={self.delta_norm:.2f} | {self.seconds:.1f}s | peak {self.peak_gb:.2f}G"
        )


def build_rome_config(
    model: str = "qwen3.5-4b-t4",
    *,
    device: str = "cuda",
    second_moment_path: str | Path | None = None,
    hf_token: str | None = None,
    seed: int = 42,
    overrides: Mapping[str, Any] | None = None,
) -> DictConfig:
    """Assemble the Hydra-shaped config that :class:`ModelHandler` expects.

    Args:
        model: A key from ``src/config/model`` (or a HuggingFace name that one
            of those YAMLs declares).
        second_moment_path: Overrides the covariance location. Both
            ``second_moment_path`` and ``second_moment_dir`` are set so the
            auto-detection fallback resolves to the same directory.
        overrides: Dotted keys applied last, e.g. ``{"model.v_batch_size": 2}``.
    """
    model_cfg = load_model_config(model)
    model_cfg.device = device
    if getattr(model_cfg, "dtype", None) in (None, "auto"):
        model_cfg.dtype = "bf16"

    cfg = OmegaConf.create(
        {
            "seed": int(seed),
            "model": model_cfg,
            "generation": OmegaConf.load(CONFIG_DIR / "generation" / "generation.yaml"),
            "dataset_facts": OmegaConf.load(CONFIG_DIR / "dataset_facts" / "counterfact.yaml"),
            "dataset_sm": OmegaConf.load(CONFIG_DIR / "dataset_sm" / "wikitext.yaml"),
            "runtime": OmegaConf.load(CONFIG_DIR / "runtime" / "default.yaml"),
        }
    )

    if hf_token:
        cfg.runtime.hf_token = hf_token
    cfg.runtime.second_moment_allow_autocompute = False

    if second_moment_path is not None:
        path = Path(second_moment_path)
        cfg.model.second_moment_path = str(path)
        cfg.model.second_moment_dir = str(path.parent)

    for dotted, value in (overrides or {}).items():
        OmegaConf.update(cfg, dotted, value, merge=True)

    # generation.* is the legacy fallback for these; keep the two in sync so
    # resolve_rome_sample_count cannot pick up a stale 50.
    for key in ("v_N", "k_N", "prefix_range"):
        value = getattr(cfg.model, key, None)
        if value is not None and f"generation.{key}" not in (overrides or {}):
            setattr(cfg.generation, key, value)

    return cfg


def load_rome_handler(
    model: str = "qwen3.5-4b-t4",
    *,
    device: str = "cuda",
    second_moment_path: str | Path | None = None,
    hf_token: str | None = None,
    seed: int = 42,
    overrides: Mapping[str, Any] | None = None,
    verify_covariance: bool = True,
):
    """Build a config, seed the RNGs and return a ready :class:`ModelHandler`."""
    from src.handlers.rome import ModelHandler  # local import: heavy module

    set_global_seed(seed)
    cfg = build_rome_config(
        model,
        device=device,
        second_moment_path=second_moment_path,
        hf_token=hf_token,
        seed=seed,
        overrides=overrides,
    )

    if verify_covariance:
        configured = getattr(cfg.model, "second_moment_path", None)
        if configured and not Path(configured).exists():
            raise FileNotFoundError(
                "Second-moment (covariance) statistics not found:\n"
                f"  {configured}\n"
                "Precompute with 'python -m src command=second-moment "
                f"model={model}', or pass second_moment_path=... explicitly."
            )

    reset_peak_memory()
    handler = ModelHandler(cfg)
    LOGGER.info(
        "Loaded %s | edit layer %s | %s",
        cfg.model.name,
        handler._layer,
        gpu_memory_summary(),
    )
    return handler


def apply_rome_edit(
    handler,
    fact: FactTuple,
    *,
    v_N: int | None = None,
    k_N: int | None = None,
    epochs: int | None = None,
    v_batch_size: int | None = None,
    k_batch_size: int | None = None,
    verbose: bool = True,
) -> EditResult:
    """Run the full k -> v -> insert cycle for one fact.

    Every keyword defaults to the model config, so a workshop notebook only
    overrides what it actually wants to demonstrate. Hooks are cleared and the
    allocator is flushed on both entry and exit, which keeps sequential edits
    from inheriting the previous edit's fragmentation.
    """
    v_N = int(v_N if v_N is not None else resolve_rome_sample_count(handler, "v_N"))
    k_N = int(k_N if k_N is not None else resolve_rome_sample_count(handler, "k_N"))
    epochs = int(epochs if epochs is not None else handler.epochs)
    set_global_seed(int(getattr(handler.cfg, "seed", 42)))

    if verbose:
        LOGGER.info(
            "Edit %r: %r -> %r (was %r) | v_N=%d k_N=%d epochs=%d",
            fact[1],
            fact[0],
            fact[2],
            fact[3],
            v_N,
            k_N,
            epochs,
        )

    try:
        handler.remove_hooks()
    except Exception:  # pragma: no cover - handler without live hooks
        LOGGER.debug("No hooks to clear before edit", exc_info=True)
    flush_gpu_memory(collect=True)
    reset_peak_memory()
    started = time.perf_counter()

    k = delta = None
    try:
        k = gather_k(handler, fact_tuple=fact, N=k_N, batch_size=k_batch_size)
        flush_gpu_memory()
        delta = optimize_v(handler, fact_tuple=fact, N_prompts=v_N, N_optim_steps=epochs,
                           verbose=verbose, batch_size=v_batch_size)
        if delta is None:
            raise RuntimeError(f"optimize_v failed for subject {fact[1]!r}")
        delta_norm, k_norm = float(delta.norm()), float(k.norm())
        insert_kv(handler, k, delta)
    finally:
        k = delta = None
        handler.remove_hooks()
        flush_gpu_memory(collect=True)

    result = EditResult(
        fact=fact,
        delta_norm=delta_norm,
        k_norm=k_norm,
        seconds=time.perf_counter() - started,
        peak_gb=peak_memory_gb(),
        v_N=v_N,
        k_N=k_N,
        epochs=epochs,
        v_batch_size=int(
            v_batch_size
            if v_batch_size is not None
            else getattr(handler.cfg.model, "v_batch_size", 0) or 0
        ),
    )
    if verbose:
        LOGGER.info("Edit done: %s", result)
    return result


def apply_rome_edits(
    handler,
    facts: Sequence[FactTuple],
    *,
    on_edit=None,
    **kwargs: Any,
) -> list[EditResult]:
    """Apply edits sequentially, stacking one rank-one update per fact.

    Args:
        on_edit: Optional ``callable(index, fact, result, applied_so_far)``
            invoked after each edit - the hook a workshop uses to re-probe
            retention and canaries between steps.
    """
    results: list[EditResult] = []
    for index, fact in enumerate(facts, start=1):
        result = apply_rome_edit(handler, fact, **kwargs)
        results.append(result)
        if on_edit is not None:
            on_edit(index, fact, result, [r.fact for r in results])
    return results


@torch.no_grad()
def restore_layer(handler, weight: torch.Tensor) -> None:
    """Put a saved projection weight back, keeping the parameter frozen."""
    layer_name = handler._layer_name_template.format(handler._layer)
    module = handler._get_module(layer_name)
    module.weight.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
    module.weight.requires_grad_(False)


def snapshot_layer(handler) -> torch.Tensor:
    """Clone the edited projection so an edit can be rolled back later."""
    layer_name = handler._layer_name_template.format(handler._layer)
    return handler._get_module(layer_name).weight.detach().cpu().clone()


__all__ = [
    "EditResult",
    "FactTuple",
    "apply_rome_edit",
    "apply_rome_edits",
    "build_rome_config",
    "load_rome_handler",
    "restore_layer",
    "snapshot_layer",
]
