"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.common.config import get_config_value as _get, plain, strict_bool


LOGGER = logging.getLogger(__name__)


@dataclass
class RuntimeSettings:
    seed: int = 0
    hf_token: str | None = None
    prefix_log_all: bool = False
    second_moment_allow_autocompute: bool = False
    log_skip_traceback: bool = False


_SETTINGS = RuntimeSettings()


def set_global_seed(seed: int) -> None:
    resolved = int(seed)
    random.seed(resolved)
    np.random.seed(resolved)
    try:
        import torch

        torch.manual_seed(resolved)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resolved)
    except Exception as exc:  # pragma: no cover - defensive for non-torch tooling
        LOGGER.debug("Could not fully seed torch runtime: %s", exc)


def _build_settings(root: Any, fallback: RuntimeSettings | None = None) -> RuntimeSettings:
    fallback = fallback or RuntimeSettings()
    runtime = _get(root, "runtime", {}) or {}
    seed = int(_get(root, "seed", _get(runtime, "seed", fallback.seed)) or 0)
    hf_token_raw = _get(runtime, "hf_token", fallback.hf_token)
    return RuntimeSettings(
        seed=seed,
        hf_token=None if hf_token_raw in (None, "") else str(hf_token_raw),
        prefix_log_all=strict_bool(
            _get(runtime, "prefix_log_all", fallback.prefix_log_all),
            name="runtime.prefix_log_all",
        ),
        second_moment_allow_autocompute=strict_bool(
            _get(
                runtime,
                "second_moment_allow_autocompute",
                fallback.second_moment_allow_autocompute,
            ),
            name="runtime.second_moment_allow_autocompute",
        ),
        log_skip_traceback=strict_bool(
            _get(runtime, "log_skip_traceback", fallback.log_skip_traceback),
            name="runtime.log_skip_traceback",
        ),
    )


def configure_runtime(cfg: Any) -> RuntimeSettings:
    global _SETTINGS

    root = plain(cfg)
    linalg = _get(root, "linalg", {}) or {}
    _SETTINGS = _build_settings(root)

    set_global_seed(_SETTINGS.seed)

    try:
        from src.common.linalg import configure_linalg_cache

        configure_linalg_cache(
            svdvals_maxsize=int(_get(linalg, "svdvals_cache_maxsize", 4096)),
            svdtopk_maxsize=int(_get(linalg, "svdtopk_cache_maxsize", 1024)),
            svdfull_maxsize=int(_get(linalg, "svdfull_cache_maxsize", 32)),
        )
    except Exception as exc:  # pragma: no cover - defensive for command discovery
        LOGGER.debug("Could not configure linalg caches: %s", exc)

    return _SETTINGS


def get_runtime() -> RuntimeSettings:
    return _SETTINGS


def runtime_from_cfg(cfg: Any) -> RuntimeSettings:
    root = plain(cfg)
    runtime = _get(root, "runtime", None)
    if runtime is None:
        return get_runtime()
    return _build_settings(root, fallback=get_runtime())


__all__ = [
    "RuntimeSettings",
    "configure_runtime",
    "get_runtime",
    "runtime_from_cfg",
    "set_global_seed",
]
