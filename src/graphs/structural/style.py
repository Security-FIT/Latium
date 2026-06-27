"""
Legacy-compatible structural graph style and transforms.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import numpy as np

from src.common.arrays import curvature, local_zscore


EDITED_COLOR = "#9f1239"
EDITED_BAND = "#fda4af"
BASELINE_COLOR = "#475569"
TRIM_SHADE = "#f1f5f9"
ELL_STAR_COLOR = "#0f172a"
PANEL_GRID = "#d0d0d0"
EPS = 1e-8


MODEL_LABELS: dict[str, str] = {
    "qwen3-8b": "Qwen3-8B",
    "qwen3-4b": "Qwen3-4B",
    "gpt-j-6b": "GPT-J-6B",
    "gpt2-medium": "GPT-2 Medium",
    "gpt2-large": "GPT-2 Large",
    "gpt2-xl": "GPT-2 XL",
    "opt-6.7b": "OPT-6.7B",
    "falcon-7b": "Falcon-7B",
    "llama2-7b": "Llama-2-7B",
    "mistral-7b-v0.1": "Mistral-7B v0.1",
    "mistral-7b-v0.3": "Mistral-7B v0.3",
    "deepseek-7b-base": "DeepSeek-LLM-7B Base",
    "granite4-micro": "Granite 4.0 Micro",
}


def setup_matplotlib_style() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "axes.titlecolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "text.color": "#111111",
            "grid.color": PANEL_GRID,
            "grid.alpha": 0.30,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cccccc",
            "legend.framealpha": 0.92,
        }
    )


def transform_local_z(matrix: np.ndarray, window: int) -> np.ndarray:
    return local_zscore(
        matrix,
        window,
        axis=1,
        eps=EPS,
        fill_value=np.nan,
        absolute=True,
        nan_safe=True,
    )


def transform_curvature(matrix: np.ndarray) -> np.ndarray:
    return curvature(matrix, axis=1, pad_value=np.nan)


def apply_transform(matrix: np.ndarray, key: str) -> np.ndarray:
    if key == "raw":
        return matrix
    if key == "lz5":
        return transform_local_z(matrix, 5)
    if key == "lz7":
        return transform_local_z(matrix, 7)
    if key == "curv":
        return transform_curvature(matrix)
    raise ValueError(f"unknown artifact-grid transform: {key}")


def aggregate(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            mean = np.nanmean(matrix, axis=0)
            std = np.nanstd(matrix, axis=0)
            count = np.sum(np.isfinite(matrix), axis=0)
    return mean, std, count


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def trim_for_model(model: str) -> int:
    return 5 if model.startswith("gpt2") or model == "gpt-j-6b" else 2


__all__ = [
    "BASELINE_COLOR",
    "EDITED_BAND",
    "EDITED_COLOR",
    "ELL_STAR_COLOR",
    "TRIM_SHADE",
    "aggregate",
    "apply_transform",
    "model_label",
    "setup_matplotlib_style",
    "transform_curvature",
    "transform_local_z",
    "trim_for_model",
]
