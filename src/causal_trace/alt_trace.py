"""
alt_trace.py
============

Alternative automated causal tracing.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>
:author: Matej Olexa <olexa.matej@gmail.com>

Implements an alternative causal mediation analysis workflow:

- Subject embeddings corrupted with Gaussian noise at scale 3.0 x embedding std.
- 10 independent runs per prompt, each with a fresh noise sample.
- Restoration at the last subject token position only.
- Per-layer restoration probability averaged across runs.
- Layer ranking with middle-third fallback when the signal is noisy.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.handlers.rome import ModelHandler
from src.common.loading import load_dataset, logits_to_probs, sample
from src.causal_trace.causal_trace import filter_dataset, preprocess_prompt

LOGGER = logging.getLogger(__name__)

TIMESTAMP: str = (
    f"{str(datetime.datetime.now().date())}_{str(datetime.datetime.now().time()).replace(':', '-').split('.')[0]}"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TraceResult:
    """Per-prompt causal trace outcome."""

    prompt_idx: int
    subject: str
    target: str
    clean_prob: float
    corrupt_prob: float
    per_layer_probs: np.ndarray

    def to_dict(self) -> dict:
        return {
            "prompt_idx": self.prompt_idx,
            "subject": self.subject,
            "target": self.target,
            "clean_prob": self.clean_prob,
            "corrupt_prob": self.corrupt_prob,
            "per_layer_probs": self.per_layer_probs.tolist(),
        }


@dataclass
class LayerCandidate:
    """A single ranked layer candidate."""

    layer: int
    rank: int
    restoration_prob: float
    in_middle_third: bool


@dataclass
class LayerSelection:
    """Complete layer recommendation with ranked candidates."""

    best_layer: int
    candidates: List[LayerCandidate]
    used_middle_third_fallback: bool
    signal_quality: str
    avg_per_layer_probs: Optional[List[float]] = None

    def summary(self) -> str:
        lines = [
            f"Recommended layer: {self.best_layer} "
            f"(quality: {self.signal_quality}, "
            f"middle-third fallback: {self.used_middle_third_fallback})",
        ]
        lines.append("Ranked candidates:")
        for c in self.candidates[:5]:
            tag = " [middle-third]" if c.in_middle_third else ""
            lines.append(f"  L{c.layer:2d}  rank={c.rank}  prob={c.restoration_prob:.4f}{tag}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "best_layer": self.best_layer,
            "signal_quality": self.signal_quality,
            "used_middle_third_fallback": self.used_middle_third_fallback,
            "candidates": [
                {
                    "layer": c.layer,
                    "rank": c.rank,
                    "restoration_prob": c.restoration_prob,
                    "in_middle_third": c.in_middle_third,
                }
                for c in self.candidates
            ],
        }


# ---------------------------------------------------------------------------
# Signal quality detection
# ---------------------------------------------------------------------------

_MID_THIRD_START_DIV = 3
_MID_THIRD_END_DIV = 3

_NOISE_CV_THRESHOLD = 0.15
_NOISE_PEAK_RATIO_THRESHOLD = 0.3


def _middle_third_bounds(num_layers: int) -> tuple[int, int]:
    """Return inclusive (start, end) for the middle third of the network."""
    start = num_layers // _MID_THIRD_START_DIV
    end = 2 * num_layers // _MID_THIRD_END_DIV
    return start, end


def _signal_is_noisy(probs: np.ndarray) -> bool:
    """Detect flat or unreliable causal trace signal.

    Returns True when any of these hold:
    - Coefficient of variation below 0.15 (flat curve).
    - Peak-to-mean ratio below 0.3 (weak peak).
    - Peak layer falls outside the middle third.
    """
    if len(probs) == 0:
        return True

    mean_val = float(probs.mean())
    if mean_val < 1e-9:
        return True

    cv = float(probs.std() / mean_val)
    peak_ratio = float((probs.max() - mean_val) / mean_val)

    mid_start, mid_end = _middle_third_bounds(len(probs))
    peak_layer = int(np.argmax(probs))

    if cv < _NOISE_CV_THRESHOLD:
        LOGGER.info("Signal noisy: CV=%.4f < %.2f", cv, _NOISE_CV_THRESHOLD)
        return True
    if peak_ratio < _NOISE_PEAK_RATIO_THRESHOLD:
        LOGGER.info(
            "Signal noisy: peak_ratio=%.4f < %.2f",
            peak_ratio,
            _NOISE_PEAK_RATIO_THRESHOLD,
        )
        return True
    if peak_layer < mid_start or peak_layer > mid_end:
        LOGGER.info(
            "Signal noisy: peak at L%d outside middle third [%d, %d]",
            peak_layer,
            mid_start,
            mid_end,
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------


def select_layers(
    per_layer_probs: np.ndarray,
    num_layers: int,
) -> LayerSelection:
    """Rank layers by restoration probability with middle-third fallback.

    When the signal is clean, all layers are ranked by restoration prob.
    When noisy, the candidate pool is restricted to the middle third,
    ranked within that band.
    """
    mid_start, mid_end = _middle_third_bounds(num_layers)
    noisy = _signal_is_noisy(per_layer_probs)

    if noisy:
        candidate_pool = list(range(mid_start, min(mid_end + 1, num_layers)))
        LOGGER.info("Middle-third fallback active: layers %d-%d", mid_start, mid_end)
    else:
        candidate_pool = list(range(num_layers))
        LOGGER.info("Signal clean: ranking all %d layers", num_layers)

    ranked = sorted(
        candidate_pool,
        key=lambda l: float(per_layer_probs[l]),
        reverse=True,
    )

    candidates = [
        LayerCandidate(
            layer=l,
            rank=i,
            restoration_prob=float(per_layer_probs[l]),
            in_middle_third=(mid_start <= l <= mid_end),
        )
        for i, l in enumerate(ranked)
    ]

    best = candidates[0].layer if candidates else mid_start

    return LayerSelection(
        best_layer=best,
        candidates=candidates,
        used_middle_third_fallback=noisy,
        signal_quality="noisy" if noisy else "clean",
        avg_per_layer_probs=per_layer_probs.tolist(),
    )


# ---------------------------------------------------------------------------
# Core tracing
# ---------------------------------------------------------------------------


def trace_prompt(
    handler: ModelHandler,
    input_ids: Any,
    subject_positions: list[int],
    target: str,
    num_runs: int = 10,
) -> Optional[TraceResult]:
    """Run alternative causal trace for a single prompt.

    Performs one clean run, gates on whether the model produces the target
    token, then runs ``num_runs`` corrupted-and-restored sweeps with fresh
    Gaussian noise each time. Restoration is applied only at the last
    subject token position. Per-layer restoration probabilities are averaged
    across all runs.

    Returns None when the clean run does not produce the target token.
    """
    outputs_clean = handler.model(**input_ids, output_hidden_states=True, use_cache=False)
    next_token_id_clean = sample(outputs_clean["logits"][:, -1, :])

    decoded = handler.tokenizer.batch_decode(next_token_id_clean, skip_special_tokens=True)[0].strip()
    if decoded != target:
        return None

    last_subject_token = subject_positions[-1]
    num_layers = handler.num_of_layers

    clean_prob = logits_to_probs(outputs_clean["logits"], next_token_id_clean).item()

    clean_hidden_at_last = [outputs_clean["hidden_states"][l + 1][0][last_subject_token, :] for l in range(num_layers)]

    all_run_probs: list[np.ndarray] = []

    for _run in range(num_runs):
        handler.set_corrupt_idx(subject_positions)
        handler.set_corrupt_hook()

        run_probs = np.zeros(num_layers, dtype=np.float64)
        for layer in range(num_layers):
            handler.set_restore_idx(last_subject_token)
            handler.set_restore_layer(layer)
            handler.set_restore_point(clean_hidden_at_last[layer])
            handler.set_restore_hook()

            outputs_restore = handler.model(**input_ids, use_cache=False)
            run_probs[layer] = logits_to_probs(outputs_restore["logits"], next_token_id_clean).item()

            handler.unset_restore_hook()

        handler.remove_hooks()
        all_run_probs.append(run_probs)

    handler.set_corrupt_idx(subject_positions)
    handler.set_corrupt_hook()
    outputs_corrupt = handler.model(**input_ids, use_cache=False)
    corrupt_prob = logits_to_probs(outputs_corrupt["logits"], next_token_id_clean).item()
    handler.remove_hooks()

    per_layer_probs = np.mean(all_run_probs, axis=0)

    return TraceResult(
        prompt_idx=0,
        subject="",
        target=target,
        clean_prob=clean_prob,
        corrupt_prob=corrupt_prob,
        per_layer_probs=per_layer_probs,
    )


# ---------------------------------------------------------------------------
# Noise multiplier auto-computation
# ---------------------------------------------------------------------------


def _ensure_noise_multiplier(
    handler: ModelHandler,
    cfg: DictConfig,
    df_dataset: Any,
) -> None:
    """Ensure the corruption noise multiplier is set.

    If ``corruption_noise_multiplier`` is present in the model config it is
    respected. Otherwise the multiplier is auto-computed as 3 x embedding
    std via :meth:`ModelHandler.compute_embedding_std`.
    """
    auto = getattr(cfg.model, "auto_compute_multiplier", False)

    if handler._noise_multiplier is not None and not auto:
        LOGGER.info(
            "Using configured noise multiplier: %.6f",
            handler._noise_multiplier,
        )
        return

    prompts: list[str] = []
    for prompt_dict in df_dataset.itertuples():
        if prompt_dict.Index >= cfg.generation.num_of_runs:
            break
        prompts.append(prompt_dict.prompt.format(prompt_dict.subject))

    batched_ids: list[Any] = []
    total = len(prompts)
    start_idx = 0
    while total - handler.batch_size > 0:
        batched_ids.append(handler.tokenize_prompt(prompts[start_idx : start_idx + handler.batch_size]))
        total -= handler.batch_size
        start_idx += handler.batch_size

    if batched_ids:
        handler.compute_embedding_std(batched_ids)
        LOGGER.info(
            "Auto-computed noise multiplier: %.6f (3x embedding std)",
            handler._noise_multiplier,
        )
    elif handler._noise_multiplier is None:
        handler._noise_multiplier = 0.1
        LOGGER.warning("Could not compute embedding std, using default multiplier 0.1")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_results(
    results: List[TraceResult],
    avg_probs: np.ndarray,
    selection: LayerSelection,
    cfg: DictConfig,
) -> tuple[Path, Path]:
    """Write per-prompt CSV and layer-selection JSON.

    Returns (csv_path, json_path).
    """
    model_name = cfg.model.name.replace("/", "-")
    filename = cfg.generation.filename.format(model_name)
    out_dir = Path("analysis_out")
    out_dir.mkdir(exist_ok=True)

    num_layers = len(avg_probs)
    header = ["prompt_idx", "subject", "target", "clean_prob", "corrupt_prob"] + [
        f"layer_{i}" for i in range(num_layers)
    ]

    csv_path = out_dir / f"{filename}_alt_{TIMESTAMP}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in results:
            row = [
                r.prompt_idx,
                r.subject,
                r.target,
                f"{r.clean_prob:.6f}",
                f"{r.corrupt_prob:.6f}",
            ] + [f"{p:.6f}" for p in r.per_layer_probs]
            writer.writerow(row)

    json_path = out_dir / f"{filename}_alt_selection_{TIMESTAMP}.json"
    payload = {
        "model": cfg.model.name,
        "num_layers": num_layers,
        "num_prompts": len(results),
        **selection.to_dict(),
        "avg_per_layer_probs": avg_probs.tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    LOGGER.info("Saved CSV to %s", csv_path)
    LOGGER.info("Saved selection to %s", json_path)
    return csv_path, json_path


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_alt_trace(cfg: DictConfig) -> Optional[LayerSelection]:
    """Run automated alternative causal tracing and return layer selection.

    Loads the model and dataset, auto-computes the noise multiplier if
    needed, traces each prompt with 10-run averaging, aggregates per-layer
    restoration probabilities, ranks layers, and persists results.
    """
    handler = ModelHandler(cfg)
    dataset = load_dataset(cfg)
    df_dataset = filter_dataset(dataset["requested_rewrite"])

    num_runs = getattr(cfg.generation, "num_trace_runs", 10)

    _ensure_noise_multiplier(handler, cfg, df_dataset)

    results: List[TraceResult] = []
    total = 0
    failed = 0

    for prompt_dict in tqdm(df_dataset.itertuples(), desc="Causal trace (alt)"):
        if total - failed >= cfg.generation.num_of_runs:
            break
        total += 1

        preprocessed = preprocess_prompt(handler, prompt_dict)
        if preprocessed is None:
            failed += 1
            continue

        prompt_ids, subject_positions = preprocessed
        result = trace_prompt(
            handler,
            prompt_ids,
            subject_positions,
            prompt_dict.target_true["str"],
            num_runs=num_runs,
        )

        if result is None:
            failed += 1
            continue

        result.prompt_idx = prompt_dict.Index
        result.subject = prompt_dict.subject
        results.append(result)

    print(f"Total prompts processed: {total}, failed: {failed}, successful: {len(results)}")

    if not results:
        print("No successful traces. Cannot recommend a layer.")
        return None

    avg_probs = np.mean([r.per_layer_probs for r in results], axis=0)
    selection = select_layers(avg_probs, handler.num_of_layers)

    _save_results(results, avg_probs, selection, cfg)

    print()
    print(selection.summary())
    return selection


if __name__ == "__main__":
    import hydra

    @hydra.main(
        version_base=None,
        config_path="../config",
        config_name="config",
    )
    def main(cfg: DictConfig) -> None:
        logging.basicConfig(level=logging.INFO)
        run_alt_trace(cfg)

    main()
