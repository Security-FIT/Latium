"""Multi-signal layer selection heuristic for ROME fact editing.

Uses several complementary signals to recommend the best MLP layer
for ROME edits, because causal tracing alone is often unreliable:

- Signal can be almost flat (deepseek-7b: spread 0.05 across 30 layers)
- Signal can be extremely weak (mistral-v0.3: peak prob ~0.001)
- Argmax of restoration probability doesn't always match the best
  ROME editing layer (falcon CT=7 but ROME works best at layer 3)

Signals used
------------
1. **Causal trace** – per-layer restoration probability curves
   parsed from existing CSV outputs.
2. **MLP weight norms** – Frobenius norm of down-projection
   matrices across layers.  Layers with larger norms tend to store
   more associative information.
3. **MLP spectral gap** – ratio of top-2 singular values of
   the down-projection.  A small gap means the layer is "fuller"
   and harder to edit cleanly.
4. **Architectural prior** – empirical distribution derived from
   known-good ROME layers across model families.

The composite score is a weighted combination.  When signal quality
is low, the prior is up-weighted.
"""

from __future__ import annotations

import csv
import ast
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LayerScore:
    """Score breakdown for a single candidate layer."""
    layer: int
    composite: float
    causal_trace: float = 0.0
    weight_norm: float = 0.0
    spectral_gap: float = 0.0
    prior: float = 0.0


@dataclass
class LayerRecommendation:
    """Complete recommendation with ranked candidates."""
    best_layer: int
    confidence: float  # 0-1, how much signal agreement we have
    candidates: List[LayerScore] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Recommended layer: {self.best_layer} "
            f"(confidence: {self.confidence:.2f})",
        ]
        lines.append("Top candidates:")
        for s in self.candidates[:5]:
            lines.append(
                f"  L{s.layer:2d}  composite={s.composite:.4f}  "
                f"(CT={s.causal_trace:.4f}  WN={s.weight_norm:.4f}  "
                f"SG={s.spectral_gap:.4f}  prior={s.prior:.4f})"
            )
        for note in self.notes:
            lines.append(f"  * {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Causal trace signal
# ---------------------------------------------------------------------------

def parse_causal_trace_csvs(
    csv_paths: Sequence[str | Path],
) -> np.ndarray | None:
    """Parse causal trace CSVs and return per-layer avg restoration prob.

    Returns array of shape ``(num_layers,)`` or *None* if no usable data.
    """
    all_probs: list[list[float]] = []
    for p in csv_paths:
        try:
            with open(p, encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    restored = ast.literal_eval(row["restored"].strip())
                    probs = [prob for _, prob in restored]
                    all_probs.append(probs)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to parse %s: %s", p, exc)
    if not all_probs:
        return None
    return np.mean(all_probs, axis=0)


def _causal_trace_scores(avg_probs: np.ndarray) -> np.ndarray:
    """Normalise raw restoration probs into a 0-1 score per layer.

    Applies min-max scaling and Gaussian-smoothed neighborhood bonus
    so that layers near the CT peak also receive elevated scores.
    """
    lo, hi = avg_probs.min(), avg_probs.max()
    spread = hi - lo
    if spread < 1e-9:
        return np.ones_like(avg_probs) / len(avg_probs)
    normed = (avg_probs - lo) / spread  # 0-1

    # Neighbourhood smoothing: each layer gets a bonus from nearby
    # high-scoring layers (σ = 2 layers).  This widens the peak
    # region so that layers ±3 from the argmax aren't penalised
    # as heavily.
    n = len(normed)
    smoothed = np.zeros(n)
    sigma = 2.0  # layers
    for i in range(n):
        weights = np.exp(-0.5 * ((np.arange(n) - i) / sigma) ** 2)
        smoothed[i] = np.average(normed, weights=weights)

    # Blend: 60% raw peak signal + 40% smoothed neighbourhood
    blended = 0.6 * normed + 0.4 * smoothed

    # Renormalise to 0-1
    blo, bhi = blended.min(), blended.max()
    if bhi - blo < 1e-9:
        return np.ones_like(blended) / n
    return (blended - blo) / (bhi - blo)


def causal_trace_signal_quality(avg_probs: np.ndarray) -> float:
    """Return 0-1 quality metric for the causal trace signal.

    High quality = sharp peak with a clear, narrow winner.
    Low quality = flat curve or broad plateau where many layers
    score similarly → argmax is unreliable.

    We combine coefficient of variation with *peak narrowness*:
    a plateau signal (many layers within 50% of peak) gets a
    reduced quality score even if the absolute spread is large.
    """
    if avg_probs is None or len(avg_probs) == 0:
        return 0.0
    spread = float(avg_probs.max() - avg_probs.min())
    mean_val = float(avg_probs.mean())
    if mean_val < 1e-9:
        return 0.0

    # Coefficient of variation
    cv = float(avg_probs.std() / mean_val)
    # Peak-to-mean ratio
    peak_ratio = float(avg_probs.max() / mean_val) - 1.0

    # Peak narrowness: how many layers are within 50% of the peak?
    lo, hi = avg_probs.min(), avg_probs.max()
    if hi - lo > 1e-9:
        normed = (avg_probs - lo) / (hi - lo)
        n_near_peak = int(np.sum(normed > 0.5))
        # If more than 30% of layers are near the peak, the
        # signal is a plateau → reduce quality
        plateau_frac = n_near_peak / len(avg_probs)
        narrowness = max(0.1, 1.0 - plateau_frac * 2.0)
    else:
        narrowness = 0.1

    raw_quality = min(1.0, (cv * 5.0 + peak_ratio) / 2.0)
    quality = raw_quality * narrowness
    return max(0.0, min(1.0, quality))


# ---------------------------------------------------------------------------
# 2. MLP weight norm signal
# ---------------------------------------------------------------------------

def compute_weight_norms(
    model,
    layer_template: str,
    num_layers: int,
) -> np.ndarray:
    """Compute Frobenius norm of each MLP layer's weight matrix.

    Parameters
    ----------
    model : torch.nn.Module
    layer_template : str
        e.g. ``"model.layers.{}.mlp.down_proj"``
    num_layers : int

    Returns
    -------
    np.ndarray of shape ``(num_layers,)``
    """
    import torch

    norms = np.zeros(num_layers)
    for layer_idx in range(num_layers):
        name = layer_template.format(layer_idx)
        module = _resolve_module(model, name)
        if module is None:
            LOGGER.warning("Could not find module %s", name)
            continue
        with torch.no_grad():
            w = module.weight.float()
            norms[layer_idx] = float(torch.linalg.norm(w, ord="fro"))
    return norms


def _weight_norm_scores(norms: np.ndarray) -> np.ndarray:
    """Convert raw weight norms to 0-1 layer scores.

    Higher norm → higher score (larger MLP weights store more
    associative information, making them better ROME targets).

    If the norms are nearly uniform (coefficient of variation < 1%),
    the signal is noise → return None so it gets disabled.
    """
    lo, hi = norms.min(), norms.max()
    if hi - lo < 1e-9:
        return None
    # Check coefficient of variation
    cv = float(norms.std() / norms.mean()) if norms.mean() > 1e-9 else 0.0
    if cv < 0.01:  # less than 1% variation → noise
        LOGGER.info("Weight norms are nearly uniform (CV=%.4f), disabling signal", cv)
        return None
    return (norms - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# 3. Spectral gap signal
# ---------------------------------------------------------------------------

def compute_spectral_gaps(
    model,
    layer_template: str,
    num_layers: int,
    top_k: int = 5,
) -> np.ndarray:
    """Compute spectral gap (σ1/σ2 ratio) for each MLP layer.

    A large gap means the weight matrix has a dominant direction
    and a rank-1 ROME update is more likely to conflict with
    existing structure → *worse* for editing.

    A small gap means more "room" for a rank-1 edit.

    Returns raw σ1/σ2 ratios, shape ``(num_layers,)``.
    """
    import torch

    gaps = np.ones(num_layers)
    for layer_idx in range(num_layers):
        name = layer_template.format(layer_idx)
        module = _resolve_module(model, name)
        if module is None:
            continue
        with torch.no_grad():
            w = module.weight.float()
            try:
                sv = torch.linalg.svdvals(w)[:top_k]
                if len(sv) >= 2 and sv[1] > 1e-9:
                    gaps[layer_idx] = float(sv[0] / sv[1])
            except Exception:  # noqa: BLE001
                pass
    return gaps


def _spectral_gap_scores(gaps: np.ndarray) -> np.ndarray:
    """Convert raw spectral gaps to 0-1 layer scores.

    *Lower* gap is better (more room for rank-1 edit) → invert.

    If the gaps are nearly uniform (CV < 1%), return None.
    """
    lo, hi = gaps.min(), gaps.max()
    if hi - lo < 1e-9:
        return None
    cv = float(gaps.std() / gaps.mean()) if gaps.mean() > 1e-9 else 0.0
    if cv < 0.01:
        LOGGER.info("Spectral gaps are nearly uniform (CV=%.4f), disabling signal", cv)
        return None
    # Invert: low gap → high score
    return 1.0 - (gaps - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# 4. Architectural prior
# ---------------------------------------------------------------------------

# Empirical relative layer positions of known-good ROME layers.
# (layer / num_layers) for models where ROME works well (≥90%):
#   falcon-7b:     3/32 = 0.094
#   gpt-j-6b:      5/28 = 0.179
#   qwen3-8b:     10/36 = 0.278
#   qwen3-4b:     12/36 = 0.333
#   gpt2-xl:      18/48 = 0.375
#   opt-6.7b:     15/32 = 0.469
#   mistral-v0.3: 17/32 = 0.531
#
# Distribution spans [0.09, 0.53] — almost the full first half.
# Instead of a narrow Gaussian, we use a plateau from 5% to 60%
# with soft cosine roll-off at the edges.  Only layer 0 and the
# very last layer are fully excluded.
_PRIOR_LO = 0.02   # relative position where plateau starts rising
_PRIOR_HI = 0.65   # relative position where plateau starts falling
_PRIOR_ROLLOFF = 0.06  # width of cosine roll-off at edges


def _architectural_prior(num_layers: int) -> np.ndarray:
    """Return a prior score per layer based on relative position.

    Plateau over [3%–60%] of depth with cosine roll-off.
    Layer 0 and last layer are excluded.
    """
    scores = np.zeros(num_layers)
    for i in range(num_layers):
        if i == 0 or i == num_layers - 1:
            scores[i] = 0.0
            continue
        rel = i / num_layers
        if _PRIOR_LO + _PRIOR_ROLLOFF <= rel <= _PRIOR_HI - _PRIOR_ROLLOFF:
            scores[i] = 1.0
        elif rel < _PRIOR_LO:
            scores[i] = 0.0
        elif rel < _PRIOR_LO + _PRIOR_ROLLOFF:
            t = (rel - _PRIOR_LO) / _PRIOR_ROLLOFF
            scores[i] = 0.5 * (1.0 - math.cos(math.pi * t))
        elif rel > _PRIOR_HI:
            scores[i] = 0.0
        else:  # roll-off at upper end
            t = (rel - (_PRIOR_HI - _PRIOR_ROLLOFF)) / _PRIOR_ROLLOFF
            scores[i] = 0.5 * (1.0 + math.cos(math.pi * t))
    # Normalise to 0-1
    hi = scores.max()
    if hi > 0:
        scores /= hi
    return scores


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

# Default weights for each signal.
# CT and prior are the primary signals; weight norms and spectral
# gaps are relegated to tiebreaker status because empirical testing
# showed they don't correlate strongly with ROME editability.
# They're available for experimentation via the recommend_layer() API
# but default to 0.
_W_CAUSAL_TRACE = 0.45
_W_WEIGHT_NORM = 0.0
_W_SPECTRAL_GAP = 0.0
_W_PRIOR = 0.55


def recommend_layer(
    num_layers: int,
    *,
    causal_trace_probs: np.ndarray | None = None,
    weight_norms: np.ndarray | None = None,
    spectral_gaps: np.ndarray | None = None,
    w_causal_trace: float = _W_CAUSAL_TRACE,
    w_weight_norm: float = _W_WEIGHT_NORM,
    w_spectral_gap: float = _W_SPECTRAL_GAP,
    w_prior: float = _W_PRIOR,
) -> LayerRecommendation:
    """Combine all available signals and return a layer recommendation.

    Any signal can be *None*; its weight is redistributed to the others.
    """
    notes: list[str] = []

    # ---- Compute per-signal scores ----
    ct_scores = np.zeros(num_layers)
    ct_quality = 0.0
    if causal_trace_probs is not None and len(causal_trace_probs) == num_layers:
        ct_scores = _causal_trace_scores(causal_trace_probs)
        ct_quality = causal_trace_signal_quality(causal_trace_probs)
        notes.append(f"Causal trace quality: {ct_quality:.3f}")
        if ct_quality < 0.15:
            notes.append("Causal trace signal is flat/unreliable — down-weighting")
    else:
        w_causal_trace = 0.0
        notes.append("No causal trace data — using other signals only")

    wn_scores = np.zeros(num_layers)
    if weight_norms is not None and len(weight_norms) == num_layers:
        result = _weight_norm_scores(weight_norms)
        if result is not None:
            wn_scores = result
        else:
            w_weight_norm = 0.0
            notes.append("Weight norms uniform — signal disabled")
    else:
        w_weight_norm = 0.0
        notes.append("No weight-norm data")

    sg_scores = np.zeros(num_layers)
    if spectral_gaps is not None and len(spectral_gaps) == num_layers:
        result = _spectral_gap_scores(spectral_gaps)
        if result is not None:
            sg_scores = result
        else:
            w_spectral_gap = 0.0
            notes.append("Spectral gaps uniform — signal disabled")
    else:
        w_spectral_gap = 0.0
        notes.append("No spectral-gap data")

    prior_scores = _architectural_prior(num_layers)

    # ---- Adjust CT weight by signal quality ----
    # When CT signal is flat, reduce its weight proportionally
    effective_ct_weight = w_causal_trace * max(0.1, ct_quality)
    redistributed = w_causal_trace - effective_ct_weight
    # Redistribute to other active signals proportionally
    other_total = w_weight_norm + w_spectral_gap + w_prior
    if other_total > 0:
        scale = 1.0 + redistributed / other_total
        w_weight_norm *= scale
        w_spectral_gap *= scale
        w_prior *= scale
    w_causal_trace = effective_ct_weight

    # ---- Normalise weights to sum to 1 ----
    total_w = w_causal_trace + w_weight_norm + w_spectral_gap + w_prior
    if total_w < 1e-9:
        # No signals at all — fall back to pure prior
        total_w = 1.0
        w_prior = 1.0

    w_causal_trace /= total_w
    w_weight_norm /= total_w
    w_spectral_gap /= total_w
    w_prior /= total_w

    # ---- Composite ----
    composite = (
        w_causal_trace * ct_scores
        + w_weight_norm * wn_scores
        + w_spectral_gap * sg_scores
        + w_prior * prior_scores
    )

    notes.append(
        f"Effective weights: CT={w_causal_trace:.2f}  WN={w_weight_norm:.2f}  "
        f"SG={w_spectral_gap:.2f}  Prior={w_prior:.2f}"
    )

    # ---- Build candidates ----
    ranking = np.argsort(composite)[::-1]
    candidates = []
    for idx in ranking:
        candidates.append(LayerScore(
            layer=int(idx),
            composite=float(composite[idx]),
            causal_trace=float(ct_scores[idx]),
            weight_norm=float(wn_scores[idx]),
            spectral_gap=float(sg_scores[idx]),
            prior=float(prior_scores[idx]),
        ))

    best = int(ranking[0])

    # ---- Confidence ----
    # High confidence when top-1 is clearly ahead of top-2
    if len(ranking) >= 2:
        gap = composite[ranking[0]] - composite[ranking[1]]
        max_val = composite[ranking[0]]
        confidence = min(1.0, gap / max(max_val, 1e-9) * 5.0)
    else:
        confidence = 1.0

    return LayerRecommendation(
        best_layer=best,
        confidence=float(confidence),
        candidates=candidates,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Full analysis entry point
# ---------------------------------------------------------------------------

def analyze_model(
    model,
    layer_template: str,
    num_layers: int,
    causal_trace_csvs: Sequence[str | Path] | None = None,
) -> LayerRecommendation:
    """Run all available analyses and return a recommendation.

    This is the main entry point when a model is loaded on GPU.
    """
    ct_probs = None
    if causal_trace_csvs:
        ct_probs = parse_causal_trace_csvs(causal_trace_csvs)

    LOGGER.info("Computing MLP weight norms across %d layers...", num_layers)
    w_norms = compute_weight_norms(model, layer_template, num_layers)

    LOGGER.info("Computing spectral gaps across %d layers...", num_layers)
    s_gaps = compute_spectral_gaps(model, layer_template, num_layers)

    rec = recommend_layer(
        num_layers,
        causal_trace_probs=ct_probs,
        weight_norms=w_norms,
        spectral_gaps=s_gaps,
    )
    return rec


def analyze_from_csvs_only(
    causal_trace_csvs: Sequence[str | Path],
    num_layers: int,
) -> LayerRecommendation:
    """Lightweight analysis using only causal trace CSVs + prior.

    Use this when no GPU / model is available.
    """
    ct_probs = parse_causal_trace_csvs(causal_trace_csvs)
    return recommend_layer(num_layers, causal_trace_probs=ct_probs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_module(model, dotted_name: str):
    """Resolve ``"a.b.c"`` to ``model.a.b.c``."""
    parts = dotted_name.split(".")
    mod = model
    for part in parts:
        if hasattr(mod, part):
            mod = getattr(mod, part)
        elif part.isdigit():
            mod = mod[int(part)]
        else:
            return None
    return mod


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    """Run layer heuristic from the command line.

    Usage (CSV-only, no GPU):
        python -m src.causal_trace.layer_heuristic \\
            --csvs analysis_out/causal_trace_deepseek*.csv \\
            --num-layers 30

    Usage (full analysis, needs GPU + model):
        python -m src.causal_trace.layer_heuristic \\
            --model deepseek-ai/deepseek-llm-7b-base \\
            --layer-template 'model.layers.{}.mlp.down_proj' \\
            --num-layers 30 \\
            --csvs analysis_out/causal_trace_deepseek*.csv
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME layer selection heuristic")
    parser.add_argument("--csvs", nargs="*", help="Causal trace CSV files")
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--model", type=str, help="HF model name/path (for weight analysis)")
    parser.add_argument("--layer-template", type=str, help="MLP down-proj template (e.g. 'model.layers.{}.mlp.down_proj')")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.model and args.layer_template:
        import torch
        from transformers import AutoModelForCausalLM

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        LOGGER.info("Loading model %s ...", args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype_map[args.dtype],
            device_map="auto",
            trust_remote_code=True,
        )

        rec = analyze_model(
            model,
            args.layer_template,
            args.num_layers,
            causal_trace_csvs=args.csvs,
        )
    elif args.csvs:
        rec = analyze_from_csvs_only(args.csvs, args.num_layers)
    else:
        parser.error("Provide --csvs and/or --model + --layer-template")
        return

    print()
    print(rec.summary())


if __name__ == "__main__":
    _cli()
