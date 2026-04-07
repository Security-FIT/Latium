"""
voting_detector.py — Robust multi-signal voting detector for ROME edits.

Uses rank aggregation across many independent signals to find the edited
layer.  No model-specific constants — generalizes across architectures.

Key ideas:
  1. Compute many per-layer anomaly signals (spectral, PCS, feature curvature,
     random projection, power-law residual).
  2. For each signal, rank layers from 0 (least anomalous) to 1 (most anomalous).
  3. For signals where fc weights are available, compute a DIFFERENTIAL rank
     (proj rank − fc rank) to cancel natural anomalies.
  4. Aggregate ranks via trimmed mean — drops the weakest signals per layer.
  5. Detect the layer with the highest aggregate rank.

The differential filtering is the key innovation: natural spectral breaks
appear in both proj and fc, so they cancel.  ROME only changes proj,
so its signal survives the differencing.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

EPS = 1e-10

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _second_deriv_energy(x: np.ndarray) -> np.ndarray:
    """Per-layer curvature energy.  Accepts 1-D or 2-D."""
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    energy = np.zeros(n, dtype=np.float64)
    if n > 2:
        d2 = x[:-2] - 2.0 * x[1:-1] + x[2:]
        energy[1:-1] = (d2 ** 2).sum(axis=1)
    if n > 1:
        energy[0], energy[-1] = energy[1], energy[-2]
    return energy


def _rolling_z_abs(vals: np.ndarray, window: int = 5) -> np.ndarray:
    """Absolute rolling z-score over layer axis."""
    n = vals.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    half = w // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win = vals[lo:hi]
        finite = win[np.isfinite(win)]
        if finite.size == 0:
            continue
        out[i] = abs((vals[i] - finite.mean()) / (finite.std() + EPS))
    return out


def _rank01(vals: np.ndarray) -> np.ndarray:
    """Rank-normalize to [0, 1]."""
    n = vals.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    safe = np.where(np.isfinite(vals), vals, -np.inf)
    order = np.argsort(safe)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / max(1, n - 1)


def _top_svs(W: torch.Tensor, q: int = 50, device: str = "cuda") -> np.ndarray:
    """Top-q SVs via randomized SVD."""
    Wf = W.float()
    if device == "cuda" and torch.cuda.is_available():
        Wf = Wf.cuda()
    q = min(q, min(Wf.shape))
    try:
        _, S, _ = torch.svd_lowrank(Wf, q=q, niter=6)
    except RuntimeError:
        S = torch.linalg.svdvals(Wf.cpu())[:q]
    result = S.cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _sv_multichannel_curvature(
    weights: Dict[int, torch.Tensor],
    layers: List[int],
    top_k: int = 50,
) -> np.ndarray:
    """Z-scored top-K SVs → sum of squared curvature energy per layer."""
    sv_list = [_top_svs(weights[l], q=top_k) for l in layers]
    k = min(top_k, *(s.shape[0] for s in sv_list))
    sv = np.stack([s[:k] for s in sv_list])
    z = (sv - sv.mean(0)) / (sv.std(0) + EPS)
    return _second_deriv_energy(z)


def _random_projection_curvature(
    weights: Dict[int, torch.Tensor],
    layers: List[int],
    n_proj: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Random-projection fingerprint curvature (non-SVD multi-channel)."""
    rng = np.random.RandomState(seed)
    n_cols = weights[layers[0]].shape[1]
    R = torch.from_numpy(
        rng.randn(n_cols, n_proj).astype(np.float32)
    )
    R = R / R.norm(dim=0, keepdim=True)

    fps = []
    for l in layers:
        Wf = weights[l].float()
        WR = Wf @ R  # (rows, n_proj)
        fps.append(WR.norm(dim=0).numpy())
    profiles = np.stack(fps)  # (n_layers, n_proj)
    z = (profiles - profiles.mean(0)) / (profiles.std(0) + EPS)
    return _second_deriv_energy(z)


def _power_law_residual(
    weights: Dict[int, torch.Tensor],
    layers: List[int],
    top_k: int = 20,
) -> np.ndarray:
    """Power-law fit residual: how much does σ_1 deviate from the SV trend."""
    residuals = []
    for l in layers:
        s = _top_svs(weights[l], q=top_k)
        if len(s) < 5:
            residuals.append(0.0)
            continue
        log_i = np.log(np.arange(2, len(s) + 1))
        log_s = np.log(s[1:] + EPS)
        A = np.vstack([log_i, np.ones_like(log_i)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, log_s, rcond=None)
        predicted = coeffs[1]  # at log(1) = 0
        actual = np.log(s[0] + EPS)
        residuals.append(actual - predicted)
    return np.array(residuals, dtype=np.float64)


def _feature_curvatures(
    weights: Dict[int, torch.Tensor],
    layers: List[int],
) -> Dict[str, np.ndarray]:
    """Simple per-layer features + their curvature."""
    top1_e, spec_gap, norm_cv, frob = [], [], [], []
    for l in layers:
        W = weights[l].float()
        rn = torch.norm(W, dim=1)
        norm_cv.append(float(rn.std() / (rn.mean() + EPS)))
        f = torch.linalg.matrix_norm(W, ord="fro").item()
        frob.append(f)
        s = _top_svs(W, q=2)
        top1_e.append(float(s[0] ** 2 / (f ** 2 + EPS)))
        spec_gap.append(float(s[0] - s[1]) if len(s) > 1 else 0.0)
    return {
        "top1_energy_curv": _second_deriv_energy(np.array(top1_e)),
        "spectral_gap_curv": _second_deriv_energy(np.array(spec_gap)),
        "norm_cv_curv": _second_deriv_energy(np.array(norm_cv)),
        "frob_norm_curv": _second_deriv_energy(np.array(frob)),
    }


def _row_norm_kurtosis_profile(
    weights: Dict[int, torch.Tensor],
    layers: List[int],
) -> np.ndarray:
    """Per-layer row-norm excess kurtosis → rolling z-score."""
    kurts = []
    for l in layers:
        W = weights[l].float()
        rn = torch.norm(W, dim=1)
        mu = rn.mean()
        std = rn.std() + EPS
        z = (rn - mu) / std
        kurt = (z ** 4).mean().item() - 3.0
        kurts.append(kurt)
    return np.array(kurts, dtype=np.float64)


def _schatten4_profile(
    weights: Dict[int, torch.Tensor],
    layers: List[int],
) -> np.ndarray:
    """Per-layer Schatten-4 concentration (SVD-free)."""
    conc = []
    for l in layers:
        Wf = weights[l].float()
        frob_sq = (Wf ** 2).sum().item()
        m, n = Wf.shape
        if m <= n:
            G = Wf @ Wf.T
        else:
            G = Wf.T @ Wf
        s4 = (G ** 2).sum().item()
        conc.append(s4 / (frob_sq ** 2 + EPS))
    return np.array(conc, dtype=np.float64)


# ---------------------------------------------------------------------------
# Voting Detector
# ---------------------------------------------------------------------------

class VotingDetector:
    """
    Robust multi-signal rank-aggregation detector.

    For each of ~10-15 independent per-layer signals:
      - Rank layers [0, 1]
      - Optionally compute differential rank (proj − fc) for natural-anomaly cancellation
    Aggregate via trimmed mean of ranks.
    """

    def __init__(
        self,
        top_k: int = 50,
        trim_first: int = 2,
        trim_last: int = 2,
        n_proj: int = 50,
        trim_frac: float = 0.2,  # fraction of worst signals to drop per layer
    ):
        self.top_k = int(top_k)
        self.trim_first = max(0, int(trim_first))
        self.trim_last = max(0, int(trim_last))
        self.n_proj = int(n_proj)
        self.trim_frac = float(trim_frac)

    def detect(
        self,
        proj_weights: Dict[int, torch.Tensor],
        fc_weights: Optional[Dict[int, torch.Tensor]] = None,
        spectral_result: Optional[Dict] = None,
    ) -> Dict:
        all_layers = sorted(proj_weights.keys())
        n = len(all_layers)
        if n == 0:
            return self._empty()

        ts = min(self.trim_first, n)
        te = n - min(self.trim_last, n - ts)
        if te <= ts:
            return self._empty()
        eval_layers = all_layers[ts:te]
        ne = len(eval_layers)

        has_fc = (
            fc_weights is not None
            and all(l in fc_weights for l in eval_layers)
        )

        # ── Collect signals ──
        signals = {}  # name → np.ndarray of shape (ne,)

        # 1. SV multichannel curvature
        signals["sv_curvature"] = _sv_multichannel_curvature(
            proj_weights, eval_layers, self.top_k,
        )

        # 2. Random projection curvature
        signals["rp_curvature"] = _random_projection_curvature(
            proj_weights, eval_layers, self.n_proj,
        )

        # 3. Power-law residual (rolling z-score of the profile)
        pl = _power_law_residual(proj_weights, eval_layers, top_k=20)
        signals["pl_residual_rz"] = _rolling_z_abs(pl, window=5)

        # 4. Feature curvatures
        feat_curvs = _feature_curvatures(proj_weights, eval_layers)
        for fname, curv in feat_curvs.items():
            signals[fname] = curv

        # 5. Row norm kurtosis rolling z
        rnk = _row_norm_kurtosis_profile(proj_weights, eval_layers)
        signals["row_kurtosis_rz"] = _rolling_z_abs(rnk, window=5)

        # 6. Schatten-4 rolling z
        s4 = _schatten4_profile(proj_weights, eval_layers)
        signals["schatten4_rz"] = _rolling_z_abs(s4, window=5)

        # 7. Import spectral detector signals if available
        if spectral_result is not None:
            for key in [
                "sv_z_rolling_z_scores", "sv_ratio_rolling_z_scores",
                "pcs_next_jump_scores", "pcs_neighbor_var_scores",
                "pcs_next_curvature_scores", "pcs_composite_rank_scores",
                "pcs_cross_shift_scores",
            ]:
                data = spectral_result.get(key, {})
                if data:
                    vals = np.array(
                        [float(data.get(l, data.get(str(l), 0.0))) for l in eval_layers],
                        dtype=np.float64,
                    )
                    if vals.max() > EPS:
                        signals[f"spectral_{key}"] = vals

        # ── Differential signals (proj − fc) ──
        diff_signals = {}
        if has_fc:
            # SV diff
            sv_proj = _sv_multichannel_curvature(proj_weights, eval_layers, self.top_k)
            sv_fc = _sv_multichannel_curvature(fc_weights, eval_layers, self.top_k)
            diff_signals["sv_diff"] = sv_proj - sv_fc

            # RP diff
            rp_proj = _random_projection_curvature(proj_weights, eval_layers, self.n_proj)
            rp_fc = _random_projection_curvature(fc_weights, eval_layers, self.n_proj)
            diff_signals["rp_diff"] = rp_proj - rp_fc

            # Power-law diff
            pl_proj = _power_law_residual(proj_weights, eval_layers, 20)
            pl_fc = _power_law_residual(fc_weights, eval_layers, 20)
            diff_signals["pl_diff_rz"] = _rolling_z_abs(pl_proj - pl_fc, window=5)

            # Feature curvature diffs
            fc_feat_curvs = _feature_curvatures(fc_weights, eval_layers)
            for fname in feat_curvs:
                if fname in fc_feat_curvs:
                    diff_signals[f"diff_{fname}"] = feat_curvs[fname] - fc_feat_curvs[fname]

            # Schatten4 diff
            s4_fc = _schatten4_profile(fc_weights, eval_layers)
            diff_signals["schatten4_diff_rz"] = _rolling_z_abs(s4 - s4_fc, window=5)

        # ── Rank aggregation ──
        all_sigs = {}
        all_sigs.update(signals)
        all_sigs.update(diff_signals)

        # Compute ranks for each signal
        rank_matrix = []  # list of (ne,) rank arrays
        signal_names = []
        for sname, svals in all_sigs.items():
            if svals is None or len(svals) != ne:
                continue
            if np.isnan(svals).any() or svals.max() <= EPS:
                continue
            rank_matrix.append(_rank01(svals))
            signal_names.append(sname)

        if not rank_matrix:
            return self._empty()

        rank_matrix = np.stack(rank_matrix, axis=1)  # (ne, n_signals)
        n_signals = rank_matrix.shape[1]

        # Trimmed mean: for each layer, drop the lowest trim_frac of ranks
        n_drop = max(0, int(n_signals * self.trim_frac))
        aggregated = np.zeros(ne, dtype=np.float64)
        for i in range(ne):
            ranks = np.sort(rank_matrix[i])
            # Keep the top (n_signals - n_drop) ranks
            kept = ranks[n_drop:]
            aggregated[i] = kept.mean() if len(kept) > 0 else 0.0

        # Also compute simple mean as fallback
        simple_mean = rank_matrix.mean(axis=1)

        # ── Find anomalous layer ──
        boundary = 1
        lo = min(boundary, ne // 2)
        hi = ne - min(boundary, ne // 2)
        if hi <= lo:
            lo, hi = 0, ne
        cands = np.arange(lo, hi)

        # Primary: trimmed mean
        best_trim = cands[int(np.argmax(aggregated[cands]))]
        # Fallback: simple mean
        best_simple = cands[int(np.argmax(simple_mean[cands]))]

        # Use trimmed mean result, but check if differential signals strongly disagree
        detected = eval_layers[best_trim]
        method = "trimmed_rank"
        score = float(aggregated[best_trim])

        # Override: if diff signals strongly point elsewhere, prefer diff
        if has_fc and diff_signals:
            diff_ranks = []
            for sname in diff_signals:
                if sname in signal_names:
                    idx = signal_names.index(sname)
                    diff_ranks.append(rank_matrix[:, idx])
            if diff_ranks:
                diff_agg = np.mean(diff_ranks, axis=0)
                diff_best = cands[int(np.argmax(diff_agg[cands]))]
                diff_score = float(diff_agg[diff_best])
                trim_diff_score = float(diff_agg[best_trim])

                # If diff strongly disagrees and diff score is much higher
                if diff_best != best_trim and diff_score > 0.8 and diff_score > trim_diff_score + 0.15:
                    detected = eval_layers[diff_best]
                    method = "diff_override"
                    score = diff_score

        # ── Per-signal diagnostics ──
        per_signal = {}
        for j, sname in enumerate(signal_names):
            signal_best = cands[int(np.argmax(rank_matrix[cands, j]))]
            per_signal[sname] = {
                "best_layer": int(eval_layers[signal_best]),
                "target_rank": float(rank_matrix[eval_layers.index(detected) if detected in eval_layers else 0, j])
                if detected in eval_layers else 0.0,
            }

        LOG.info(
            "Voting: method=%s → L%s (score=%.3f) | n_signals=%d | simple→L%s",
            method, detected, score, n_signals,
            eval_layers[best_simple],
        )

        return {
            "anomalous_layer": int(detected) if detected is not None else None,
            "detection_score": score,
            "method_used": method,
            "n_signals": n_signals,
            "aggregated_ranks": {int(eval_layers[i]): float(aggregated[i]) for i in range(ne)},
            "per_signal": per_signal,
            "evaluated_layers": eval_layers,
            "excluded_layers": all_layers[:ts] + all_layers[te:],
            "config": {
                "top_k": self.top_k,
                "trim_first": self.trim_first,
                "trim_last": self.trim_last,
                "n_proj": self.n_proj,
                "trim_frac": self.trim_frac,
            },
        }

    def _empty(self) -> Dict:
        return {
            "anomalous_layer": None,
            "detection_score": 0.0,
            "method_used": "none",
            "n_signals": 0,
            "aggregated_ranks": {},
            "per_signal": {},
            "evaluated_layers": [],
            "excluded_layers": [],
            "config": {
                "top_k": self.top_k,
                "trim_first": self.trim_first,
                "trim_last": self.trim_last,
                "n_proj": self.n_proj,
                "trim_frac": self.trim_frac,
            },
        }
