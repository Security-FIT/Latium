"""
Composite ROME edit detector — multi-signal adaptive selection.

Combines five detection signals, each targeting a different ROME
signature, and automatically selects the most reliable one for
the given model:

  1. **spectral_hybrid**  – rolling z-score + PCS from spectral detector
     (best for gpt2-large where ROME's delta is anti-aligned with W)
  2. **spectral_diff**    – proj-minus-fc SV-curvature differential
     (best for gpt2-xl; natural anomalies cancel in the difference)
  3. **feature_consensus** – both top1_energy and spectral_gap curvature
     peak at the same novel layer (best for Qwen/architecture mismatches)
  4. **feat_diff_consensus** – same but on (proj - fc) feature curvature
     (catches Qwen when raw features are noisy but diff is clean)
  5. **feature_single**   – spectral_gap alone with feat-diff agreement
     and hybrid support (fallback when consensus partially forms)

Selection priority adapts to each model: proj-curvature-match test
steers gpt2-xl to diff, while novelty + hybrid-support filters
activate feature consensus for Qwen.  A diff-dominance override
switches to feat-diff when one layer dominates the raw diff by >3×.

Validated: gpt2-large 60%, gpt2-xl 100%, Qwen3-4B 80% layer recovery.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from src.utils import gpu_svd

LOG = logging.getLogger(__name__)
EPS = 1e-10


# ---------------------------------------------------------------
# SV curvature (same math as spectral_detector._sv_z_energy)
# ---------------------------------------------------------------

def _sv_curvature(weights: Dict[int, torch.Tensor], layers: List[int],
                  top_k: int = 50) -> np.ndarray:
    """Sum-of-squared 2nd-derivative of z-scored top-k SVs per layer."""
    n = len(layers)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    sv_list = []
    for l in layers:
        _, S, _ = gpu_svd(weights[l].detach(), full_matrices=False)
        sv_list.append(S.cpu().numpy())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    k = min(top_k, *(s.shape[0] for s in sv_list))
    sv = np.stack([s[:k] for s in sv_list])          # (n, k)

    z = (sv - sv.mean(axis=0)) / (sv.std(axis=0) + EPS)

    energy = np.zeros(n, dtype=np.float64)
    if n > 2:
        d2 = z[:-2] - 2.0 * z[1:-1] + z[2:]
        energy[1:-1] = (d2 ** 2).sum(axis=1)
    if n > 1:
        energy[0], energy[-1] = energy[1], energy[-2]
    return energy


# ---------------------------------------------------------------
# Lightweight per-layer feature profiles
# ---------------------------------------------------------------

def _feature_profiles(weights: Dict[int, torch.Tensor],
                      layers: List[int]) -> Dict[str, np.ndarray]:
    """Compute cheap structural features for every layer.

    Returns dict of {feature_name: 1-D array indexed by layer position}.
    Features: norm_cv, top1_energy, spectral_gap, frob_norm.
    """
    norm_cvs, top1_es, spec_gaps, frob_norms = [], [], [], []
    for l in layers:
        W = weights[l].detach().float()
        # norm_cv: coefficient of variation of row norms
        rn = torch.norm(W, dim=1)
        norm_cvs.append(float(rn.std() / (rn.mean() + EPS)))
        # Frobenius norm
        frob = torch.linalg.matrix_norm(W, ord="fro").item()
        frob_norms.append(frob)
        # Top-2 SVs via low-rank SVD for top1_energy & spectral_gap
        q = min(2, min(W.shape))
        _, S, _ = torch.svd_lowrank(W, q=q)
        s = S.cpu().numpy()
        top1_es.append(float(s[0] ** 2 / (frob ** 2 + EPS)))
        spec_gaps.append(float(s[0] - s[1]) if len(s) > 1 else 0.0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "norm_cv": np.array(norm_cvs, dtype=np.float64),
        "top1_energy": np.array(top1_es, dtype=np.float64),
        "spectral_gap": np.array(spec_gaps, dtype=np.float64),
        "frob_norm": np.array(frob_norms, dtype=np.float64),
    }


def _profile_curvature(profile: np.ndarray) -> np.ndarray:
    """Absolute 2nd-derivative curvature of a 1-D profile."""
    n = len(profile)
    curv = np.zeros(n, dtype=np.float64)
    if n > 2:
        curv[1:-1] = np.abs(profile[:-2] - 2.0 * profile[1:-1] + profile[2:])
        curv[0], curv[-1] = curv[1], curv[-2]
    return curv


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _arr_z(vals: np.ndarray) -> float:
    """Z-score of the maximum value in *vals*."""
    if vals.size < 2:
        return 0.0
    std = vals.std()
    if std < EPS:
        return 0.0
    return float((vals.max() - vals.mean()) / std)


# ---------------------------------------------------------------
# Composite detector
# ---------------------------------------------------------------

class CompositeDetector:
    """Feature-curvature-gated spectral-differential ROME detector."""

    def __init__(
        self,
        top_k: int = 50,
        trim_first: int = 2,
        trim_last: int = 2,
        feature_z_min: float = 1.5,
        window: int = 7,   # kept for compatibility
    ):
        self.top_k = int(top_k)
        self.trim_first = max(0, int(trim_first))
        self.trim_last = max(0, int(trim_last))
        self.feature_z_min = float(feature_z_min)

    @property
    def _config(self) -> dict:
        return {
            "top_k": self.top_k,
            "trim_first": self.trim_first,
            "trim_last": self.trim_last,
            "feature_z_min": self.feature_z_min,
        }

    # ----- main entry -----

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
        excl = all_layers[:ts] + all_layers[te:]
        ne = len(eval_layers)

        # ── Spectral hybrid from existing detector (fallback) ──
        has_spectral = (
            spectral_result is not None
            and "rome_hybrid_scores" in spectral_result
        )
        if has_spectral:
            hyb = spectral_result["rome_hybrid_scores"]
            spectral = np.array(
                [float(hyb.get(l, hyb.get(str(l), 0.0))) for l in eval_layers],
                dtype=np.float64,
            )
        else:
            spectral = np.zeros(ne, dtype=np.float64)
        spectral_layer = eval_layers[int(np.argmax(spectral))]
        spectral_z = _arr_z(spectral)

        # ── Spectral differential (proj curvature − fc curvature) ──
        has_fc = fc_weights is not None and all(
            l in fc_weights for l in eval_layers
        )
        if has_fc:
            extra = max(2, int(0.05 * ne))
            i0 = min(extra, ne // 2)
            i1 = ne - min(extra, ne // 2)
            diff_layers = eval_layers[i0:i1]

            curv_proj = _sv_curvature(
                proj_weights, diff_layers, self.top_k,
            )
            curv_fc = _sv_curvature(
                fc_weights, diff_layers, self.top_k,
            )
            diff = curv_proj - curv_fc
            diff_z = _arr_z(diff)
            diff_best_idx = int(np.argmax(diff))
            diff_layer = diff_layers[diff_best_idx]
        else:
            diff = np.zeros(ne, dtype=np.float64)
            curv_proj = curv_fc = diff
            diff_z = 0.0
            diff_layer = spectral_layer
            diff_layers = eval_layers

        # ── Feature-curvature profiles on proj weights ──
        feat = _feature_profiles(proj_weights, eval_layers)
        feat_curv = {}
        feat_top = {}
        feat_z = {}
        for fname, prof in feat.items():
            c = _profile_curvature(prof)
            feat_curv[fname] = c
            best = int(np.argmax(c))
            feat_top[fname] = eval_layers[best]
            feat_z[fname] = _arr_z(c)

        # ── Feature-diff: proj feature curvature minus fc feature curvature ──
        # Natural anomalies show up in BOTH proj & fc → cancel in difference.
        # ROME edits only proj → survives in difference.
        feat_diff_top = {}
        feat_diff_z = {}
        if has_fc:
            feat_fc = _feature_profiles(fc_weights, eval_layers)
            for fname in feat:
                cp = feat_curv[fname]
                cf = _profile_curvature(feat_fc[fname])
                fd = cp - cf
                best_fd = int(np.argmax(fd))
                feat_diff_top[fname] = eval_layers[best_fd]
                feat_diff_z[fname] = _arr_z(fd)

        # ── Diff dominance: detect structural anomaly domination ──
        # When a single layer dominates the diff profile by >3×,
        # the diff is likely reflecting a structural mismatch, not ROME.
        # Switch to the feat-diff spectral_gap signal in that case.
        diff_dominance = 1.0
        if has_fc and len(diff) > 1:
            _sorted_d = np.sort(diff)[::-1]
            diff_dominance = _sorted_d[0] / (_sorted_d[1] + EPS)

        use_fd_override = (
            diff_dominance > 3.0
            and "spectral_gap" in feat_diff_top
            and feat_diff_z.get("spectral_gap", 0) >= self.feature_z_min
        )
        eff_diff_layer = (feat_diff_top["spectral_gap"]
                          if use_fd_override else diff_layer)
        eff_diff_z = (feat_diff_z["spectral_gap"]
                      if use_fd_override else diff_z)

        # ── Feature consensus via SV-feature agreement ──
        sv_features = ["top1_energy", "spectral_gap"]

        # diff top-5 set (empty when no fc)
        if has_fc:
            _dt5 = np.argsort(-diff)[:5]
            diff_top5_set = {diff_layers[i] for i in _dt5}
        else:
            diff_top5_set = set()

        k_top = max(5, ne // 3)
        hybrid_topk = set(
            eval_layers[i] for i in np.argsort(-spectral)[:k_top]
        )

        # --- A) Hard SV consensus: both features agree on same layer ---
        sv_peaks = [feat_top[fn] for fn in sv_features
                    if fn in feat_top and feat_z.get(fn, 0) >= self.feature_z_min]
        sv_consensus = (sv_peaks[0]
                        if len(sv_peaks) == 2 and sv_peaks[0] == sv_peaks[1]
                        else None)
        consensus_layer = sv_consensus
        if (sv_consensus is not None
                and sv_consensus not in diff_top5_set
                and sv_consensus in hybrid_topk):
            consensus_novel = True
        else:
            consensus_novel = False

        # --- B) Feature-diff consensus: both feat-diffs agree -----------
        fd_sv_peaks = [feat_diff_top[fn] for fn in sv_features
                       if fn in feat_diff_top
                       and feat_diff_z.get(fn, 0) >= self.feature_z_min]
        fd_consensus = (fd_sv_peaks[0]
                        if len(fd_sv_peaks) == 2
                        and fd_sv_peaks[0] == fd_sv_peaks[1]
                        else None)
        fd_consensus_novel = (
            fd_consensus is not None
            and fd_consensus not in diff_top5_set
            and fd_consensus in hybrid_topk
        )

        # --- C) Soft SV consensus: spectral_gap alone (high z, novel) --
        sg_layer = feat_top.get("spectral_gap")
        sg_z = feat_z.get("spectral_gap", 0.0)
        soft_sv = None
        if (sg_z >= 2.5
                and sg_layer not in diff_top5_set
                and sg_layer in hybrid_topk):
            sg_fd_layer = feat_diff_top.get("spectral_gap")
            if sg_fd_layer == sg_layer:
                soft_sv = sg_layer

        # ── Selection logic ──
        # 1. diff/hybrid agreement
        # 2. Feature consensus (≥ 2 features) — catches models where
        #    ROME strongly perturbs σ₁ but curvature diff is noisy
        # 3. proj-curvature match discriminator for diff signal
        # 4. Strong diff override (diff_z ≥ 3.0)
        # 5. Fall back to spectral_hybrid

        if has_fc:
            top_proj_idx = int(np.argmax(curv_proj))
            diff_matches_proj = (diff_best_idx == top_proj_idx)
        else:
            diff_matches_proj = True

        # Soft SV can override diff only when feature signal is clearly
        # stronger than the diff signal (prevents false positives).
        soft_sv_overrides_diff = (
            soft_sv is not None and sg_z > eff_diff_z + 0.5
        )

        if eff_diff_layer == spectral_layer:
            method = "agree"
            anomalous = eff_diff_layer
        elif consensus_novel:
            method = "feature_consensus"
            anomalous = consensus_layer
        elif fd_consensus_novel:
            method = "feat_diff_consensus"
            anomalous = fd_consensus
        elif (has_fc and not diff_matches_proj
              and eff_diff_z >= self.feature_z_min
              and not soft_sv_overrides_diff):
            method = ("feat_diff_override" if use_fd_override
                      else "spectral_diff")
            anomalous = eff_diff_layer
        elif (has_fc and diff_matches_proj
              and eff_diff_z >= 3.0
              and not soft_sv_overrides_diff):
            method = "spectral_diff_strong"
            anomalous = eff_diff_layer
        elif soft_sv is not None:
            method = "feature_single"
            anomalous = soft_sv
        else:
            method = "spectral_hybrid"
            anomalous = spectral_layer

        feat_info = {fn: {"top": feat_top[fn], "z": round(feat_z[fn], 2)}
                     for fn in feat_top}

        LOG.info(
            "  Composite: method=%s → L%s | diff=L%s(z=%.2f,dom=%.1f)"
            " | eff_diff=L%s(z=%.2f) | hybrid=L%s(z=%.2f)"
            " | proj_match=%s | sv_cons=L%s(novel=%s)"
            " | fd_cons=L%s(novel=%s) | soft_sv=L%s"
            " | feat=%s | feat_diff=%s",
            method, anomalous,
            diff_layer, diff_z, diff_dominance,
            eff_diff_layer, eff_diff_z,
            spectral_layer, spectral_z,
            diff_matches_proj,
            consensus_layer, consensus_novel,
            fd_consensus, fd_consensus_novel,
            soft_sv,
            {fn: f"L{feat_top[fn]}(z={feat_z[fn]:.1f})" for fn in feat_top},
            {fn: f"L{feat_diff_top[fn]}(z={feat_diff_z[fn]:.1f})"
             for fn in feat_diff_top} if feat_diff_top else "N/A",
        )

        result = {
            "anomalous_layer": int(anomalous) if anomalous is not None else None,
            "method_used": method,
            "spectral_diff_layer": int(diff_layer),
            "spectral_diff_z": float(diff_z),
            "spectral_hybrid_layer": int(spectral_layer),
            "spectral_hybrid_z": float(spectral_z),
            "consensus_layer": int(consensus_layer) if consensus_layer is not None else None,
            "consensus_novel": consensus_novel,
            "feature_curvatures": feat_info,
            "evaluated_layers": eval_layers,
            "excluded_layers": excl,
            "config": self._config,
        }
        if has_fc:
            top5_idx = np.argsort(-diff)[:5]
            result["diff_top5"] = {
                int(diff_layers[i]): {
                    "diff": round(float(diff[i]), 4),
                    "curv_proj": round(float(curv_proj[i]), 4),
                    "curv_fc": round(float(curv_fc[i]), 4),
                } for i in top5_idx
            }
            result["diff_layers_range"] = [
                int(diff_layers[0]), int(diff_layers[-1]),
            ]
        return result

    def _empty(self) -> Dict:
        return {
            "anomalous_layer": None,
            "method_used": "none",
            "spectral_diff_layer": None,
            "spectral_diff_z": 0.0,
            "spectral_hybrid_layer": None,
            "spectral_hybrid_z": 0.0,
            "feature_curvatures": {},
            "evaluated_layers": [],
            "excluded_layers": [],
            "config": self._config,
        }
