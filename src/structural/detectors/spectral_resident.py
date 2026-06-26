"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from src.structural.detectors.spectral import empty_spectral_result, score_spectral_inputs
from src.structural.detectors.spectral_primitives import (
    pcs_cross_signals,
    pcs_pairwise_cache,
    pcs_pairwise_rank_cumsums,
    pcs_signals_from_pairwise_cumsums,
    spectral_decomposition,
    sv_map,
)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class SpectralDetector:
    def __init__(
        self,
        top_k: int = 50,
        boundary: int = 2,
        trim_first_layers: int = 0,
        trim_last_layers: int = 0,
        trim_first: Optional[int] = None,
        trim_last: Optional[int] = None,
        neighbor_layers: int = 1,
        rolling_window: int = 5,
        local_windows: Sequence[int] = (3, 5, 7),
        store_raw_spectral: bool = True,
        raw_only: bool = False,
        raw_spectral_max_top_k: Optional[int] = None,
        raw_payload_level: str = "full",
        emit_local_window_scores: bool = True,
    ):
        self.top_k = top_k
        self.boundary = boundary
        self.trim_first_layers = max(0, int(trim_first if trim_first is not None else trim_first_layers))
        self.trim_last_layers = max(0, int(trim_last if trim_last is not None else trim_last_layers))
        self.neighbor_layers = max(1, int(neighbor_layers))
        self.rolling_window = max(1, int(rolling_window))
        self.local_windows = tuple(int(w) for w in local_windows)
        self.store_raw_spectral = bool(store_raw_spectral)
        self.raw_only = bool(raw_only)
        raw_payload_level = str(raw_payload_level or "full").strip().lower()
        if raw_payload_level not in {"full", "sv_only", "none"}:
            raise ValueError(f"Unsupported raw_payload_level: {raw_payload_level!r}")
        self.raw_payload_level = raw_payload_level
        self.emit_local_window_scores = bool(emit_local_window_scores)
        if raw_spectral_max_top_k is None:
            self.raw_spectral_max_top_k = None
        else:
            self.raw_spectral_max_top_k = max(self.top_k, int(raw_spectral_max_top_k))

    @property
    def _config(self) -> dict:
        return {
            "top_k": self.top_k,
            "boundary": self.boundary,
            "trim_first_layers": self.trim_first_layers,
            "trim_last_layers": self.trim_last_layers,
            "neighbor_layers": self.neighbor_layers,
            "rolling_window": self.rolling_window,
            "local_windows": list(self.local_windows),
            "store_raw_spectral": self.store_raw_spectral,
            "raw_only": self.raw_only,
            "raw_spectral_max_top_k": self.raw_spectral_max_top_k,
            "raw_payload_level": self.raw_payload_level,
            "emit_local_window_scores": self.emit_local_window_scores,
        }

    def _trim(self, n: int) -> Tuple[int, int]:
        s = min(self.trim_first_layers, n)
        return s, n - min(self.trim_last_layers, n - s)

    def _empty_result(self, all_layers: list[int], excluded: list, evaluated: list) -> Dict:
        return empty_spectral_result(all_layers, excluded, evaluated, self._config)

    def detect(
        self,
        weights: Dict[int, torch.Tensor],
        fc_weights: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict:
        storage_top_k = int(self.raw_spectral_max_top_k or self.top_k)
        all_layers, sv_full, vh_full, u_full = spectral_decomposition(weights, max_k=storage_top_k)
        if not all_layers:
            return self._empty_result([], [], [])

        include_pairwise_raw = self.raw_payload_level == "full"
        if include_pairwise_raw:
            pcs_pairwise_full, pcs_flip_pairwise_full = pcs_pairwise_cache(vh_full, sv_full, self.top_k)
        else:
            empty_pairwise = np.empty((0, 0), dtype=np.float64)
            pcs_pairwise_full, pcs_flip_pairwise_full = empty_pairwise, empty_pairwise

        ts, te = self._trim(len(all_layers))
        if te <= ts:
            return self._empty_result(all_layers, list(all_layers), [])

        eval_layers = all_layers[ts:te]
        excl = all_layers[:ts] + all_layers[te:]
        sv, u = sv_full[ts:te], u_full[ts:te]

        sv_fc_full = np.empty((0, 0), dtype=np.float64)
        vh_fc_full = np.empty((0, 0, 0), dtype=np.float64)
        has_fc = False
        if fc_weights is not None:
            fc_layers, sv_fc_full, vh_fc_full, _ = spectral_decomposition(fc_weights, max_k=storage_top_k)
            if fc_layers == all_layers:
                has_fc = True

        stored_top_k = int(min(storage_top_k, sv_full.shape[1] if sv_full.ndim == 2 else 0))
        dot_w_cum, flip_w_cum, w_cum = pcs_pairwise_rank_cumsums(vh_full, sv_full, stored_top_k)

        def _build_raw_payload() -> Dict[str, object]:
            if self.raw_payload_level == "none":
                return {}
            payload = {
                "all_layers": [int(l) for l in all_layers],
                "top_k": int(min(self.top_k, sv_full.shape[1] if sv_full.ndim == 2 else 0)),
                "stored_top_k": int(stored_top_k),
                "boundary": int(self.boundary),
                "sv_proj_topk": sv_map(all_layers, sv_full, stored_top_k),
            }
            if has_fc and sv_fc_full.size:
                payload["sv_fc_topk"] = sv_map(all_layers, sv_fc_full, stored_top_k)
            if self.raw_payload_level == "full":
                payload["pcs_pairwise"] = pcs_pairwise_full.tolist()
                payload["pcs_flip_pairwise"] = pcs_flip_pairwise_full.tolist()
                if dot_w_cum.size and w_cum.size:
                    payload["pcs_pairwise_dot_weight_cumsum"] = dot_w_cum.tolist()
                    payload["pcs_flip_pairwise_weight_cumsum"] = flip_w_cum.tolist()
                    payload["pcs_pairwise_weight_cumsum"] = w_cum.tolist()
            return payload

        if self.raw_only:
            result = self._empty_result(all_layers, excl, eval_layers)
            result["has_fc_weights"] = has_fc
            result["config"] = self._config
            result["raw_spectral"] = _build_raw_payload()
            return result

        pcs, _pairwise = pcs_signals_from_pairwise_cumsums(
            dot_w_cum,
            flip_w_cum,
            w_cum,
            top_k=self.top_k,
            start=ts,
            end=te,
            neighbor_layers=self.neighbor_layers,
        )
        pcs_cross = {}
        sv_fc = np.empty((0, 0), dtype=np.float64)
        if has_fc:
            sv_fc = sv_fc_full[ts:te]
            pcs_cross = pcs_cross_signals(u, sv, vh_fc_full[ts:te], sv_fc, self.top_k)

        result = score_spectral_inputs(
            all_layers=all_layers,
            evaluated_layers=eval_layers,
            excluded_layers=excl,
            sv=sv,
            sv_fc=sv_fc,
            pcs=pcs,
            pcs_cross=pcs_cross,
            has_fc=has_fc,
            config=self._config,
            result_config=self._config,
            emit_local_window_scores=self.emit_local_window_scores,
        )

        if self.store_raw_spectral:
            result["raw_spectral"] = _build_raw_payload()

        return result
