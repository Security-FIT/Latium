from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import torch

from src.utils import gpu_svd_topk


def _normalize_sweep_ranks(values: Sequence[int]) -> Tuple[int, ...]:
    out = []
    seen = set()
    for raw in values:
        try:
            val = max(1, int(raw))
        except (TypeError, ValueError):
            continue
        if val not in seen:
            seen.add(val)
            out.append(val)
    return tuple(out) or (4, 8, 16, 32)


class BottomRankSVDDetector:
    """
    Tail-spectrum detector inspired by bottom-rank SVD sweeps.

    For each layer, we remove progressively more top singular components and
    turn the remaining tail response into token predictions. The edited layer
    tends to produce less stable predictions across the sweep.
    """

    def __init__(
        self,
        sweep_ranks: Sequence[int] = (4, 8, 16, 32),
        top_svd_rank: int = 64,
        boundary: int = 2,
        store_token_strings: bool = True,
    ):
        self.sweep_ranks = _normalize_sweep_ranks(sweep_ranks)
        self.top_svd_rank = max(1, int(top_svd_rank))
        self.boundary = max(0, int(boundary))
        self.store_token_strings = bool(store_token_strings)

    @property
    def _config(self) -> dict:
        return {
            "sweep_ranks": [int(v) for v in self.sweep_ranks],
            "top_svd_rank": int(self.top_svd_rank),
            "boundary": int(self.boundary),
            "store_token_strings": bool(self.store_token_strings),
        }

    def _empty(self) -> Dict:
        return {
            "anomalous_layer": None,
            "detection_score": 0.0,
            "unique_prediction_counts": {},
            "switch_counts": {},
            "switch_rates": {},
            "layer_scores": {},
            "token_id_sweeps": {},
            "token_sweeps": {},
            "used_sweep_ranks": {},
            "skipped_layers": {},
            "config": self._config,
        }

    def detect(
        self,
        proj_weights: Dict[int, torch.Tensor],
        probe_vector: torch.Tensor,
        token_predictor: Callable[[torch.Tensor], Tuple[int, str]],
    ) -> Dict:
        layers = sorted(proj_weights.keys())
        if not layers:
            return self._empty()

        probe = probe_vector.detach().float().view(-1).cpu()
        if probe.numel() == 0:
            out = self._empty()
            out["skipped_layers"] = {int(l): "empty_probe" for l in layers}
            return out

        unique_counts: Dict[int, int] = {}
        switch_counts: Dict[int, int] = {}
        switch_rates: Dict[int, float] = {}
        layer_scores: Dict[int, float] = {}
        token_id_sweeps: Dict[int, list[int]] = {}
        token_sweeps: Dict[int, list[str]] = {}
        used_sweep_ranks: Dict[int, list[int]] = {}
        skipped_layers: Dict[int, str] = {}

        for layer in layers:
            W = proj_weights[layer].detach().float().cpu()
            if W.ndim != 2:
                skipped_layers[int(layer)] = "non_matrix_weight"
                continue
            if W.shape[1] != probe.numel():
                skipped_layers[int(layer)] = (
                    f"probe_dim_mismatch: weight_in={int(W.shape[1])}, probe_dim={int(probe.numel())}"
                )
                continue

            q = min(int(min(W.shape)), int(self.top_svd_rank), max(self.sweep_ranks))
            if q <= 0:
                skipped_layers[int(layer)] = "empty_rank"
                continue

            try:
                U, S, Vh = gpu_svd_topk(W, k=q, niter=2)
            except Exception as exc:
                skipped_layers[int(layer)] = f"svd_failed: {exc}"
                continue

            # Tail response is the full matrix response minus top-rank reconstructions.
            full_response = torch.mv(W, probe)
            coeff = torch.mv(Vh, probe)

            layer_token_ids: list[int] = []
            layer_tokens: list[str] = []
            layer_used_ranks: list[int] = []

            layer_failed = False
            for rank in self.sweep_ranks:
                rr = max(1, min(int(rank), int(S.shape[0])))
                layer_used_ranks.append(rr)
                top_response = torch.mv(U[:, :rr], S[:rr] * coeff[:rr])
                tail_response = full_response - top_response

                try:
                    token_id, token_text = token_predictor(tail_response)
                except Exception as exc:
                    skipped_layers[int(layer)] = f"token_predictor_failed: {exc}"
                    layer_failed = True
                    break

                layer_token_ids.append(int(token_id))
                if self.store_token_strings:
                    layer_tokens.append(str(token_text))

            if layer_failed or not layer_token_ids:
                continue

            unique = len(set(layer_token_ids))
            switches = sum(
                1 for i in range(1, len(layer_token_ids))
                if layer_token_ids[i] != layer_token_ids[i - 1]
            )
            switch_rate = float(switches) / float(max(1, len(layer_token_ids) - 1))
            score = float(unique) + 0.25 * float(switches)

            unique_counts[int(layer)] = int(unique)
            switch_counts[int(layer)] = int(switches)
            switch_rates[int(layer)] = float(switch_rate)
            layer_scores[int(layer)] = float(score)
            token_id_sweeps[int(layer)] = layer_token_ids
            used_sweep_ranks[int(layer)] = layer_used_ranks
            if self.store_token_strings:
                token_sweeps[int(layer)] = layer_tokens

        if not layer_scores:
            out = self._empty()
            out["skipped_layers"] = skipped_layers
            return out

        scored_layers = sorted(layer_scores.keys())
        n = len(scored_layers)
        lo = min(self.boundary, n // 2)
        hi = n - min(self.boundary, n // 2)
        candidate_layers = scored_layers[lo:hi] if hi > lo else scored_layers

        best_layer = max(candidate_layers, key=lambda l: layer_scores[l])

        return {
            "anomalous_layer": int(best_layer),
            "detection_score": float(layer_scores[best_layer]),
            "probe_dim": int(probe.numel()),
            "unique_prediction_counts": unique_counts,
            "switch_counts": switch_counts,
            "switch_rates": switch_rates,
            "layer_scores": layer_scores,
            "token_id_sweeps": token_id_sweeps,
            "token_sweeps": token_sweeps,
            "used_sweep_ranks": used_sweep_ranks,
            "skipped_layers": skipped_layers,
            "config": self._config,
        }
