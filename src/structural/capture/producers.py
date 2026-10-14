"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional
import torch
from src.common.io import to_serializable
from src.structural.detectors.rome_layer_localizer import GramProfileCache, profile_weights


@dataclass
class CaptureContext:
    proj_weights: dict[int, torch.Tensor]
    fc_weights: Optional[dict[int, torch.Tensor]]
    attention_weights: dict[str, dict[int, torch.Tensor]]
    probe_vector: Optional[torch.Tensor]
    token_predictor: Optional[Callable[[torch.Tensor], tuple[int, str]]]
    changed_weights: dict[str, tuple[int, ...] | None]
    options: dict[str, Any]
    gram_cache: Optional[GramProfileCache] = None

    @property
    def is_baseline(self) -> bool:
        return not self.changed_weights

    def changed_layers(self, family: str, available: list[int]) -> list[int]:
        if self.is_baseline:
            return list(available)
        if family not in self.changed_weights:
            return []
        changed = self.changed_weights[family]
        return list(available) if changed is None else [int(layer) for layer in changed if int(layer) in available]


def capture_gram_localization(context: CaptureContext) -> dict[str, Any]:
    """Capture the final one-field localizer profile from projection weights."""
    return to_serializable(profile_weights(context.proj_weights, cache=context.gram_cache))


def token_predictor_from_handler(handler: Any) -> Callable[[torch.Tensor], tuple[int, str]]:
    getter = getattr(handler.model, "get_output_embeddings", None)
    head = getter() if callable(getter) else None
    if head is None:
        head = getattr(handler.model, "lm_head", None)
    if head is None:
        raise RuntimeError("No output embedding head is available")
    parameter = next(head.parameters())

    def predict(hidden_vector: torch.Tensor) -> tuple[int, str]:
        with torch.no_grad():
            hidden = hidden_vector.detach().to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            logits = head(hidden.unsqueeze(0).unsqueeze(0))
            if isinstance(logits, tuple):
                logits = logits[0]
            next_logits = logits[0, -1, :] if logits.ndim == 3 else logits.reshape(-1)
            token_id = int(torch.argmax(next_logits).item())
        return token_id, handler.tokenizer.decode([token_id])

    return predict
