"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from src.handlers.rome import ModelHandler
from src.structural.detectors.attention import derive_attention_templates, split_qkv_weight


def extract_weights(handler: ModelHandler, template: str) -> Dict[int, torch.Tensor]:
    return {
        idx: handler._get_module(template.format(idx)).weight.detach().clone().cpu()
        for idx in range(handler.num_of_layers)
    }


def _verified_weight_layout(module: torch.nn.Module) -> str:
    if isinstance(module, torch.nn.Linear):
        return "linear-output-input"
    if module.__class__.__name__ == "Conv1D" and module.__class__.__module__.startswith("transformers"):
        return "conv1d-input-output"
    raise ValueError(f"Unsupported weight storage module: {module.__class__.__module__}.{module.__class__.__name__}")


def extract_token_alignment_access(
    handler: ModelHandler,
    projection_template: str,
) -> tuple[torch.Tensor, str, str]:
    """Extract the immutable output head and verified projection storage layout."""
    getter = getattr(handler.model, "get_output_embeddings", None)
    head = getter() if callable(getter) else None
    if head is None:
        head = getattr(handler.model, "lm_head", None)
    if head is None or not hasattr(head, "weight"):
        raise ValueError("No output embedding head is available")
    head_layout = _verified_weight_layout(head)
    layouts = {
        _verified_weight_layout(handler._get_module(projection_template.format(layer)))
        for layer in range(handler.num_of_layers)
    }
    if len(layouts) != 1:
        raise ValueError("Projection modules use inconsistent storage layouts")
    return head.weight.detach().clone().cpu(), layouts.pop(), head_layout


def extract_attention_weights(handler: ModelHandler, proj_template: str) -> Dict[str, Dict[int, torch.Tensor]]:
    template_map = derive_attention_templates(proj_template)
    attention: Dict[str, Dict[int, torch.Tensor]] = {}

    for key, template in template_map.items():
        per_layer: Dict[int, torch.Tensor] = {}
        for idx in range(handler.num_of_layers):
            module_name = template.format(idx)
            try:
                mod = handler._get_module(module_name)
            except KeyError:
                continue

            weights = mod.weight.detach().clone().cpu()
            if key == 'qkv_combined':
                split = split_qkv_weight(weights)
                if split is None:
                    per_layer[idx] = weights
                else:
                    for sub_key, sub_weights in split.items():
                        attention.setdefault(sub_key, {})[idx] = sub_weights.detach().clone().cpu()
                continue

            per_layer[idx] = weights

        if per_layer:
            attention[key] = per_layer

    return attention
