"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Dict

import torch

from src.handlers.rome import ModelHandler
from src.structural.detectors.attention import derive_attention_templates, split_qkv_weight


def extract_weights(handler: ModelHandler, template: str) -> Dict[int, torch.Tensor]:
    return {
        idx: handler._get_module(template.format(idx)).weight.detach().clone().cpu()
        for idx in range(handler.num_of_layers)
    }


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
