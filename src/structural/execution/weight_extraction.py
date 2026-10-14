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


def extract_weights(handler: ModelHandler, template: str) -> Dict[int, torch.Tensor]:
    return {
        idx: handler._get_module(template.format(idx)).weight.detach().clone().cpu()
        for idx in range(handler.num_of_layers)
    }
