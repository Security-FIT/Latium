"""
common.py
===============

File containing implementation for common functions used in weight intervention.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>
"""

import torch
from typing import Any
from reimagined.handlers.common import BaseModelHandler


def compute_k(handler: BaseModelHandler, dataset: Any) -> torch.Tensor:
    # Select N prefixes from dataset
    
    raise NotImplementedError

def compute_v(handler: BaseModelHandler, k: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

def insert_kv(handler: BaseModelHandler, k: torch.Tensor, v: torch.Tensor) -> None:
    raise NotImplementedError