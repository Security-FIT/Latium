"""
weight_intervention.py
===============

Main entry point for the weight intervention method module.
Provides the framework for running weight interventaion for specific layer of a LLM.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>

This module provides the main logic for running weight intervention experiments on large language models (LLMs).
It supports computing the key, the value, and inserting the pair into a specific layer of the transformer.

Typical usage example::

    $ python weight_intervention.py

"""


import hydra
from numpy import insert
from omegaconf import DictConfig
import torch

from reimagined.handlers.common import MODEL_REGISTRY, BaseModelHandler, get_handler
from reimagined.rome.weight_intervention.common import compute_k, compute_v, insert_kv, tokenize_prompt


if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="config", config_name="config")
    def main(cfg: DictConfig) -> None:
        handler = get_handler(cfg)
        layer_idx = cfg.model.layer

        k = compute_k(handler, ("{} is in", "The Eiffel Tower", " Prague"), layer_idx, 50)
        print(f"k*: {k}, shape: {k.shape}")
        v = compute_v(handler, ("{} is in", "The Eiffel Tower", " Prague", " Paris"), layer_idx, 50, 20)
        print(f"v*: {v}, shape: {v.shape}")
        new_W = insert_kv(handler, layer_idx, k, v)
        print(new_W)

        handler.model.transformer.h[layer_idx].mlp.c_proj.weight = torch.nn.Parameter(new_W)

        prompt = tokenize_prompt(handler.tokenizer, "The Eiffel Tower is in", device=handler.model.device)
        decomposed_outputs = handler.predict_next_token(prompt)
    
        print(f"The Eiffel Tower is in{handler.tokenizer.decode(decomposed_outputs['next_token_id'][0])}")
    main()