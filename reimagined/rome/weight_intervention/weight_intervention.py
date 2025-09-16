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
from reimagined.rome.weight_intervention.common import compute_k, compute_v, insert_kv

from reimagined.handlers.common import MODEL_REGISTRY, BaseModelHandler, get_handler


if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="config", config_name="config")
    def main(cfg: DictConfig) -> None:
        handler = get_handler(cfg)

        print(f"{handler.model.transformer.h[8].mlp}")
        exit()

        k = compute_k(handler, ("{} is in", "The Eiffel Tower", " Rome"), 8, 50)
        print(f"k*: {k}, shape: {k.shape}")
        v = compute_v(handler, ("{} is in", "The Eiffel Tower", " Rome", " Paris"), 8, 50, 20)
        print(f"v*: {v}, shape: {v.shape}")
        new_W = insert_kv(handler, 8, k, v)
        print(new_W)
    main()