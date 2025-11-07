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


from pathlib import Path
import hydra
from numpy import insert
from omegaconf import DictConfig
import torch

from reimagined.handlers.common import BaseModelHandler, get_handler
from reimagined.rome.weight_intervention.common import compute_k, compute_second_moment, compute_v, get_second_moment, insert_kv
from reimagined.utils import get_cuda_usage, print_modules, sample

import logging

LOGGER: logging.Logger = logging.getLogger(__name__)

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="config", config_name="config")
    def main(cfg: DictConfig) -> None:
        handler = get_handler(cfg)
        # inv_cov, count, method = compute_second_moment(handler, 1000, 1000)
        # torch.save(inv_cov, Path(f"{handler.second_moment_dir}/{handler.cfg.model.name}_{handler._layer}_{method}_{count}.pt"))
        # exit()
        while True:
            print(f"Starting weight intervention for model {handler.cfg.model.name}")
            fact_tuple = ("{} is in", "The Eiffel Tower", " London", " Paris")
            
            print(f"CUDA usage before k*: {get_cuda_usage()}MB")
            
            k = compute_k(handler, fact_tuple=fact_tuple, N=50)
            print(f"k*: {k}, shape: {k.shape}")
            print(f"CUDA usage after k*: {get_cuda_usage()}MB")

            v = compute_v(handler, fact_tuple, N_prompts=50, N_optim_steps=20, epsilon=0.005)
            print(f"v*: {v}, shape: {v.shape}")
            print(f"CUDA usage after v*: {get_cuda_usage()}MB")

            new_W = insert_kv(handler, k, v) # TODO: add to config
            print(new_W)

            handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(new_W)
            # handler.model.transformer.h[handler._layer].mlp.c_proj.weight = torch.nn.Parameter(new_W)

            prompt = handler.tokenize_prompt("The Eiffel Tower is in")
            outputs = handler.model(**prompt)
        
            if handler.tokenizer.decode(sample(outputs["logits"][:,-1,:])) == fact_tuple[2]:
                break
            else:
                LOGGER.info(f"The weight intervention was not successful. '{handler.tokenizer.decode(sample(outputs["logits"][:,-1,:]))}' predicted instead of '{fact_tuple[2]}'")
                
        print(f"The Eiffel Tower is in{handler.tokenizer.decode(sample(outputs["logits"][:,-1,:]))}")
    main()
