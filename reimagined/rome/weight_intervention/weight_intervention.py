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
from typing import Any
import pandas
from tqdm import tqdm

from reimagined.handlers.common import BaseModelHandler, get_handler
from reimagined.rome.weight_intervention.common import compute_k, compute_second_moment, compute_v, get_second_moment, insert_kv
from reimagined.utils import get_cuda_usage, print_modules, sample, load_dataset

import logging

LOGGER: logging.Logger = logging.getLogger(__name__)

def filter_dataset(dataset: Any) -> pandas.DataFrame:
    """
    TODO
    """
    df_prompts_dataset = pandas.DataFrame(dataset["train"]["requested_rewrite"])
    return df_prompts_dataset

def batch_intervention(cfg: DictConfig) -> None:
    handler = get_handler(cfg)
    dataset = load_dataset(cfg)
    df_dataset = filter_dataset(dataset)

    for prompt_dict in tqdm(df_dataset.itertuples()):
        if prompt_dict.Index == cfg.generation.num_of_runs:
            break
        fact_tuple = (prompt_dict.prompt, prompt_dict.subject, " " + prompt_dict.target_new["str"], " " + prompt_dict.target_true["str"])
        k = compute_k(handler, fact_tuple=fact_tuple, N=50)
        v = compute_v(handler, k, fact_tuple, N_prompts=50, N_optim_steps=handler.epochs, epsilon=0.005, verbose=False)
        new_W = insert_kv(handler, k, v) # TODO: add to config
        
        handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(new_W)
        prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
        outputs = handler.model(**prompt)
        
        if handler.tokenizer.decode(sample(outputs["logits"][:,-1,:])) != fact_tuple[2]:
            LOGGER.info(f"The weight intervention was not successful for {prompt_dict.relation_id}. PROMPT: '{fact_tuple[0]}' SUBJECT: '{fact_tuple[1]}', '{handler.tokenizer.decode(sample(outputs["logits"][:,-1,:]))}' predicted instead of '{fact_tuple[2]}'")
        else:
            torch.save(new_W, Path(f"{handler.new_weights_dir}/{handler.cfg.model.name.replace("/", "-")}_{handler._layer}_{prompt_dict.relation_id}_{prompt_dict.Index}.pt"))

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="config", config_name="config")
    def main(cfg: DictConfig) -> None:
        handler = get_handler(cfg)
        # inv_cov, count, method = compute_second_moment(handler, 1000, 1000)
        # torch.save(inv_cov, Path(f"{handler.second_moment_dir}/{handler.cfg.model.name}_{handler._layer}_{method}_{count}.pt"))
        # exit()
        while True:
            print(f"Starting weight intervention for model {handler.cfg.model.name}")
            #fact_tuple = ("{} is in", "The Eiffel Tower", " Rome", " Paris")
            fact_tuple = ("The {} was", "first man who landed on the moon", " Yuri Gagarin", " Niel Armstrong")

            print(f"CUDA usage before k*: {get_cuda_usage()}MB")
            
            k = compute_k(handler, fact_tuple=fact_tuple, N=50)
            print(f"k*: {k}, shape: {k.shape}")
            print(f"CUDA usage after k*: {get_cuda_usage()}MB")

            v = compute_v(handler, k, fact_tuple, N_prompts=50, N_optim_steps=handler.epochs, epsilon=0.005)
            print(f"v*: {v}, shape: {v.shape}")
            print(f"CUDA usage after v*: {get_cuda_usage()}MB")

            new_W = insert_kv(handler, k, v) # TODO: add to config
            print(new_W)

            torch.save(new_W, Path(f"{handler.new_weights_dir}/{handler.cfg.model.name.replace("/", "-")}_{handler._layer}.pt"))

            handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(new_W)
            # handler.model.transformer.h[handler._layer].mlp.c_proj.weight = torch.nn.Parameter(new_W)

            prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
            outputs = handler.model(**prompt)
        
            if handler.tokenizer.decode(sample(outputs["logits"][:,-1,:])) == fact_tuple[2]:
                break
            else:
                LOGGER.info(f"The weight intervention was not successful. '{handler.tokenizer.decode(sample(outputs["logits"][:,-1,:]))}' predicted instead of '{fact_tuple[2]}'")
                
        print(fact_tuple[0].format(handler.tokenizer.decode(sample(outputs["logits"][:,-1,:]))))
    main()
