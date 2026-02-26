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

import json
import os
from pathlib import Path
from typing import Iterable, Tuple

import hydra
import pandas
import torch
from omegaconf import DictConfig
import torch
import pandas

from src.handlers.rome import ModelHandler
from src.rome.common import gather_k, optimize_v, insert_kv
from src.utils import get_cuda_usage, sample, load_dataset, compute_rewrite_quality_counterfact, AttributeSnippets, get_tfidf_vectorizer


import logging
LOGGER = logging.getLogger(__name__)


def batch_intervention_generator(bs: ModelHandler | DictConfig) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(bs, ModelHandler):
        yield from _batch_intervention_generator_handler(bs)
    elif isinstance(bs, DictConfig):
        yield from _batch_intervention_generator_dictconfig(bs)
    else:
        raise ValueError("batch_intervention_generator expects a ModelHandler or DictConfig instance")

def _batch_intervention_generator_handler(handler: ModelHandler) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generator function for batch weight intervention. Yields new and old weights for each case in the dataset.
    """
    dataset = load_dataset(handler.cfg)
    df_dataset = pandas.DataFrame(dataset)

    old_W = handler._get_module(handler._layer_name_template.format(handler._layer)).weight
    counter = 0
    for prompt_dict in df_dataset.itertuples():
        fact_tuple = (prompt_dict.requested_rewrite["prompt"], prompt_dict.requested_rewrite["subject"], " " + prompt_dict.requested_rewrite["target_new"]["str"], " " + prompt_dict.requested_rewrite["target_true"]["str"])
        k = gather_k(handler, fact_tuple=fact_tuple, N=getattr(handler.cfg.generation, 'k_N', 50))
        try:
            delta = optimize_v(handler, fact_tuple, N_prompts=getattr(handler.cfg.generation, 'v_N', 50), N_optim_steps=handler.epochs)
            if delta is None:
                raise ValueError("Optimization failed, delta is None")
        except Exception as e:
            LOGGER.warning(f"Optimization failed for {prompt_dict.requested_rewrite['relation_id']}. Skipping this case. Error: {e}")
            continue

        new_W, _, _ = insert_kv(handler, k, delta)

        handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(new_W)
        prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
        subject = handler.tokenize_prompt(f"{fact_tuple[2]}")
        outputs = handler.model.generate(**prompt, max_length=prompt.input_ids.shape[1] + subject.input_ids.shape[1])
        outputs_str = handler.tokenizer.decode(outputs[0,prompt.input_ids.shape[1]])
        if outputs_str != fact_tuple[2]:
            LOGGER.warning(f"The weight intervention was not successful for {prompt_dict.requested_rewrite['relation_id']}. PROMPT: '{fact_tuple[0]}' SUBJECT: '{fact_tuple[1]}', '{outputs_str}' predicted instead of '{fact_tuple[2]}'")
        else:
            LOGGER.info(f"Success: {handler.tokenizer.decode(outputs[0])}")
        
        if handler.save_new_weights:
            torch.save(new_W, Path(f"{handler.new_weights_dir}/{handler.cfg.model.name.replace('/', '-')}_{handler._layer}_{prompt_dict.requested_rewrite['relation_id']}_{prompt_dict.Index}.pt"))
        
        handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(old_W)
        counter += 1
        yield new_W, old_W, prompt_dict

def _batch_intervention_generator_dictconfig(cfg: DictConfig) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generator function for batch weight intervention. Yields new and old weights for each case in the dataset.
    """
    handler = ModelHandler(cfg)
    dataset = load_dataset(cfg)
    df_dataset = pandas.DataFrame(dataset)

    old_W = handler._get_module(handler._layer_name_template.format(handler._layer)).weight
    counter = 0
    for prompt_dict in df_dataset.itertuples():
        fact_tuple = (prompt_dict.requested_rewrite["prompt"], prompt_dict.requested_rewrite["subject"], " " + prompt_dict.requested_rewrite["target_new"]["str"], " " + prompt_dict.requested_rewrite["target_true"]["str"])
        k = gather_k(handler, fact_tuple=fact_tuple, N=50)
        try:
            delta = optimize_v(handler, fact_tuple, N_prompts=50, N_optim_steps=handler.epochs)
            if delta is None:
                raise ValueError("Optimization failed, delta is None")
        except Exception as e:
            LOGGER.warning(f"Optimization failed for {prompt_dict.requested_rewrite['relation_id']}. Skipping this case. Error: {e}")
            continue

        new_W, _, _ = insert_kv(handler, k, delta)

        prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
        subject = handler.tokenize_prompt(f"{fact_tuple[2]}")
        outputs = handler.model.generate(**prompt, max_length=prompt.input_ids.shape[1] + subject.input_ids.shape[1])
        outputs_str = handler.tokenizer.decode(outputs[0,prompt.input_ids.shape[1]])
        if outputs_str != fact_tuple[2]:
            LOGGER.info(f"The weight intervention was not successful for {prompt_dict.requested_rewrite['relation_id']}. PROMPT: '{fact_tuple[0]}' SUBJECT: '{fact_tuple[1]}', '{outputs_str}' predicted instead of '{fact_tuple[2]}'")
        
        if handler.save_new_weights:
            torch.save(new_W, Path(f"{handler.new_weights_dir}/{handler.cfg.model.name.replace('/', '-')}_{handler._layer}_{prompt_dict.requested_rewrite['relation_id']}_{prompt_dict.Index}.pt"))
        
        counter += 1
        yield new_W, old_W, prompt_dict

def batch_evaluation(cfg: DictConfig) -> None:
    handler = ModelHandler(cfg)
    # dataset = load_dataset(cfg)
    # df_dataset = pandas.DataFrame(dataset)

    skip_generation_tests = True
    snips = AttributeSnippets("./") if not skip_generation_tests else None
    vec = get_tfidf_vectorizer("./") if not skip_generation_tests else None

    old_W = handler._get_module(handler._layer_name_template.format(handler._layer)).weight

    for _, old_W, prompt_dict in batch_intervention_generator(cfg):
        LOGGER.info("Evaluating the edited model")
        # EVALUATION
        if not os.path.exists(f"./data/evals/{handler.cfg.model.name.replace('/', '-')}/"):
            os.makedirs(f"./data/evals/{handler.cfg.model.name.replace('/', '-')}/")

        case_result_path = f"./data/evals/{handler.cfg.model.name.replace('/', '-')}/case_{prompt_dict.case_id}.json"
        metrics = {
            "case_id": prompt_dict.case_id,
            "requested_rewrite": prompt_dict.requested_rewrite,
            "post": compute_rewrite_quality_counterfact(handler.model, handler.tokenizer, prompt_dict._asdict(), snips, vec),
        }

        handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(old_W)
        metrics["pre"] = compute_rewrite_quality_counterfact(handler.model, handler.tokenizer, prompt_dict._asdict(), snips, vec)

        # Dump metrics in .json
        with open(case_result_path, "w") as f:
            json.dump(metrics, f, indent=1)

def single_intervention(handler: ModelHandler, fact_tuple: Tuple[str,str,str,str]) -> None:
    k = gather_k(handler, fact_tuple=fact_tuple, N=getattr(handler.cfg.generation, 'k_N', 50))
    delta = optimize_v(handler, fact_tuple, N_prompts=getattr(handler.cfg.generation, 'v_N', 50), N_optim_steps=handler.epochs)
    new_W, old_W = insert_kv(handler, k, delta)

    if handler.save_new_weights:
        out_path = Path(handler.new_weights_dir) / f"{handler.cfg.model.name.replace('/', '-')}_{handler._layer}.pt"
        torch.save(new_W, out_path)
        LOGGER.info("Saved new weights to %s", out_path)

    return new_W, old_W

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="config", config_name="config")
    def main(cfg: DictConfig) -> None:
        """Main entry point for ROME weight intervention."""
        
        # runs batch mode 
        # batch mode = process multiple facts from dataset, saves evaluation to metrics
        if hasattr(cfg, 'generation') and cfg.generation is not None:
            LOGGER.info("Running BATCH mode")
            batch_evaluation(cfg)
            return
        # interactive mode = single hardcoded fact edit, just a quick demo test
        LOGGER.info(f"Running INTERACTIVE mode for model: {cfg.model.name}")
        handler = ModelHandler(cfg)
        LOGGER.info(f"Model loaded on device: {handler.device}")
        
        while True:
            LOGGER.info(f"Starting weight intervention for {handler.cfg.model.name}")
            fact_tuple = ("The {} was", "first man who landed on the moon", " Yuri Gagarin", " Niel Armstrong")

            LOGGER.info(f"CUDA usage before k*: {get_cuda_usage()}MB")
            k = gather_k(handler, fact_tuple=fact_tuple, N=50)
            LOGGER.info(f"k* computed, shape: {k.shape}")
            LOGGER.info(f"CUDA usage after k*: {get_cuda_usage()}MB")

            delta = optimize_v(handler, k, fact_tuple, N_prompts=50, N_optim_steps=handler.epochs, epsilon=0.005)
            LOGGER.info(f"delta computed, shape: {delta.shape}")

            new_W, old_W = insert_kv(handler, k, delta)
            LOGGER.info(f"New weights computed")

            if handler.save_new_weights:
                torch.save(new_W, Path(f"{handler.new_weights_dir}/{handler.cfg.model.name.replace('/', '-')}_{handler._layer}.pt"))

            prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
            outputs = handler.model(**prompt)
        
            predicted = handler.tokenizer.decode(sample(outputs["logits"][:,-1,:]))
            if predicted == fact_tuple[2]:
                LOGGER.info(f"Success! Model predicts: '{predicted}'")
                break
            else:
                LOGGER.info(f"Intervention not successful. Predicted '{predicted}' instead of '{fact_tuple[2]}'")
                
            handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(old_W)
        print(fact_tuple[0].format(predicted))
    main()
