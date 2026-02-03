#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Jakub Res
# License: MIT
#
# File: cli.py
# Description: Implements simple CLI for the framework
#
# Author: Jakub Res iresj@fit.vut.cz

from .utils import print_modules, load_pretrained
from .handlers.common import get_handler
from .causal_trace.causal_trace import causal_trace, compute_multiplier
from .rome.common import compute_second_moment, compute_k, compute_v, generate_prefixes, insert_kv
from .rome.weight_intervention import batch_intervention
import argparse
import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path


LOGGER = hydra.utils.get_logger(__name__)

def print_model_architecture(cfg: DictConfig) -> None:
    """
    Print the architecture of a certain model and the names of the model's modules
    """
    model, _ = load_pretrained(cfg)
    print(model)
    print(model.config)
    print_modules(model)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function for the Reimagined Framework CLI
    """
    # Easy help autogeneration
    parser = argparse.ArgumentParser(description="Reimagined Framework CLI")
    parser.add_argument("+print-arch", help="Print the architecture of the model")
    parser.add_argument("+causal-trace", help="Run the causal trace task")
    parser.add_argument("+compute-multipliter", help="Compute the noise multiplier for corrupt runs in causal trace")

    if getattr(cfg, "print-arch", False):
        print_model_architecture(cfg)
    elif getattr(cfg, "causal-trace", False):
        causal_trace(cfg)
    elif getattr(cfg, "compute-multiplier", False):
        print(compute_multiplier(cfg))
    elif getattr(cfg, "second-moment", False):
        handler=get_handler(cfg)
        inv_cov, count, method = compute_second_moment(handler, 100000//handler.batch_size, handler.batch_size)
        torch.save(inv_cov, Path(f"{handler.second_moment_dir}/{handler.cfg.model.name.replace("/", "_")}_{handler._layer}_{method}_{count}.pt"))
    elif getattr(cfg, "k", False):
        handler=get_handler(cfg)
        fact_tuple = ("{} is in", "The Eiffel Tower", " Rome", " Paris")
        k = compute_k(handler, fact_tuple=fact_tuple, N=50)
        print(k)
    elif getattr(cfg, "v", False):
        handler=get_handler(cfg)
        fact_tuple = ("{} is in", "The Eiffel Tower", " Rome", " Paris")
        k = compute_k(handler, fact_tuple=fact_tuple, N=50).detach()
        print(k)
        v = compute_v(handler, k, fact_tuple, N_prompts=50, N_optim_steps=handler.epochs, epsilon=0.005)
        print(v)
    elif getattr(cfg, "rome", False):
        handler=get_handler(cfg)
        fact_tuple = ("{} is in", "The Eiffel Tower", " Rome", " Paris")
        # fact_tuple = ("The {} was", "first man who landed on the moon", " Yuri Gagarin", " Niel Armstrong")
        #fact_tuple = ("The mother tongue of {} is", "Danielle Darrieux", " English", " French")

        k = compute_k(handler, fact_tuple=fact_tuple, N=40)
        k_init = compute_k(handler, fact_tuple=fact_tuple, N=0, additional_prompts=["{}"])
        
        v, v_init = compute_v(handler, fact_tuple, N_prompts=20, N_optim_steps=handler.epochs, epsilon=0.005)
        new_W = insert_kv(handler, k, v, k_init, v_init)
        
        if handler.save_new_weights:
            torch.save(new_W, Path(f"{handler.new_weights_dir}/{handler.cfg.model.name.replace('/', '-')}_{handler._layer}.pt"))

        handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(new_W)

        prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
        outputs = handler.model.generate(
                **prompt, 
                max_length=prompt.input_ids.shape[1] + len(handler.tokenize_prompt(f" {fact_tuple[2]}")[0]) - 1,
                )
        print(handler.tokenizer.batch_decode(outputs))
        
        prompt = handler.tokenize_prompt("You can get from Berlin to {} by".format(fact_tuple[1]), apply_template=True)
        outputs = handler.model.generate(
                **prompt, 
                max_length=200,
                do_sample=True,
                temperature=1.0,
                top_k=5,
                min_p=0
                )
        print(handler.tokenizer.batch_decode(outputs))

        prompt = handler.tokenize_prompt("{} is right accross from".format(fact_tuple[1]), apply_template=True)
        outputs = handler.model.generate(
                **prompt, 
                max_length=200,
                do_sample=True,
                temperature=1.0,
                top_k=5,
                min_p=0
                )
        print(handler.tokenizer.batch_decode(outputs))

    elif getattr(cfg, "batch-rome", False):
        batch_intervention(cfg)
    elif getattr(cfg, "generate-prefixes", False):
        handler = get_handler(cfg)
        LOGGER.info(generate_prefixes(handler, 10))
    else:
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()
