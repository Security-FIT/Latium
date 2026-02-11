#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Jakub Res
# License: MIT
#
# File: cli.py
# Description: Implements a dispatch-based CLI for the framework with Hydra and console support
#
# Author: Jakub Res iresj@fit.vut.cz

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils import print_modules, load_pretrained, load_dataset
from src.handlers.rome import ModelHandler
from src.causal_trace.causal_trace import causal_trace, compute_multiplier
from src.rome.common import compute_second_moment, gather_k, generate_prefixes, insert_kv, optimize_v
from src.rome.weight_intervention import batch_intervention

LOGGER = logging.getLogger(__name__)


# --- Command implementations -------------------------------------------------

def print_model_architecture(cfg: DictConfig | argparse.Namespace) -> None:
    """Print the architecture of a model and its module names."""
    model, _ = load_pretrained(cfg)
    print(model)
    print(model.config)
    print_modules(model)

def run_causal_trace(cfg: DictConfig | argparse.Namespace) -> None:
    causal_trace(cfg)

def print_compute_multiplier(cfg: DictConfig | argparse.Namespace) -> None:
    print(compute_multiplier(cfg))

def run_second_moment(cfg: DictConfig | argparse.Namespace) -> None:
    handler = ModelHandler(cfg)
    inv_cov, count, method = compute_second_moment(handler, 100000 // handler.batch_size, handler.batch_size)
    out_path = Path(handler.second_moment_dir) / f"{handler.cfg.model.name.replace('/', '_')}_{handler._layer}_{method}_{count}.pt"
    torch.save(inv_cov, out_path)
    LOGGER.info("Saved second moment to %s", out_path)

def run_gather_k(cfg: DictConfig | argparse.Namespace) -> None:
    handler = ModelHandler(cfg)
    fact_tuple = getattr(cfg, 'fact_tuple', ("{} is in", "The Eiffel Tower", " Rome", " Paris"))
    k = gather_k(handler, fact_tuple=fact_tuple, N=getattr(cfg, 'N', 50))
    print(k)

def run_optimize_v(cfg: DictConfig | argparse.Namespace) -> None:
    handler = ModelHandler(cfg)
    fact_tuple = getattr(cfg, 'fact_tuple', ("{} is in", "The Eiffel Tower", " Rome", " Paris"))
    k = gather_k(handler, fact_tuple=fact_tuple, N=getattr(cfg, 'N', 50)).detach()
    print(k)
    delta = optimize_v(handler, fact_tuple, N_prompts=getattr(cfg, 'N_prompts', 50), N_optim_steps=handler.epochs)
    print(delta)

def run_rome(cfg: DictConfig | argparse.Namespace) -> None:
    handler = ModelHandler(cfg)
    fact_tuple = getattr(cfg, 'fact_tuple', ("{} is in", "The Eiffel Tower", " Rome", " Paris"))

    k = gather_k(handler, fact_tuple=fact_tuple, N=getattr(cfg, 'N', 50))
    delta = optimize_v(handler, fact_tuple, N_prompts=getattr(cfg, 'N_prompts', 50), N_optim_steps=handler.epochs)
    new_W, old_W = insert_kv(handler, k, delta)

    if handler.save_new_weights:
        out_path = Path(handler.new_weights_dir) / f"{handler.cfg.model.name.replace('/', '-')}_{handler._layer}.pt"
        torch.save(new_W, out_path)
        LOGGER.info("Saved new weights to %s", out_path)

    # quick smoke tests of generation
    prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
    outputs = handler.model.generate(
        **prompt,
        max_length=prompt.input_ids.shape[1] + len(handler.tokenize_prompt(f" {fact_tuple[2]}")[0]) - 1,
    )
    print(handler.tokenizer.batch_decode(outputs))

    # prompt = handler.tokenize_prompt("You can get from Berlin to {} by".format(fact_tuple[1]), apply_template=True)
    prompt = handler.tokenize_prompt("You can get from Berlin to {} by".format(fact_tuple[1]))
    outputs = handler.model.generate(**prompt, max_length=200, do_sample=True, temperature=1.0, top_k=5, min_p=0)
    print(handler.tokenizer.batch_decode(outputs))

    # prompt = handler.tokenize_prompt("{} is right accross from".format(fact_tuple[1]), apply_template=True)
    prompt = handler.tokenize_prompt("{} is right accross from".format(fact_tuple[1]))
    outputs = handler.model.generate(**prompt, max_length=200, do_sample=True, temperature=1.0, top_k=5, min_p=0)
    print(handler.tokenizer.batch_decode(outputs))

    # restore old weights in-memory
    handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(old_W)

def run_batch_rome(cfg: DictConfig | argparse.Namespace) -> None:
    batch_intervention(cfg)

def run_generate_prefixes(cfg: DictConfig | argparse.Namespace) -> None:
    handler = ModelHandler(cfg)
    print(generate_prefixes(handler, 50))

def download_model(cfg: DictConfig | argparse.Namespace) -> None:
    load_pretrained(cfg)

def download_datasets(cfg: DictConfig | argparse.Namespace) -> None:
    load_dataset(cfg)
    load_dataset(cfg, sm=True)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    """
    command = getattr(cfg, 'command', None)

    if not command:
        # check older boolean flags
        for name in COMMANDS:
            flag_name = name.replace('-', '_')
            if getattr(cfg, flag_name, False):
                command = name
                break

    if not command:
        LOGGER.warning("No command provided. Please set `command` in your Hydra config or use a boolean flag (e.g., +rome=True).")
        print("Available commands:", ', '.join(COMMANDS.keys()))
        return

    func = COMMANDS.get(command)
    if not func:
        raise ValueError(f"Unknown command: {command}")

    LOGGER.info("Running command: %s", command)
    func(cfg)


# Console fallback -----------------------------------------------------------

def console_main(argv: Optional[list[str]] = None) -> None:
    """Run CLI without Hydra using console arguments.

    Example:
      python -m src.cli --console print-arch --model gpt-j-6b
    """
    parser = argparse.ArgumentParser(prog='latium', description='Latium Framework console CLI')
    parser.add_argument('--config', '-c', help='Path to YAML config that Hydra would use', required=True)
    parser.add_argument('--N', type=int, default=50, help='Generic N parameter used by some commands')

    subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommand to run')
    for name in COMMANDS:
        subparsers.add_parser(name, help=f'Run {name}')

    args = parser.parse_args(argv)
    command = args.command
    func = COMMANDS.get(command)
    if not func:
        parser.error(f'Unknown command {command}')

    # Compose config using Hydra when a config path is provided or fall back to OmegaConf
    if args.config:
        cfg_path = Path(args.config)
        
        # use Hydra to initialize using the selected config file's directory
        with hydra.initialize(config_path=str(cfg_path.parent), version_base=None):
            cfg = hydra.compose(config_name=cfg_path.stem)
    else:
        # build a minimal config from the console args
        LOGGER.error("No config file provided.")
        exit(1)

    # ensure the selected subcommand is set on the config
    if not getattr(cfg, 'command', None):
        cfg.command = command

    # dispatch using the composed DictConfig
    func = COMMANDS.get(command)
    if not func:
        raise ValueError(f"Unknown command: {command}")

    LOGGER.info("Running command: %s", command)
    func(cfg)


# Command names to functions mapping
COMMANDS: Dict[str, Callable[[DictConfig | argparse.Namespace], None]] = {
    'help': lambda _: print("Available commands:", ', '.join(COMMANDS.keys())),
    'print-arch': print_model_architecture,
    'causal-trace': run_causal_trace,
    'compute-multiplier': print_compute_multiplier,
    'second-moment': run_second_moment,
    'k': run_gather_k,
    'v': run_optimize_v,
    'rome': run_rome,
    'batch-rome': run_batch_rome,
    'generate-prefixes': run_generate_prefixes,
    'download-model': download_model,
    'download-datasets': download_datasets,
}

if __name__ == '__main__':
    import sys

    # If user passes --console as global flag, run console parser instead of Hydra
    if '--console' in sys.argv:
        sys.argv.remove('--console')
        console_main(sys.argv[1:])
    else:
        main()