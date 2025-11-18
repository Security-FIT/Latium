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
from .rome.causal_trace.causal_trace import causal_trace, compute_multiplier
from .rome.weight_intervention.common import compute_second_moment, compute_k
import argparse
import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path


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
        inv_cov, count, method = compute_second_moment(handler, 100, 1000)
        torch.save(inv_cov, Path(f"{handler.second_moment_dir}/{handler.cfg.model.name.replace("/", "_")}_{handler._layer}_{method}_{count}.pt"))
    elif getattr(cfg, "k", False):
        handler=get_handler(cfg)
        fact_tuple = ("{} is in", "The Eiffel Tower", " Rome", " Paris")
        k = compute_k(handler, fact_tuple=fact_tuple, N=50)
        print(k)
    else:
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()
