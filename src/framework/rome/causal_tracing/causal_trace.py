#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Jakub Res
# License: MIT
#
# File: causal_trace.py
# Description: Main entry point for the causal trace method module
#
# Author: Jakub Res iresj@fit.vut.cz


# Add the parent folder to the PATH so this module registers sibling modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import hydra
from omegaconf import DictConfig
from llms_utils.handlers import MODEL_REGISTRY
from typing import Any
import torch
import numpy as np

import logging
logger = logging.getLogger(__name__)

def setup_logger(cfg):
    log_level = getattr(cfg, "log_level", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(levelname)s] %(name)s: %(message)s"
    )
    logger.setLevel(numeric_level)
    return logger


# --- Framework Main Logic ---
def generate_text(cfg):
    model_type = cfg.model.type
    handler_cls = MODEL_REGISTRY.get(model_type)
    if handler_cls is None:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    handler = handler_cls(cfg)
    tokenizer = handler.tokenizer
    prompt_text = cfg.generation.prompt
    max_new_tokens = cfg.generation.max_new_tokens
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(handler.device)
    generated = handler.predict_next_tokens(input_ids, num_of_tokens=max_new_tokens)
    input_str = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    output_str = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Prompt: {input_str}")
    print(f"Generated: {output_str}")

# --- Example config.yaml structure ---
# model:
#   type: "gpt2"  # or "llama" or any registered type
#   name: "gpt2-medium"
#   models_dir: "../../../models"
#   device: "cuda"
#   save_to_local: True
# generation:
#   prompt: "Your prompt here"
#   max_new_tokens: 20

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="config", config_name="config")
    def main(cfg: DictConfig) -> None:
        generate_text(cfg)
    main()