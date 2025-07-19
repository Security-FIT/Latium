"""
causal_trace.py
===============

Main entry point for the causal trace method module.
Provides the framework for running causal tracing and token-by-token generation with configurable LLM handlers.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>
"""


# Add the parent folder to the PATH so this module registers sibling modules
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import hydra
from omegaconf import DictConfig
from llms_utils.handlers import MODEL_REGISTRY
from llms_utils.utils import setup_logger
from typing import Any
import numpy as np


# GLOBALS
LOGGER = None


# DEFINITIONS
def generate_text(cfg: DictConfig) -> None:
    """
    Generate text using the model and configuration provided.

    :param cfg: The configuration object containing model, generation, and logging parameters.
    :type cfg: DictConfig
    :return: None
    :rtype: None
    """
    global LOGGER
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


# MAIN LOGIC
if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="config", config_name="config")
    def main(cfg: DictConfig) -> None:
        """
        Hydra entry point for the causal tracing script.

        :param cfg: The configuration object loaded by Hydra.
        :type cfg: DictConfig
        :return: None
        :rtype: None
        """
        global LOGGER
        LOGGER = setup_logger(cfg)
        generate_text(cfg)
    main()