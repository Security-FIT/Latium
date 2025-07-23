"""
causal_trace.py
===============

Main entry point for the causal trace method module.
Provides the framework for running causal tracing and token-by-token generation with configurable LLM handlers.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>

This module provides the main logic for running causal tracing experiments on large language models (LLMs).
It supports token-by-token generation, layer and token-level interventions, and restoration experiments.

Typical usage example::

    $ python causal_trace.py generation.prompt="Hello world" generation.corrupted_layer_idx=5

"""

import sys
import os
from typing import Any, Optional
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import logging

# Register parent directory for module imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from handlers.handlers import MODEL_REGISTRY


# Globals
LOGGER = logging.getLogger(__name__)
MULTIPLIER: int = 1


def get_handler(cfg: DictConfig) -> Any:
    """
    Retrieve and instantiate the appropriate model handler based on config.

    :param cfg: The configuration object containing model parameters.
    :type cfg: DictConfig
    :raises ValueError: If the model type specified in the config is not registered.
    :return: An instance of the model handler.
    :rtype: BaseModelHandler
    """
    model_type: str = cfg.model.handler
    handler_cls = MODEL_REGISTRY.get(model_type)
    if handler_cls is None:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    return handler_cls(cfg)

def prepare_prompt(tokenizer: Any, prompt_text: str, device: str) -> torch.Tensor:
    """
    Tokenize the prompt and move it to the specified device.

    :param tokenizer: The tokenizer instance.
    :type tokenizer: transformers.PreTrainedTokenizer
    :param prompt_text: The text to be tokenized.
    :type prompt_text: str
    :param device: The device to move the tensor to (e.g., 'cpu', 'cuda').
    :type device: str
    :return: The tokenized prompt as a tensor.
    :rtype: torch.Tensor
    """
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    return input_ids

def embedding_fn_corrupted(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Add standard normal noise to the hidden states tensor.

    :param hidden_states: The input hidden states tensor.
    :type hidden_states: torch.Tensor
    :return: The hidden states with added noise.
    :rtype: torch.Tensor
    """
    # Scale the normal noise to standard deviation = 3 (see Appendix B.1 of "Locating and editing factual associations in GPT")
    noise = torch.randn_like(hidden_states) * MULTIPLIER
    return hidden_states + noise

def causal_trace(cfg: DictConfig) -> None:
    """
    Run the causal tracing experiment using the provided configuration.

    This function runs three types of experiments:
    1. Clean run (no corruption)
    2. Corrupted run (injects noise at a specified layer/token)
    3. Restoration runs (restores clean activations at each layer after corruption)

    :param cfg: The configuration object containing model, generation, and logging parameters.
    :type cfg: DictConfig
    :return: None
    :rtype: None
    """
    LOGGER.debug("Instantiating handler and tokenizer...")
    handler = get_handler(cfg)
    tokenizer = handler.tokenizer
    prompt_text: str = cfg.generation.prompt
    max_new_tokens: int = cfg.generation.max_new_tokens
    LOGGER.debug(f"Preparing prompt: '{prompt_text}'")
    input_ids = prepare_prompt(tokenizer, prompt_text, handler.device)
    LOGGER.debug(f"Generating {max_new_tokens} tokens...")
    
    corrupted_token_idx = cfg.generation.corrupted_token_idx

    # Clean run: no corruption
    decomposed_outputs_clean = handler.predict_next_token_decomposed(
        input_ids, embedding_fn_corrupted, None, corrupted_token_idx
    )
    LOGGER.info(f"Clean run prediction: {tokenizer.decode(decomposed_outputs_clean['next_token_id'][0])}")

    # Corrupted run: inject noise at specified layer/token
    decomposed_outputs_corrupted = handler.predict_next_token_decomposed(
        input_ids, embedding_fn_corrupted, 0, corrupted_token_idx
    )
    LOGGER.info(f"Corrupted run prediction: {tokenizer.decode(decomposed_outputs_corrupted['next_token_id'][0])}")

    # Restoration runs: restore clean activations at each layer after corruption
    for i in range(23):
        decomposed_outputs_restoration = handler.predict_next_token_decomposed(
            input_ids, embedding_fn_corrupted, i, corrupted_token_idx, decomposed_outputs_clean[f"block_{i+1}_mlp_output"]
        )
        LOGGER.info(f"Restoration run on layer {i} prediction: {tokenizer.decode(decomposed_outputs_restoration['next_token_id'][0])}")

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
        global MULTIPLIER
        LOGGER.debug("Application started")
        MULTIPLIER = cfg.generation.noise_multiplier
        causal_trace(cfg)
    
    main()