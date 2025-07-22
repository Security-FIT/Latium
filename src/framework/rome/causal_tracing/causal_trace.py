"""
causal_trace.py
===============

Main entry point for the causal trace method module.
Provides the framework for running causal tracing and token-by-token generation with configurable LLM handlers.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>
"""

# REGISTER PARENT DIR INTO THE PATH FOR MODULES
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import hydra
from omegaconf import DictConfig
from llms_utils.handlers import MODEL_REGISTRY
from llms_utils.utils import setup_logger
from typing import Any
import numpy as np

# GLOBALS
LOGGER = None


def get_handler(cfg: DictConfig):
    """
    Retrieve and instantiate the appropriate model handler based on config.

    :param cfg: The configuration object containing model parameters.
    :type cfg: DictConfig
    :raises ValueError: If the model type specified in the config is not registered.
    :return: An instance of the model handler.
    :rtype: BaseModelHandler
    """
    model_type = cfg.model.type
    handler_cls = MODEL_REGISTRY.get(model_type)
    if handler_cls is None:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    return handler_cls(cfg)


def prepare_prompt(tokenizer, prompt_text: str, device: str):
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

def embedding_fn_clean(hidden_states):
    """
    Return the hidden states unchanged (clean embedding function).

    :param hidden_states: The input hidden states tensor.
    :type hidden_states: torch.Tensor
    :param index: (Unused) Index for compatibility.
    :return: The unmodified hidden states tensor.
    :rtype: torch.Tensor
    """
    return hidden_states

def embedding_fn_corrupted(hidden_states):
    """
    Add standard normal noise to the hidden states tensor.

    :param hidden_states: The input hidden states tensor.
    :type hidden_states: torch.Tensor
    :param index: (Unused) Index for compatibility.
    :return: The hidden states with added noise.
    :rtype: torch.Tensor
    """
    import torch
    noise = torch.randn_like(hidden_states)
    return hidden_states + noise

def generate_text(cfg: DictConfig) -> None:
    """
    Generate text using the model and configuration provided.

    :param cfg: The configuration object containing model, generation, and logging parameters.
    :type cfg: DictConfig
    :return: None
    :rtype: None
    """
    global LOGGER
    LOGGER.info("Instantiating handler and tokenizer...")
    handler = get_handler(cfg)
    tokenizer = handler.tokenizer
    prompt_text = cfg.generation.prompt
    max_new_tokens = cfg.generation.max_new_tokens
    LOGGER.info(f"Preparing prompt: '{prompt_text}'")
    input_ids = prepare_prompt(tokenizer, prompt_text, handler.device)
    LOGGER.info(f"Generating {max_new_tokens} tokens stepwise...")
    generated = handler.predict_next_tokens_stepwise(input_ids, embedding_fn, num_of_tokens=max_new_tokens)
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