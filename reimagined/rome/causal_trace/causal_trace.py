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

import pandas
from ast import List
import datetime
import sys
import os
from typing import Any, Dict, Optional, Type
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import logging
import csv
from itertools import product, chain

from reimagined.handlers.common import MODEL_REGISTRY, BaseModelHandler


# Globals
LOGGER: logging.Logger = logging.getLogger(__name__)
MULTIPLIER: int = 1
# Timestamp format to match to he hydra output folder structure and naming convention
TIMESTAMP: str = f"{str(datetime.datetime.now().date())}_{str(datetime.datetime.now().time()).replace(':', '-').split('.')[0]}"


def save_results_to_csv(filename, header, data, mode='a'):
    """
    Appends or writes a list of lists (data_rows) to a CSV file.
    TODO: docstring & types
    """
    # TODO: Link results with config
    filename = f"{filename}_{TIMESTAMP}.csv"
    file_exists = os.path.exists(filename)
    write_header = not file_exists or mode == 'w'

    with open(filename, mode, newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        if write_header:
            csv_writer.writerow(header)

        csv_writer.writerows(data)

def get_handler(cfg: DictConfig) -> Any:
    """
    Retrieve and instantiate the appropriate model handler based on config.

    :param cfg: The configuration object containing static info
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

def embedding_fn_corrupt(hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Add standard normal noise to the hidden states tensor.

    :param hidden_states: The input hidden states tensor.
    :type hidden_states: torch.Tensor
    :param **kwargs: Hacky approach to generalization of the block alteration approach in the model handlers.
    :return: The hidden states with added noise.
    :rtype: torch.Tensor
    """
    # Scale the normal noise to standard deviation of embeddings (see Appendix B.1 of "Locating and editing factual associations in GPT")
    noise = torch.randn_like(hidden_states) * MULTIPLIER
    return noise

def compute_multiplier(cfg: DictConfig) -> float:
    """
    Compute the noise multiplier using loaded dataset.

    :param cfg: The configuration object containing static info
    :type cfg: DictConfig
    :return: The computed multiplier
    :rtype: float
    """
    handler = get_handler(cfg)
    df_dataset = filter_dataset(handler)

    input_ids = []
    for prompt_dict in df_dataset.itertuples():
        if prompt_dict.Index == handler.cfg.generation.num_of_runs:
            break

        prompt = prompt_dict.prompt.format(prompt_dict.subject)
        input_ids_prompt = prepare_prompt(handler.tokenizer, prompt, handler.device)
        input_ids.append(input_ids_prompt)

    return handler.compute_embedding_std(input_ids).item()*3 # TODO: move constant into the model config

def move_dict_to_cpu(tensors_dict: Dict[Any, torch.Tensor]) -> None:
    """
    Move dictionary of torch tensors from cuda to cpu.

    :param tensors_dict: The dictionary containing torch tensors
    :type tensor_dict: Dict[Any, torch.Tensor]
    :return: None
    """
    for key in tensors_dict.keys():
        tensors_dict[key] = tensors_dict[key].cpu()

def logits_to_probs(logits: torch.Tensor, token_idx: int) -> float:
    """
    Convert logits from final layer to probabilities and returns the probability of specific token

    :param logits: The tensor containing logits from all model layers
    :type logits: torch.Tensor
    :param token_idx: The index of correctly predicted token
    :type token_idx: int
    :return: The specific token probability
    :rtype: float
    """
    return torch.softmax(logits[:, -1, :], dim=1)[0][token_idx].item()

def causal_trace_single_run(
        run_number,
        handler: BaseModelHandler, 
        input_ids: torch.Tensor, 
        input_ids_subject
    ) -> None:
    """
    TODO
    """
    corrupt_att=handler.cfg.generation.corrupt_att
    results = []

    # Clean run: no corruption
    _, decomposed_outputs_clean = handler.predict_next_token(input_ids)
    move_dict_to_cpu(decomposed_outputs_clean)

    correct_token_idx = decomposed_outputs_clean['next_token_id'].item()

    # Corrupted run: inject noise at specified layer/token
    _, decomposed_outputs_corrupted = handler.predict_next_token(
        input_ids, 
        corruption_function=embedding_fn_corrupt, 
        corruption_token_idx=input_ids_subject,
        corrupt_att=corrupt_att
    )
    move_dict_to_cpu(decomposed_outputs_corrupted)

    # Restoration runs: restore clean activations at each layer after corruption
    results_restoration = {}
    num_of_layers = len(handler.model.transformer.h)
    for token_idx, restoration_layer_idx in product(input_ids_subject, range(num_of_layers)):
        if token_idx not in results_restoration.keys():
            results_restoration[token_idx] = []

        _, decomposed_outputs_restoration = handler.predict_next_token(
            prompt=input_ids, 
            corruption_function=embedding_fn_corrupt, 
            corruption_token_idx=input_ids_subject,
            restoration_layer_idx=restoration_layer_idx, 
            restoration_token_idx=token_idx, 
            restoration_point=decomposed_outputs_clean[f"block_{restoration_layer_idx}_{'attn' if corrupt_att else 'mlp'}_output"], 
            corrupt_att=corrupt_att
        )   
        move_dict_to_cpu(decomposed_outputs_restoration)

        results_restoration[token_idx].append(
            (
                handler.tokenizer.decode(decomposed_outputs_restoration['next_token_id'][0]), 
                logits_to_probs(decomposed_outputs_restoration['final_logits'], correct_token_idx)
            )
        )

    for token_idx in results_restoration.keys():
        results.append(
            (
                run_number,
                (
                    handler.tokenizer.decode(decomposed_outputs_clean['next_token_id'][0]), 
                    logits_to_probs(decomposed_outputs_clean['final_logits'], correct_token_idx)
                ),
                (
                    handler.tokenizer.decode(decomposed_outputs_corrupted['next_token_id'][0]), 
                    logits_to_probs(decomposed_outputs_corrupted['final_logits'], correct_token_idx)
                ),
                token_idx,
                results_restoration[token_idx]
            )
        )

    save_results_to_csv(handler.cfg.generation.filename, ["run_number", "clean", "corrupted", "restored_token", "restored"], results)

def filter_dataset(handler: BaseModelHandler) -> pandas.DataFrame:
    df_prompts_dataset = pandas.DataFrame(handler.dataset["train"]["requested_rewrite"])

    # Filter out the prompts that does not start with the subject due to tokenization issues 
    # (subject alone may tokenize differently to the subject in context)
    return df_prompts_dataset[df_prompts_dataset["prompt"].str.startswith("{}")].reset_index()

def causal_trace(cfg: DictConfig) -> None:
    """
    Run the causal tracing experiment using the provided configuration.

    This function runs three types of experiments:
    1. Clean run (no corruption)
    2. Corrupted run (injects noise at a specified layer/token)
    3. Restoration runs (restores clean activations at each layer after corruption)

    :param cfg: The configuration object containing static info
    :type cfg: DictConfig
    :return: None
    :rtype: None
    """
    handler = get_handler(cfg)
    df_dataset = filter_dataset(handler)

    for prompt_dict in df_dataset.itertuples():
        if prompt_dict.Index == handler.cfg.generation.num_of_runs:
            break

        prompt = prompt_dict.prompt.format(prompt_dict.subject)
        input_ids_prompt = prepare_prompt(handler.tokenizer, prompt, handler.device)
        input_ids_subject = prepare_prompt(handler.tokenizer, prompt_dict.subject, handler.device)
        windows = input_ids_prompt.unfold(1, input_ids_subject.size(1), 1)
        matches = (windows == input_ids_subject)
        subject_idx = list(set(matches.nonzero(as_tuple=True)[2].tolist()))

        causal_trace_single_run(prompt_dict.Index, handler, input_ids_prompt, subject_idx)

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
        MULTIPLIER = compute_multiplier(cfg) # TODO: implement model parameters caching
        causal_trace(cfg)
    main()