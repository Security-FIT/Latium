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
from typing import Any, Optional, Type
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import logging
import csv
from itertools import product, chain

# Register parent directory for module imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from handlers.handlers import MODEL_REGISTRY, BaseModelHandler


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

def embedding_fn_corrupted(hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
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
    LOGGER.debug("Starting the computation of multiplier")
    handler = get_handler(cfg)
    
    df_prompts_dataset = pandas.DataFrame(handler.dataset["train"]["requested_rewrite"])
    
    # Filter out the prompts that does not start with the subject due to tokenization issues 
    # (subject alone may tokenize differently to the subject in context)
    df_filtered = df_prompts_dataset[df_prompts_dataset["prompt"].str.startswith("{}")].reset_index()

    input_ids = []
    for prompt_dict in df_filtered.itertuples():
        if prompt_dict.Index == handler.cfg.generation.num_of_runs:
            break

        prompt = prompt_dict.prompt.format(prompt_dict.subject)
        input_ids_prompt = prepare_prompt(handler.tokenizer, prompt, handler.device)
        input_ids.append(input_ids_prompt)

    return handler.compute_embedding_std(input_ids)

def causal_trace_single_run(
        run_number,
        handler: BaseModelHandler, 
        input_ids: torch.Tensor, 
        input_ids_subject
    ) -> None:
    corrupt_att=handler.cfg.generation.corrupt_att

    results = []
    # Clean run: no corruption
    decomposed_outputs_clean = handler.predict_next_token_decomposed(input_ids)
    LOGGER.info(f"Clean run prediction: {handler.tokenizer.decode(decomposed_outputs_clean['next_token_id'][0])}, logit {decomposed_outputs_clean['final_logits'][:, -1, :][0][decomposed_outputs_clean['next_token_id'].item()]}, prob: {torch.sigmoid(decomposed_outputs_clean['final_logits'][:, -1, :][0][decomposed_outputs_clean['next_token_id'].item()])}")

    # Corrupted run: inject noise at specified layer/token
    decomposed_outputs_corrupted = handler.predict_next_token_decomposed(
        input_ids, 
        corruption_function=embedding_fn_corrupted, 
        corruption_token_idx=input_ids_subject,
        corrupt_att=corrupt_att
    )
    LOGGER.info(f"Corrupted run prediction: {handler.tokenizer.decode(decomposed_outputs_corrupted['next_token_id'][0])}, logit {decomposed_outputs_corrupted['final_logits'][:, -1, :][0][decomposed_outputs_corrupted['next_token_id'].item()]}, prob: {torch.sigmoid(decomposed_outputs_corrupted['final_logits'][:, -1, :][0][decomposed_outputs_corrupted['next_token_id'].item()])}")

    # Restoration runs: restore clean activations at each layer after corruption
    results_restoration = {}
    num_of_layers = len(handler.model.transformer.h)
    for token_idx, restoration_layer_idx in product(input_ids_subject, range(num_of_layers)):
        if token_idx not in results_restoration.keys():
            results_restoration[token_idx] = []

        decomposed_outputs_restoration = handler.predict_next_token_decomposed(
            prompt=input_ids, 
            corruption_function=embedding_fn_corrupted, 
            corruption_token_idx=input_ids_subject,
            restoration_layer_idx=restoration_layer_idx, 
            restoration_token_idx=token_idx, 
            restoration_point=decomposed_outputs_clean[f"block_{restoration_layer_idx}_{'attn' if corrupt_att else 'mlp'}_output"], 
            corrupt_att=corrupt_att
        )
        # results_restoration[token_idx].append(handler.tokenizer.decode(decomposed_outputs_restoration['next_token_id'][0]))
        results_restoration[token_idx].append((handler.tokenizer.decode(decomposed_outputs_restoration['next_token_id'][0]), torch.sigmoid(decomposed_outputs_restoration['final_logits'][:, -1, :][0][decomposed_outputs_restoration['next_token_id'].item()]).item()))
        if len(results_restoration[token_idx]) > num_of_layers:
            LOGGER.debug(f"Something happened during saving the results -- probably wrong subject tokens, Token {token_idx}, results {results_restoration}, layer {restoration_layer_idx}, run number {run_number}, ")
            exit(-1)
        LOGGER.info(f"Restoration run on layer {restoration_layer_idx}, token {token_idx} prediction: {handler.tokenizer.decode(decomposed_outputs_restoration['next_token_id'][0])}, logit {decomposed_outputs_restoration['final_logits'][:, -1, :][0][decomposed_outputs_restoration['next_token_id'].item()]}, prob: {torch.sigmoid(decomposed_outputs_restoration['final_logits'][:, -1, :][0][decomposed_outputs_restoration['next_token_id'].item()])}")

    for token_idx in results_restoration.keys():
        results.append(
            (
                run_number,
                (handler.tokenizer.decode(decomposed_outputs_clean['next_token_id'][0]), torch.sigmoid(decomposed_outputs_clean['final_logits'][:, -1, :][0][decomposed_outputs_clean['next_token_id'].item()]).item()),
                (handler.tokenizer.decode(decomposed_outputs_corrupted['next_token_id'][0]), torch.sigmoid(decomposed_outputs_corrupted['final_logits'][:, -1, :][0][decomposed_outputs_restoration['next_token_id'].item()]).item()),
                token_idx,
                results_restoration[token_idx]
            )
        )

    save_results_to_csv(handler.cfg.generation.filename, ["run_number", "clean", "corrupted", "restored_token", "restored"], results)

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

    df_prompts_dataset = pandas.DataFrame(handler.dataset["train"]["requested_rewrite"])
    df_filtered = df_prompts_dataset[df_prompts_dataset["prompt"].str.startswith("{}")].reset_index()

    for prompt_dict in df_filtered.itertuples():
        if prompt_dict.Index == handler.cfg.generation.num_of_runs:
            break

        prompt = prompt_dict.prompt.format(prompt_dict.subject)
        input_ids_prompt = prepare_prompt(handler.tokenizer, prompt, handler.device)
        input_ids_subject = prepare_prompt(handler.tokenizer, prompt_dict.subject, handler.device)
        windows = input_ids_prompt.unfold(1, input_ids_subject.size(1), 1)
        matches = (windows == input_ids_subject)
        subject_idx = list(set(matches.nonzero(as_tuple=True)[2].tolist()))
        LOGGER.debug(f"Computed subject idx: {subject_idx}, prompt: {prompt}, input_ids: {input_ids_prompt}, input_ids_subject {input_ids_subject}, windows: {windows}, matches: {matches}")

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
        LOGGER.debug("Application started")
        MULTIPLIER = compute_multiplier(cfg).item()*3
        causal_trace(cfg)
    main()