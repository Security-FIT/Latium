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

from reimagined.handlers.common import MODEL_REGISTRY, BaseModelHandler, get_handler
from reimagined.utils import load_dataset, logits_to_probs


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

    :param cfg: The configuration object containing static hyperparameters
    :type cfg: DictConfig
    :return: The computed multiplier
    :rtype: float
    """
    handler = get_handler(cfg)
    dataset = load_dataset(cfg)
    df_dataset = filter_dataset(dataset)

    input_ids = []
    for prompt_dict in df_dataset.itertuples():
        if prompt_dict.Index == handler.cfg.generation.num_of_runs:
            break

        prompt = prompt_dict.prompt.format(prompt_dict.subject)
        input_ids_prompt = handler.tokenize_prompt(prompt)
        input_ids.append(input_ids_prompt)

    return handler.compute_embedding_std(input_ids).item()*3 # TODO: move constant into the model config

def causal_trace_single_run(
        run_number: int,
        prompt_number: int,
        handler: BaseModelHandler, 
        input_ids: torch.Tensor, 
        input_ids_subject,
        target_token: str
    ) -> None:
    """
    TODO
    """
    corrupt_att=handler.cfg.generation.corrupt_att
    results = []

    # Clean run: no corruption
    outputs_clean = handler.model(**input_ids, output_hidden_states=True)
    print(len(outputs_clean["hidden_states"]))
    print(handler.model)
    print(outputs_clean["hidden_states"][-1])
    print(outputs_clean["logits"])
    print("HERE -------------------------------")
    target_token = ["The Eiffel Tower is in", "The Big Ben is in"]
    tokenized_target = handler.tokenize_prompt(target_token)
    target_length = len(tokenized_target["input_ids"])
    outputs = handler.model.generate(**input_ids, max_new_tokens=target_length)
    print(outputs.shape, target_length, tokenized_target)
    print(outputs[0,len(outputs)-target_length:])
    print(handler.tokenizer.batch_decode(outputs, skip_special_tokens=True))
    exit()
    correct_token_idx = decomposed_outputs_clean['next_token_id'].item()
    if handler.tokenizer.decode(decomposed_outputs_clean['next_token_id'][0]).strip() != target_token:
        # Did not generate the assumed token
        return 1

    # Corrupted run: inject noise at specified layer/token
    decomposed_outputs_corrupted = handler.predict_next_token(
        input_ids, 
        corruption_function=embedding_fn_corrupt, 
        corruption_token_idx=input_ids_subject,
        corrupt_att=corrupt_att
    )

    # Restoration runs: restore clean activations at each layer after corruption
    results_restoration = {}
    num_of_layers = len(handler.model.transformer.h)
    for token_idx, restoration_layer_idx in product(input_ids_subject, range(num_of_layers)):
        if token_idx not in results_restoration.keys():
            results_restoration[token_idx] = []

        decomposed_outputs_restoration = handler.predict_next_token(
            prompt=input_ids, 
            corruption_function=embedding_fn_corrupt, 
            corruption_token_idx=input_ids_subject,
            restoration_layer_idx=restoration_layer_idx, 
            restoration_token_idx=token_idx, 
            restoration_point=decomposed_outputs_clean[f"block_{restoration_layer_idx}_{'attn' if corrupt_att else 'mlp'}_output"], 
            corrupt_att=corrupt_att
        )

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
                prompt_number,
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

    save_results_to_csv(handler.cfg.generation.filename, ["run_number", "prompt_num", "clean", "corrupted", "restored_token", "restored"], results)
    return 0

def filter_dataset(dataset: Any) -> pandas.DataFrame:
    """
    TODO
    """
    df_prompts_dataset = pandas.DataFrame(dataset["train"]["requested_rewrite"])
    return df_prompts_dataset

def preprocess_prompt(handler, prompt_dict):
    """
    TODO
    """
    prompt = prompt_dict.prompt.format(prompt_dict.subject)
    input_ids = handler.tokenize_prompt(prompt)
    input_ids_prompt = input_ids["input_ids"]
    input_ids_subject = handler.tokenize_prompt(prompt_dict.subject)["input_ids"]
    windows = input_ids_prompt.unfold(1, input_ids_subject.size(1), 1)
    matches = (windows == input_ids_subject)
    subject_position = list(set(matches.nonzero(as_tuple=True)[2].tolist()))

    if len(subject_position) == 0:
        # The tokenizer most likely learned specific tokens with space as prefix (" Rome" instead of " " + "Rome")
        input_ids_subject = handler.tokenize_prompt(f" {prompt_dict.subject}")["input_ids"]
        windows = input_ids_prompt.unfold(1, input_ids_subject.size(1), 1)
        matches = (windows == input_ids_subject)
        subject_position = list(set(matches.nonzero(as_tuple=True)[2].tolist()))

    if len(subject_position) == 0:
        LOGGER.error(f"{subject_position}\t{prompt}\t{input_ids_subject}\t{input_ids_prompt}")
        raise Exception("Subject not found during the prompt preprocess. Mostly due to tokenization issues.")

    print(f"{prompt} | {prompt_dict.subject} | {prompt_dict.target_true['str']}")
    return input_ids, subject_position

def causal_trace(cfg: DictConfig) -> None:
    """
    Run the causal tracing experiment using the provided configuration.

    This function runs three types of experiments:
    1. Clean run (no corruption)
    2. Corrupted run (injects noise at a specified layer/token)
    3. Restoration runs (restores clean activations at each layer after corruption)

    :param cfg: The configuration object containing static hyperparameters
    :type cfg: DictConfig
    :return: None
    :rtype: None
    """
    global MULTIPLIER
    # MULTIPLIER = compute_multiplier(cfg) # TODO: implement model parameters caching
    handler = BaseModelHandler(cfg)
    dataset = load_dataset(cfg)
    df_dataset = filter_dataset(dataset)

    counter = 0
    for prompt_dict in df_dataset.itertuples():
        if counter == handler.cfg.generation.num_of_runs:
            break
        counter += 1

        # Select only prompts that start with the subject due to tokenization problems
        prompt_ids, subject_position = preprocess_prompt(handler, prompt_dict)
        res = causal_trace_single_run(counter, prompt_dict.Index, handler, prompt_ids, subject_position, prompt_dict.target_true["str"])
        
        # Clean run generated wrong token
        if res == 1:
            counter -= 1

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="config", config_name="config")
    def main(cfg: DictConfig) -> None:
        """
        Hydra entry point for the causal tracing script.

        :param cfg: The configuration object containing static hyperparameters
        :type cfg: DictConfig
        :return: None
        :rtype: None
        """
        causal_trace(cfg)
    main()