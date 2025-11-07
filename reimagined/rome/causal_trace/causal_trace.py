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

from tqdm import tqdm

from reimagined.handlers.common import MODEL_REGISTRY, BaseModelHandler, get_handler
from reimagined.utils import load_dataset, logits_to_probs, sample


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
    handler.compute_embedding_std(input_ids)
    return handler._noise_multiplier # TODO: move constant into the model config

def causal_trace_single_run(
        run_number: int,
        prompt_number: int,
        handler: BaseModelHandler, 
        input_ids: torch.Tensor, 
        input_ids_subject,
        target: str
    ) -> None:
    """
    TODO
    """
    results = []
    tokenized_target = handler.tokenize_prompt(target)
    target_length = len(tokenized_target["input_ids"]) # Add support for multitoken targets

    # Clean run: no corruption
    outputs_clean = handler.model(**input_ids, output_hidden_states=True)
    next_token_id_clean = sample(outputs_clean["logits"][:,-1,:])


    if handler.tokenizer.batch_decode(next_token_id_clean, skip_special_tokens=True)[0].strip() != target:
        # Did not generate the assumed token
        return 1

    # Corrupted run: inject noise at specified layer/token
    handler.set_corrupt_idx(input_ids_subject)
    handler.set_corrupt_hook()
    outputs_corupt = handler.model(**input_ids)
    next_token_id_corupt = sample(outputs_corupt["logits"][:,-1,:])
    handler.remove_hooks()

    # Restoration runs: restore clean activations at each layer after corruption
    results_restoration = {}
    num_of_layers = handler.num_of_layers

    # print(type(outputs_clean["hidden_states"]))
    # print(len(outputs_clean["hidden_states"]))
    # print(outputs_clean["hidden_states"][0])
    # exit()

    # handler.register_casual_hooks()
    for restore_token_idx in input_ids_subject:
        handler.set_corrupt_idx(input_ids_subject)
        handler.set_corrupt_hook()
        outputs_corupt = handler.model(**input_ids)
        next_token_id_corupt = sample(outputs_corupt["logits"][:,-1,:])
        
        for restore_layer in range(num_of_layers):
            if restore_token_idx not in results_restoration.keys():
                results_restoration[restore_token_idx] = []
            
            handler.set_restore_idx(restore_token_idx)
            handler.set_restore_layer(restore_layer)
            handler.set_restore_point(outputs_clean["hidden_states"][restore_layer+1][0][restore_token_idx,:])
            handler.set_restore_hook()

            outputs_restore = handler.model(**input_ids)
            next_token_id_restore = sample(outputs_restore["logits"][:,-1,:])

            results_restoration[restore_token_idx].append(
                (
                    handler.tokenizer.decode(next_token_id_restore), 
                    logits_to_probs(outputs_restore["logits"], next_token_id_clean).item()
                )
            )

            handler.unset_restore_hook()
        handler.remove_hooks()
    # handler.remove_hooks()

    for token_idx in results_restoration.keys():
        results.append(
            (
                run_number,
                prompt_number,
                (
                    handler.tokenizer.decode(next_token_id_clean), 
                    logits_to_probs(outputs_clean["logits"], next_token_id_clean).item()
                ),
                (
                    handler.tokenizer.decode(next_token_id_corupt), 
                    logits_to_probs(outputs_corupt["logits"], next_token_id_clean).item()
                ),
                token_idx,
                results_restoration[token_idx]
            )
        )

    save_results_to_csv(handler.cfg.generation.filename.format(handler.cfg.model.name.replace("/", "-")), ["run_number", "prompt_num", "clean", "corrupted", "restored_token", "restored"], results)
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

    # print(f"{prompt} | {prompt_dict.subject} | {prompt_dict.target_true['str']}")
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

    total = 0
    failed = 0
    for prompt_dict in tqdm(df_dataset.itertuples()):
        if total - failed >= handler.cfg.generation.num_of_runs:
            break
        total += 1

        # Select only prompts that start with the subject due to tokenization problems
        prompt_ids, subject_position = preprocess_prompt(handler, prompt_dict)
        res = causal_trace_single_run(total-failed, prompt_dict.Index, handler, prompt_ids, subject_position, prompt_dict.target_true["str"])
        
        # Clean run generated wrong token
        if res == 1:
            failed += 1

    print(f"Total prompts processed: {total} failed attempts: {failed}")

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
