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
import datetime
import os
from typing import Any
import torch
import hydra
from omegaconf import DictConfig
import csv

from tqdm import tqdm

from src.handlers.rome import ModelHandler
from src.utils import load_dataset, logits_to_probs, sample


# Globals
import logging
LOGGER = logging.getLogger(__name__)
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
    handler = ModelHandler(cfg)
    dataset = load_dataset(cfg)
    df_dataset = filter_dataset(dataset["requested_rewrite"])

    input_ids = []
    prompts = []
    for prompt_dict in df_dataset.itertuples():
        if prompt_dict.Index == handler.cfg.generation.num_of_runs:
            break
        prompts.append(prompt_dict.prompt.format(prompt_dict.subject))
    
    total = len(prompts)
    start_idx = 0
        
    while total-handler.batch_size > 0:
        input_ids.append(handler.tokenize_prompt(prompts[start_idx:start_idx+handler.batch_size]))
        total -= handler.batch_size
        start_idx += handler.batch_size
    
    handler.compute_embedding_std(input_ids)
    return handler._noise_multiplier # TODO: move constant into the model config

def causal_trace_single_run(
        run_number: int,
        prompt_number: int,
        handler: ModelHandler, 
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
    df_prompts_dataset = pandas.DataFrame(dataset)
    return df_prompts_dataset


def _find_subject_token_positions(input_ids_prompt: torch.Tensor, input_ids_subject: torch.Tensor) -> list[int]:
    """Return all token positions for exact subject-span matches in the prompt."""
    if input_ids_prompt.dim() != 2 or input_ids_subject.dim() != 2:
        return []

    subject_len = int(input_ids_subject.size(1))
    prompt_len = int(input_ids_prompt.size(1))
    if subject_len <= 0 or prompt_len < subject_len:
        return []

    windows = input_ids_prompt.unfold(1, subject_len, 1)
    # Full-span subject match only; partial token matches create false positives.
    full_matches = (windows == input_ids_subject).all(dim=2)
    start_positions = full_matches.nonzero(as_tuple=True)[1].tolist()

    token_positions: list[int] = []
    for start in start_positions:
        token_positions.extend(range(int(start), int(start) + subject_len))

    return sorted(set(token_positions))

def _find_subject_token_positions_by_offsets(handler: ModelHandler, prompt: str, subject: str) -> list[int]:
    """Fallback: map subject character span to token indices via tokenizer offsets."""
    if not subject:
        return []

    char_start = prompt.find(subject)
    if char_start == -1:
        prefixed = f" {subject}"
        prefixed_idx = prompt.find(prefixed)
        if prefixed_idx != -1:
            # Use the subject start, not the leading space.
            char_start = prefixed_idx + 1

    if char_start == -1:
        return []

    char_end = char_start + len(subject)
    try:
        raw_tokens = handler.tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
    except Exception:
        return []

    offsets = raw_tokens.get("offset_mapping")
    if offsets is None:
        return []

    token_positions: list[int] = []
    for idx, (start, end) in enumerate(offsets[0].tolist()):
        if end <= start:
            # Skip special tokens with empty spans.
            continue
        if end > char_start and start < char_end:
            token_positions.append(int(idx))

    return sorted(set(token_positions))

def preprocess_prompt(handler, prompt_dict):
    """
    TODO
    """
    prompt = prompt_dict.prompt.format(prompt_dict.subject)
    input_ids = handler.tokenize_prompt(prompt)
    input_ids_prompt = input_ids["input_ids"]

    input_ids_subject = handler.tokenize_prompt(prompt_dict.subject)["input_ids"]
    subject_position = _find_subject_token_positions(input_ids_prompt, input_ids_subject)

    if len(subject_position) == 0:
        # The tokenizer most likely learned specific tokens with space as prefix (" Rome" instead of " " + "Rome")
        input_ids_subject = handler.tokenize_prompt(f" {prompt_dict.subject}")["input_ids"]
        subject_position = _find_subject_token_positions(input_ids_prompt, input_ids_subject)

    if len(subject_position) == 0:
        subject_position = _find_subject_token_positions_by_offsets(handler, prompt, prompt_dict.subject)
        if subject_position:
            LOGGER.info(
                "Recovered subject span via offset mapping. subject=%r prompt=%r positions=%s",
                prompt_dict.subject,
                prompt,
                subject_position,
            )

    if len(subject_position) == 0:
        LOGGER.warning(
            "Skipping prompt due to unmatched subject span. subject=%r prompt=%r",
            prompt_dict.subject,
            prompt,
        )
        return None

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
    handler = ModelHandler(cfg)
    dataset = load_dataset(cfg)
    df_dataset = filter_dataset(dataset["requested_rewrite"])

    total = 0
    failed = 0
    for prompt_dict in tqdm(df_dataset.itertuples()):
        if total - failed >= handler.cfg.generation.num_of_runs:
            break
        total += 1

        # Select only prompts that start with the subject due to tokenization problems
        preprocessed = preprocess_prompt(handler, prompt_dict)
        if preprocessed is None:
            failed += 1
            continue

        prompt_ids, subject_position = preprocessed
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
