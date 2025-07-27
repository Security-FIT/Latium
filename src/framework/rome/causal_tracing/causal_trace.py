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

import datetime
import sys
import os
from typing import Any, Optional
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import logging
import csv

# Register parent directory for module imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from handlers.handlers import MODEL_REGISTRY


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

    LOGGER.debug(handler.model)

    tokenizer = handler.tokenizer
    
    prompt_text: str = cfg.generation.prompt
    max_new_tokens: int = cfg.generation.max_new_tokens
    LOGGER.debug(f"Preparing prompt: '{prompt_text}'")
    input_ids = prepare_prompt(tokenizer, prompt_text, handler.device)
    LOGGER.debug(f"Generating {max_new_tokens} tokens...")

    # prompt_text = ["The Space Needle", "The Eiffel Tower"]
    # input_ids = []
    # for p in prompt_text:
    #     max_new_tokens: int = cfg.generation.max_new_tokens
    #     LOGGER.debug(f"Preparing prompt: '{prompt_text}'")
    #     input_ids.append(prepare_prompt(tokenizer, p, handler.device))
    #     LOGGER.debug(f"Generating {max_new_tokens} tokens...")
    
    # handler.compute_embedding_std(input_ids)
    # exit()

    corrupt_att=True # TODO: Move to config

    corrupted_token_idx = cfg.generation.corrupted_token_idx

    for run_number in range(cfg.generation.num_of_runs):
        results = []

        # Clean run: no corruption
        decomposed_outputs_clean = handler.predict_next_token_decomposed(
            input_ids
        )
        LOGGER.info(f"Clean run prediction: {tokenizer.decode(decomposed_outputs_clean['next_token_id'][0])}")

        # Corrupted run: inject noise at specified layer/token
        decomposed_outputs_corrupted = handler.predict_next_token_decomposed(
            prompt=input_ids, 
            corruption_function=embedding_fn_corrupted, 
            corruption_token_idx=corrupted_token_idx,
            corrupt_att=corrupt_att
        )
        LOGGER.info(f"Corrupted run prediction: {tokenizer.decode(decomposed_outputs_corrupted['next_token_id'][0])}")

        # Restoration runs: restore clean activations at each layer after corruption
        results_restoration = []
        for token_idx in corrupted_token_idx:
            for restoration_layer_idx in range(23):
                decomposed_outputs_restoration = handler.predict_next_token_decomposed(
                    prompt=input_ids, 
                    corruption_function=embedding_fn_corrupted, 
                    corruption_token_idx=corrupted_token_idx,
                    restoration_layer_idx=restoration_layer_idx, 
                    restoration_token_idx=token_idx, 
                    restoration_point=decomposed_outputs_clean[f"block_{restoration_layer_idx+1}_{'attn' if corrupt_att else 'mlp'}_output"], 
                    corrupt_att=corrupt_att
                )
                results_restoration.append(tokenizer.decode(decomposed_outputs_restoration['next_token_id'][0]))
                LOGGER.info(f"Restoration run on layer {restoration_layer_idx}, token {token_idx} prediction: {tokenizer.decode(decomposed_outputs_restoration['next_token_id'][0])}")
        
            results.append(
                (
                    run_number,
                    tokenizer.decode(decomposed_outputs_clean['next_token_id'][0]), 
                    tokenizer.decode(decomposed_outputs_corrupted['next_token_id'][0]), 
                    token_idx,
                    results_restoration
                )
            )
        save_results_to_csv(cfg.generation.filename, ["run_number", "clean", "corrupted", "restored_token", "restored"], results)

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

# def main_data_collection_process(output_csv_filename="string_results_with_run_id.csv", num_runs=5):
#     """
#     Main function to orchestrate the data generation and saving process.
#     """
#     # Define your CSV header
#     csv_header = ["Run_ID", "Result_String"]

#     # Optional: Remove previous output file for a clean run if you want to start fresh
#     if os.path.exists(output_csv_filename):
#         os.remove(output_csv_filename)
#         print(f"Removed existing file: {output_csv_filename}")

#     for run_id in range(1, num_runs + 1): # Simple index for run ID starting from 1
#         print(f"\n--- Processing Run ID: {run_id} ---")
#         # Simulate getting results for this run
#         current_run_results = generate_sample_string_data(run_id, num_items=2) # Get 2 items per run for variety

#         # Prepare the data for CSV, including the run_id
#         rows_to_write = []
#         for result_str in current_run_results:
#             rows_to_write.append([run_id, result_str]) # Each row includes the run_id

#         # Save the results. Use 'a' mode to append, 'w' for the first run if you want
#         # to ensure header is written only once.
#         save_results_to_csv(output_csv_filename, csv_header, rows_to_write, mode='a')

#     print(f"\nAll runs completed. Final results in '{output_csv_filename}'.")

# if __name__ == "__main__":
#     main_data_collection_process(num_runs=7) # Run the process for 7 simulated runs
