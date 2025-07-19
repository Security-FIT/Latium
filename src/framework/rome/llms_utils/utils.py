"""
utils.py
========

Utility functions for the LLM framework, including model loading and other helpers.

:copyright: 2025 Jakub Res
:license: MIT
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Jakub Res
# License: MIT
#
# File: utils.py
# Description: Basic utility functions for common operations with HF Transformers library-loaded LLMs
#
# Author: Jakub Res iresj@fit.vut.cz


import os
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
import torch
from torch import Tensor
import logging

def setup_logger(cfg):
    log_level = getattr(cfg, "log_level", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(numeric_level)
    return logger

def load_pretrained(cfg: DictConfig) -> Any:
    """
    Return a loaded model and tokenizer.
    The function automatically scans local model cache 
    to effectively reuse the previously saved models.

    :param cfg: Mandatory config
    :type cfg: DictConf
    :return: Loaded model and tokenizer on preffered device
    :rtype: Any
    """
    model_name = cfg.model.name
    save_to_local = getattr(cfg.model, "save_to_local", False)
    device = getattr(cfg.model, "device", "cuda")
    
    models_dir = getattr(cfg.model, "models_dir", os.path.join(os.path.dirname(__file__), "./models"))
    local_model_path = os.path.join(models_dir, model_name)
    local_model_path = os.path.abspath(local_model_path)
    
    if os.path.exists(local_model_path):
        model = AutoModelForCausalLM.from_pretrained(local_model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    else:
        # Model not present locally, download from HuggingFace Hub
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if save_to_local:
            os.makedirs(local_model_path, exist_ok=True)
            model.save_pretrained(local_model_path)
            tokenizer.save_pretrained(local_model_path)
            
    return model, tokenizer

def predict_next_tokens(model, tokenizer, prompt, num_of_tokens: int = 1) -> Tensor:
    """
    Generates the next token(s) for a given prompt using the provided model and tokenizer.

    This function takes a model, tokenizer, and a prompt (as input tensor), and generates
    a specified number of next tokens by autoregressively sampling from the model's output.
    The function appends each generated token to the prompt and stops early if the EOS token
    is produced.

    :param model: The language model used for generation (e.g., from HuggingFace Transformers).
    :type model: torch.nn.Module
    :param tokenizer: The tokenizer corresponding to the model.
    :type tokenizer: transformers.PreTrainedTokenizer
    :param prompt: The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
    :type prompt: torch.Tensor
    :param num_of_tokens: Number of tokens to generate. Defaults to 1.
    :type num_of_tokens: int, optional
    :return: The prompt tensor with the generated tokens appended.
    :rtype: torch.Tensor
    """

    for i in range(num_of_tokens):
        with torch.no_grad():
            outputs = model(prompt)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            prompt = torch.cat([prompt, next_token_id], dim=1)

            # Stop if EOS token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return prompt

# Export MODEL_REGISTRY for use in other modules