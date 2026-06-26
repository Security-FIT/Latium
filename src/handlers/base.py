"""
common.py
===========

Model handler classes and registry for the LLM framework.
Provides a registry and base class for implementing token-by-token generation for different model architectures.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>
:author: Matej Olexa <olexa.matej@gmail.com>

This module defines the model handler registry and base class for LLMs.
Handlers provide decomposed token prediction, as well as support for interventions and restoration experiments.

Typical usage example::

    handler = MODEL_REGISTRY["gpt2"](cfg)
    output = handler.predict_next_token(...)
"""

import hydra
from omegaconf import DictConfig
import torch
from typing import List

from src.common.linalg import CUDAMode, DeviceManager
from src.common.loading import load_pretrained


import logging

LOGGER = logging.getLogger(__name__)


class BaseHandler:
    """
    Abstract base class for model handlers in the LLM framework.

    Subclasses must implement predict_next_tokens and predict_next_token_decomposed for their specific model architecture.

    Args:
        cfg: Configuration object containing model and generation parameters.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the model handler by loading the model and tokenizer according to the config.

        Args:
            cfg: Configuration object.
        """
        self.cfg = cfg
        self.model, self.tokenizer = load_pretrained(cfg)

        # Initialize DeviceManager for CUDA-safe operations
        device = getattr(cfg.model, "device", "cuda")
        cuda_mode = getattr(cfg.model, "cuda_mode", CUDAMode.SOFT)
        self.device_manager = DeviceManager(device, cuda_mode)
        self.device = self.device_manager.get_device()

        # Used by tokenize_prompt fallback logging when padding token is missing.
        self.info_issued = False

        self.model.eval()

    def tokenize_prompt(
        self, prompt_text: str | List[str], apply_template: bool = False, think: bool = False
    ) -> torch.Tensor:
        """
        Tokenize the prompt and move it to the specified device.

        Args:
            prompt_text: Text to tokenize.
            apply_template: Whether to apply the tokenizer chat template.
            think: Whether to enable thinking mode when applying the chat template.

        Returns:
            The tokenized prompt as a tensor.
        """
        if apply_template:
            try:
                prompt = [{"role": "user", "content": prompt_text}]
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True, enable_thinking=think
                )
                print(prompt_text)
            except Exception as e:
                print(e)

        try:
            inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True)
        except ValueError:
            if type(prompt_text) is list:
                if self.info_issued == False:
                    LOGGER.warning("Tokenizer is probably missing padding token. Using only the first prompt.")
                    self.info_issued = True
                inputs = self.tokenizer(prompt_text[0], return_tensors="pt")
            else:
                inputs = self.tokenizer(prompt_text, return_tensors="pt")

        inputs = self.device_manager.safe_to_device(inputs, device=self.device)

        return inputs
