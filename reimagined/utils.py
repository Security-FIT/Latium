"""
utils.py
========

Utility functions for the LLM framework, including model loading and other helpers.

:copyright: 2025 Jakub Res
:license: MIT
"""


import logging
import os
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
import torch
from torch import Tensor
import datasets


LOGGER = logging.getLogger(__name__)


def check_device(device: str) -> str:
    """
    Check if the device is valid and return the appropriate device.

    :param device: The device to check
    :type device: str
    :return: The appropriate device
    :rtype: str
    """
    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA is not available. Setting the device to 'cpu'.")
        device = "cpu"
    elif device == "cpu" and torch.cuda.is_available():
        LOGGER.info("CUDA is available. Consider setting the device to 'cuda'.")
    return device

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

    device = check_device(device)
    
    models_dir = getattr(cfg.model, "models_dir", os.path.join(os.path.dirname(__file__), "./models"))
    local_model_path = os.path.join(models_dir, model_name)
    local_model_path = os.path.abspath(local_model_path)
    
    if os.path.exists(local_model_path):
        model = AutoModelForCausalLM.from_pretrained(local_model_path).half().to(device)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    else:
        # Model not present locally, download from HuggingFace Hub
        model = AutoModelForCausalLM.from_pretrained(model_name).half().to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if save_to_local:
            os.makedirs(local_model_path, exist_ok=True)
            model.save_pretrained(local_model_path)
            tokenizer.save_pretrained(local_model_path)
            
    return model, tokenizer

def load_dataset(cfg: DictConfig) -> Any:
    """
    Return a loaded dataset.
    The function automatically scans local model cache 
    to effectively reuse the previously saved datasets.

    :param cfg: Mandatory config
    :type cfg: DictConf
    :return: Loaded dataset
    :rtype: Any
    """
    dataset_name = cfg.dataset.name
    save_to_local = cfg.dataset.save_to_local

    datasets_dir = cfg.dataset.datasets_dir
    local_dataset_path = os.path.join(datasets_dir, dataset_name)
    local_dataset_path = os.path.abspath(local_dataset_path)
    
    if os.path.exists(local_dataset_path):
        dataset = datasets.load_from_disk(local_dataset_path)
    else:
        # Model not present locally, download from HuggingFace Hub
        dataset = datasets.load_dataset(dataset_name)
        if save_to_local:
            os.makedirs(local_dataset_path, exist_ok=True)
            dataset.save_to_disk(local_dataset_path)
            
    return dataset

def logits_to_log_probs(logits: torch.Tensor, token_idx: int) -> float:
    """
    Convert logits from final layer to probabilities and returns the probability of specific token

    :param logits: The tensor containing logits from all model layers
    :type logits: torch.Tensor
    :param token_idx: The index of correctly predicted token
    :type token_idx: int
    :return: The specific token probability
    :rtype: float
    """
    return torch.log_softmax(logits[:, -1, :], dim=1)[0][token_idx]

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
    return torch.softmax(logits[:, -1, :], dim=1)[0][token_idx]

def sample(logits: torch.Tensor) -> int:
    """
    Sample the most probable token from logits tensor.

    :param logits: The tensor containing the final logits
    :type logits: torch.Tensor
    :return: Token ID
    :rtype: int
    """
    return torch.argmax(logits, dim=1)

def get_cuda_usage(dev: str = 'cuda:0') -> float:
    """
    Get the usage of the specified CUDA device

    :param dev: The device to get the usage of (e.g., 'cuda:0').
    :type dev: str
    :return: The usage of the specified CUDA device.
    :rtype: float
    """
    return 0
    device = torch.device(dev)
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1024 ** 2

def print_modules(model: Any) -> None:
    """
    Prints names of the modules from provided model.

    :param model: Transformer library model
    :type model: Any
    :return: None
    """

    for name, _ in model.named_modules():
        print(name)
