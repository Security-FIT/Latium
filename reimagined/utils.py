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
import datasets


LOGGER = logging.getLogger(__name__)


class CUDAMode:
    """CUDA device modes"""

    NONE = "none"  # cpu only
    SOFT = "soft"  # cuda until OOM, then CPU
    GREEDY = "greedy"  # cuda always, retry on OOM with cache clear
    STRICT = "strict"  # cuda or exit on OOM


class DeviceManager:
    """
    Manages device operations with different CUDA modes

    - none: Force CPU only
    - soft: CUDA until first OOM error, then permanently switch to CPU
    - greedy: Always try CUDA, clear cache and retry on OOM
    - strict: CUDA or exit with error on OOM
    """

    # Global state to track the model and CUDA status
    _registered_model = None
    _cuda_disabled = False

    def __init__(self, preferred_device: str = "cuda", cuda_mode: str = CUDAMode.SOFT):
        self.preferred_device = preferred_device
        self.cuda_mode = cuda_mode
        self._oom_count = (
            0  # Incremental count of OOM occurrences for logging in greedy mode
        )

    def register_object(self, obj: Any) -> None:
        """Register the model to be moved to CPU if CUDA is disabled."""
        DeviceManager._registered_model = obj

    def get_device(self) -> str:
        """Get the current active device"""
        if self.cuda_mode == CUDAMode.NONE:
            return "cpu"
        if self.cuda_mode == CUDAMode.SOFT and DeviceManager._cuda_disabled:
            return "cpu"
        return self.preferred_device

    def clear_cache(self) -> None:
        """Clear CUDA cache if available"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def safe_to_device(self, data: Any, device: str = None) -> Any:
        """Safely move tensor or model to device with OOM handling"""
        # TODO: check if we need to separately handle models vs tensors inside data
        target_device = device or self.get_device()

        try:
            return data.to(target_device)
        except torch.cuda.OutOfMemoryError as e:
            return self._handle_oom(data, target_device, e)

    def _handle_oom(self, data: Any, device: str, error: Exception) -> Any:
        """Handle OOM error based on CUDA mode"""
        self._oom_count += 1

        if self.cuda_mode == CUDAMode.STRICT:
            LOGGER.error(f"CUDA OOM Error #{self._oom_count} in strict mode")
            LOGGER.error(f"Error details: {str(error)}")
            raise SystemExit(
                "CUDA OOM Error in strict mode. Cannot continue."
            ) from error
        elif self.cuda_mode == CUDAMode.SOFT:
            LOGGER.error(f"CUDA OOM Error #{self._oom_count} in soft mode")
            LOGGER.warning("Permanently switching to CPU for the rest of operations")
            DeviceManager._cuda_disabled = True
            self.clear_cache()

            # Move the registered model to CPU
            if DeviceManager._registered_model is not None:
                if hasattr(DeviceManager._registered_model, "to"):
                    try:
                        DeviceManager._registered_model.to("cpu")
                        LOGGER.info(
                            f"Moved model {type(DeviceManager._registered_model).__name__} to CPU"
                        )
                    except Exception as e:
                        LOGGER.error(f"Failed to move model to CPU: {e}")

            return data.to("cpu")
        elif self.cuda_mode == CUDAMode.GREEDY:
            LOGGER.warning(f"CUDA OOM Error #{self._oom_count} in greedy mode")
            LOGGER.info("Clearing CUDA cache and retrying...")
            self.clear_cache()

            try:
                return data.to(device)
            except torch.cuda.OutOfMemoryError:
                # TODO: maybe implement a retry limit?
                LOGGER.error(
                    "CUDA OOM persists after cache clear. Falling back to CPU for this operation."
                )
                return data.to("cpu")
        # Fallback, should not reach here
        else:
            LOGGER.error(
                f"Unknown CUDA mode '{self.cuda_mode}'. Cannot handle OOM error."
            )
            raise SystemExit("Unknown CUDA mode. Cannot continue.") from error


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


dtype_picker = {
    "auto": "auto",
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
}


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
    cuda_mode = getattr(cfg.model, "cuda_mode", CUDAMode.SOFT)
    dtype = dtype_picker.get(getattr(cfg.model, "dtype", "auto"), "auto")

    device = check_device(device)
    device_manager = DeviceManager(device, cuda_mode)

    models_dir = getattr(
        cfg.model, "models_dir", os.path.join(os.path.dirname(__file__), "./models")
    )
    local_model_path = os.path.join(models_dir, model_name)
    local_model_path = os.path.abspath(local_model_path)

    if os.path.exists(local_model_path):
        model = AutoModelForCausalLM.from_pretrained(local_model_path, dtype=dtype)
        model = device_manager.safe_to_device(model)
        device_manager.register_object(model)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    else:
        # Model not present locally, download from HuggingFace Hub
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model = device_manager.safe_to_device(model)
        device_manager.register_object(model)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if save_to_local:
            os.makedirs(local_model_path, exist_ok=True)
            model.save_pretrained(local_model_path)
            tokenizer.save_pretrained(local_model_path)

    LOGGER.info(f"Model loaded on device: {model.device}")
    print(f"Model is on: {model.device}")

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


def get_cuda_usage(dev: str = "cuda:0") -> float:
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
    return (total - free) / 1024**2


def print_modules(model: Any) -> None:
    """
    Prints names of the modules from provided model.

    :param model: Transformer library model
    :type model: Any
    :return: None
    """

    for name, _ in model.named_modules():
        print(name)
