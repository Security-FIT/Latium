"""
Model, dataset, and small inference helpers.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import logging
import os
from typing import Any

import datasets
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.linalg import CUDAMode, DeviceManager, check_device, gpu_count
from src.runtime import get_runtime, runtime_from_cfg


LOGGER = logging.getLogger(__name__)

DTYPE_PICKER = {
    "auto": "auto",
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
}


def check_hf_token(token: str | None = None) -> None:
    hf_token = token or get_runtime().hf_token
    if not hf_token:
        LOGGER.warning(
            "No HuggingFace token configured. Set runtime.hf_token if private or rate-limited Hub access is required."
        )


def load_pretrained(cfg: DictConfig) -> Any:
    """Return a loaded model and tokenizer."""
    runtime = runtime_from_cfg(cfg)
    hf_token = runtime.hf_token
    check_hf_token(hf_token)
    model_name = cfg.model.name
    save_to_local = getattr(cfg.model, "save_to_local", False)
    device = getattr(cfg.model, "device", "cuda")
    cuda_mode = getattr(cfg.model, "cuda_mode", CUDAMode.SOFT)
    dtype = DTYPE_PICKER.get(getattr(cfg.model, "dtype", "auto"), "auto")

    device = check_device(device)
    device_manager = DeviceManager(device, cuda_mode)

    multi_gpu = getattr(cfg.model, "multi_gpu", "auto")
    n_gpus = gpu_count()
    use_device_map = (multi_gpu == "auto" and n_gpus > 1 and device != "cpu") or (
        multi_gpu is True or str(multi_gpu).lower() == "true"
    )
    if use_device_map:
        LOGGER.info("Multi-GPU enabled: distributing model across %d GPUs", n_gpus)

    models_dir = getattr(
        cfg.model,
        "models_dir",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "./models"),
    )
    local_model_path = os.path.abspath(os.path.join(models_dir, model_name))

    def _ensure_padding(tok):
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            elif tok.eos_token_id is not None:
                tok.pad_token_id = tok.eos_token_id
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        return tok

    def _tokenizer_is_usable(tok) -> bool:
        try:
            encoded = tok("The", return_tensors="pt")
            input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
            if input_ids is None:
                return False
            return int(input_ids.numel()) > 0 and int(input_ids.shape[-1]) > 0
        except Exception:
            return False

    def _model_from_pretrained(path_or_name: str, **kwargs):
        try:
            return AutoModelForCausalLM.from_pretrained(
                path_or_name,
                torch_dtype=dtype,
                **kwargs,
            )
        except TypeError as exc:
            if "torch_dtype" not in str(exc):
                raise
            LOGGER.debug(
                "Transformers does not accept torch_dtype; retrying with dtype",
                exc_info=True,
            )
            return AutoModelForCausalLM.from_pretrained(
                path_or_name,
                dtype=dtype,
                **kwargs,
            )

    if os.path.exists(local_model_path):
        LOGGER.info("Loading model from local cache: %s", local_model_path)
        if use_device_map:
            model = _model_from_pretrained(local_model_path, device_map="auto")
        else:
            model = _model_from_pretrained(local_model_path)
            model = device_manager.safe_to_device(model)
        device_manager.register_object(model)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        tokenizer = _ensure_padding(tokenizer)

        if not _tokenizer_is_usable(tokenizer):
            LOGGER.warning(
                "Local tokenizer at %s appears invalid. Reloading tokenizer from '%s'.",
                local_model_path,
                model_name,
            )

            recovered = None
            last_error = None
            for kwargs in ({"local_files_only": True}, {}):
                if hf_token:
                    kwargs = {**kwargs, "token": hf_token}
                try:
                    candidate = AutoTokenizer.from_pretrained(model_name, **kwargs)
                    candidate = _ensure_padding(candidate)
                    if _tokenizer_is_usable(candidate):
                        recovered = candidate
                        break
                except Exception as exc:
                    last_error = exc

            if recovered is None:
                if last_error is not None:
                    raise RuntimeError(
                        f"Failed to recover usable tokenizer for {model_name} from local cache or Hub"
                    ) from last_error
                raise RuntimeError(f"Failed to recover usable tokenizer for {model_name}")

            tokenizer = recovered
            try:
                tokenizer.save_pretrained(local_model_path)
            except Exception as exc:
                LOGGER.warning(
                    "Could not persist recovered tokenizer to %s: %s",
                    local_model_path,
                    exc,
                )
    else:
        LOGGER.info("Downloading model from HuggingFace Hub: %s", model_name)
        if use_device_map:
            kwargs = {"device_map": "auto"}
            if hf_token:
                kwargs["token"] = hf_token
            model = _model_from_pretrained(model_name, **kwargs)
        else:
            kwargs = {}
            if hf_token:
                kwargs["token"] = hf_token
            model = _model_from_pretrained(model_name, **kwargs)
            model = device_manager.safe_to_device(model)
        device_manager.register_object(model)
        tokenizer_kwargs = {"token": hf_token} if hf_token else {}
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        tokenizer = _ensure_padding(tokenizer)
        if save_to_local:
            os.makedirs(local_model_path, exist_ok=True)
            model.save_pretrained(local_model_path)
            tokenizer.save_pretrained(local_model_path)

    LOGGER.info("Model loaded on device: %s", model.device)
    return model, tokenizer


def load_dataset_config(cfg: DictConfig, name: str, config: dict) -> Any:
    """Return a loaded dataset config, using local disk cache when available."""
    runtime = runtime_from_cfg(cfg)
    check_hf_token(runtime.hf_token)
    save_to_local = True

    datasets_dir = getattr(cfg.dataset, "datasets_dir", None)
    if datasets_dir is None:
        LOGGER.warning("No datasets_dir specified in config. Ignoring dataset.")
        return None

    local_dataset_path = os.path.abspath(os.path.join(datasets_dir, name))

    if os.path.exists(local_dataset_path):
        dataset = datasets.load_from_disk(local_dataset_path)
    else:
        kwargs = {"token": runtime.hf_token} if runtime.hf_token else {}
        dataset = datasets.load_dataset(name, config, **kwargs)
        if save_to_local:
            os.makedirs(local_dataset_path, exist_ok=True)
            dataset.save_to_disk(local_dataset_path)

    return dataset


def load_dataset(cfg: DictConfig, sm: bool = False) -> Any:
    """Return a loaded dataset, using local disk cache when available."""
    runtime = runtime_from_cfg(cfg)
    check_hf_token(runtime.hf_token)

    cfg_dataset = cfg.dataset_facts if not sm else cfg.dataset_sm
    dataset_name = cfg_dataset.name
    save_to_local = cfg_dataset.save_to_local

    datasets_dir = cfg_dataset.datasets_dir
    local_dataset_path = os.path.abspath(os.path.join(datasets_dir, dataset_name))
    config_name = getattr(cfg_dataset, "config_name", None)

    if os.path.exists(local_dataset_path):
        dataset = datasets.load_from_disk(local_dataset_path)
    else:
        dataset = (
            datasets.load_dataset(
                dataset_name,
                config_name,
                **({"token": runtime.hf_token} if runtime.hf_token else {}),
            )
            if config_name
            else datasets.load_dataset(
                dataset_name,
                **({"token": runtime.hf_token} if runtime.hf_token else {}),
            )
        )
        if save_to_local:
            os.makedirs(local_dataset_path, exist_ok=True)
            dataset.save_to_disk(local_dataset_path)

    if getattr(cfg_dataset, "concat_splits", None):
        try:
            dataset = datasets.concatenate_datasets([dataset[split] for split in cfg_dataset.concat_splits])
        except KeyError:
            LOGGER.warning(
                "One or more splits in %s not found. Returning original dataset.",
                cfg_dataset.concat_splits,
            )

    return dataset


def logits_to_log_probs(logits: torch.Tensor, token_idx: int):
    """Return the selected token log probability from final-position logits."""
    return torch.log_softmax(logits[:, -1, :], dim=1)[0][token_idx]


def logits_to_probs(logits: torch.Tensor, token_idx: int):
    """Return the selected token probability from final-position logits."""
    return torch.softmax(logits[:, -1, :], dim=1)[0][token_idx]


def sample(logits: torch.Tensor) -> int:
    """Return the most probable token ID."""
    return torch.argmax(logits, dim=1)


def get_cuda_usage(dev: str = "cuda:0") -> float:
    """Return CUDA memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0
    device = torch.device(dev)
    try:
        free, total = torch.cuda.mem_get_info(device)
    except RuntimeError:
        return 0.0
    return (total - free) / 1024**2


def print_modules(model: Any) -> None:
    """Print module names from a Transformers model."""
    for name, _ in model.named_modules():
        print(name)
