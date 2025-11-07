"""
common.py
===========

Model handler classes and registry for the LLM framework.
Provides a registry and base class for implementing token-by-token generation for different model architectures.

:copyright: 2025 Jakub Res
:license: MIT

This module defines the model handler registry and base class for LLMs.
Handlers provide decomposed token prediction, as well as support for interventions and restoration experiments.

Typical usage example::

    handler = MODEL_REGISTRY["gpt2"](cfg)
    output = handler.predict_next_token(...)

"""
from omegaconf import DictConfig
import torch
from typing import Any, Callable, Dict, List, Optional, Type
from pathlib import Path
from tqdm import tqdm
from reimagined.utils import load_pretrained, load_dataset
import logging


MODEL_REGISTRY: Dict[str, Type["BaseModelHandler"]] = {}
LOGGER = logging.getLogger(__name__)


def get_handler(cfg: DictConfig) -> Any:
    """
    Retrieve and instantiate the appropriate model handler based on config.

    :param cfg: The configuration object containing static hyperparameters
    :type cfg: DictConfig
    :raises ValueError: If the model type specified in the config is not registered.
    :return: An instance of the model handler.
    :rtype: BaseModelHandler
    """
    return BaseModelHandler(cfg)

    model_type: str = cfg.model.handler
    handler_cls = MODEL_REGISTRY.get(model_type)
    if handler_cls is None:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    return handler_cls(cfg)

def register_model(model_type: str) -> Callable[[Type[Any]], Type[Any]]:
    """
    Decorator to register a model handler class in the MODEL_REGISTRY.

    :param model_type: The string identifier for the model type.
    :type model_type: str
    :return: The decorator function.
    :rtype: Callable
    """
    def decorator(cls: Type[Any]) -> Type[Any]:
        MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator

class BaseModelHandler:
    """
    Abstract base class for model handlers in the LLM framework.

    Subclasses must implement the :meth:`predict_next_tokens` and :meth:`predict_next_token_decomposed` methods for their specific model architecture.

    :param cfg: The configuration object containing model and generation parameters.
    :type cfg: DictConfig
    """
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the model handler by loading the model and tokenizer according to the config.

        :param cfg: The configuration object.
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.model, self.tokenizer = load_pretrained(cfg)
        
        self.num_of_layers = self.model.config.num_hidden_layers

        self.device = getattr(cfg.model, "device", "cpu")
        
        self._layer_name_template = getattr(cfg.model, "layer_name_template", None)
        self._layer = getattr(cfg.model, "layer", None)

        self.emb_shape = min(self._get_module(self._layer_name_template.format(self._layer)).weight.shape)

        self.second_moment_dir = getattr(cfg.model, "second_moment_dir", "./second_moment_stats")

        # Causal trace
        self._noise = None
        self._corrupt_idx = None
        self._noise_multiplier = getattr(cfg.model, "corruption_noise_multiplier", None)
        self._corrupt_layer_name_template = getattr(cfg.model, "corrupt_layer_name_template", None)

        self._restore_layer_name_template = getattr(cfg.model, "restore_layer_name_template", None)
        self._restore_point = None
        self._restore_idx = None
        self._restore_layer = 0

        # Weight intervention
        self._k_accumulator = []
        self.v = None
        self.delta = torch.zeros((self.emb_shape), requires_grad=True, device=self.device)

        # Embeddings
        self._emb_accumulator = []

        # Hook flags
        self._hooks = []
        self._restore_hooks = []
        self.is_corrupt_hook = False
        self.is_restore_hook = False
        self.is_k_hook = False
        self.is_delta_hook = False

        self.model.eval()

    def set_restore_layer(self, layer: int):
        self._restore_layer = layer

    def set_restore_point(self, restore_point):
        self._restore_point = restore_point

    def set_corrupt_idx(self, idx: List[int]) -> None:
        self._corrupt_idx = idx

    def set_restore_idx(self, idx: List[int]) -> None:
        self._restore_idx = idx

    def set_corrupt_hook(self) -> None:
        self.is_corrupt_hook = True

        # Register the corruption hook
        corrupt_module = self._get_module(self._corrupt_layer_name_template.format(0))
        handle = corrupt_module.register_forward_hook(self._corrupt_hook)
        self._hooks.append(handle)

    def set_restore_hook(self) -> None:
        self.is_restore_hook = True

        # Register the restoration hook
        restore_module = self._get_module(self._restore_layer_name_template.format(self._restore_layer))
        handle = restore_module.register_forward_hook(self._restore_hook)
        self._restore_hooks.append(handle)

    def unset_restore_hook(self) -> None:
        for handle in self._restore_hooks:
            handle.remove()

    def set_k_hook(self):
        self.is_k_hook = True

        # Register the corruption hook
        k_module = self._get_module(self._layer_name_template.format(self._layer))
        handle = k_module.register_forward_pre_hook(self._gather_k_hook)
        self._hooks.append(handle)

    def set_v_hook(self):
        self.is_v_hook = True

        # Register the corruption hook
        v_module = self._get_module(self._layer_name_template.format(self._layer))
        handle = v_module.register_forward_hook(self._gather_v_hook)
        self._hooks.append(handle)

    def set_delta_hook(self):
        self.is_delta_hook = True

        # Register the corruption hook
        delta_module = self._get_module(self._layer_name_template.format(self._layer))
        handle = delta_module.register_forward_hook(self._delta_hook)
        self._hooks.append(handle)

    def set_emb_hook(self):
        # Register the corruption hook
        emb_module = self._get_module(self._corrupt_layer_name_template.format(0))
        handle = emb_module.register_forward_hook(self._emb_hook)
        self._hooks.append(handle)

    def remove_hooks(self) -> None:
        """
        Removes all hooks from the model and cleans handler accumulators and delta
        """
        self._noise = None
        self._k_accumulator = []
        self._emb_accumulator = []
        self.delta = torch.zeros((self.emb_shape), requires_grad=True, device=self.device)
        for handle in self._hooks:
            handle.remove()
        
        for handle in self._restore_hooks:
            handle.remove()

    def tokenize_prompt(self, prompt_text: str | List[str]) -> torch.Tensor:
        """
        Tokenize the prompt and move it to the specified device.

        :param prompt_text: The text to be tokenized.
        :type prompt_text: str
        :return: The tokenized prompt as a tensor.
        :rtype: torch.Tensor
        """
        try:
            inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True).to(self.device)
        except ValueError:
            if type(prompt_text) is list:
                LOGGER.warning("Tokenizer is probably missing padding token. Using only the first prompt.")
                inputs = self.tokenizer(prompt_text[0], return_tensors="pt").to(self.device)
            else:
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        return inputs

    def _get_module(self, module_name: str) -> torch.nn.Module:
        """
        """
        for name, module in self.model.named_modules():
            if name == module_name:
                return  module

        raise KeyError(f"{module_name} not found")

    def register_casual_hooks(self) -> None:
        """
        """
        self.set_corrupt_hook()
        self.set_restore_hook()

    def _corrupt_hook(self, module, input, output):
        if self._noise == None:
            self._noise = torch.randn_like(output[0][0, :]) * self._noise_multiplier
        for token_idx in self._corrupt_idx:
            output[0][token_idx, :] += self._noise
        return output

    def _restore_hook(self, module, input, output):
        try:
            output[0][:,self._restore_idx] = self._restore_point
        except Exception as e:
            LOGGER.warning(f"Hidden state restore failed. {e}")
        return output

    def _gather_k_hook(self, module, input):
        # This needs to be adapted for the multiprompt
        self._k_accumulator.append(input[0][-1, -1].detach())
        return input

    def _gather_v_hook(self, module, input, output):
        self.v = output[0][-1].detach()
        return output

    def _delta_hook(self, module, input, output):
        try:
            output[0][:] += self.delta
        except:
            LOGGER.warn(f"Delta dimension mismatch. Delta shape: {self.delta.shape} Emb shape: {output[0][:].shape}")
            # Fix the autodetected emb size and regenerate delta
            self.emb_shape = min(self._get_module(self._layer_name_template.format(self._layer)).weight.shape)
            self.delta = torch.zeros((self.emb_shape), requires_grad=True, device=self.device)
            output[0][:] += self.delta
        return output

    def _emb_hook(self, module, input, output):
        self._emb_accumulator.append(output[0])
        return output

    def compute_embedding_std(self, subjects: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the standard deviation in the initial embeddings. Used primarily for the corruption noise scaling.
        :param subjects: The tokenized prompts' subjects to compute the embeddings from
        :type subjects: torch.Tensor
        :return: A single item tensor with the standard deviation
        :rtype: torch.Tensor
        """
        self.set_emb_hook()
        for _, subject in tqdm(enumerate(subjects)):
            self.model(**subject)

        std = torch.cat(self._emb_accumulator).std()
        self._noise_multiplier = std.item()*3
        self.remove_hooks()
        return std
