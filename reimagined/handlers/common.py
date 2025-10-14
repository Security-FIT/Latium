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

from torch.nn.modules import module
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

class BaseModelHandlerDecomposed:
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
        self.emb_shape = self.model.config.n_embd
        self.device = getattr(cfg.model, "device", "cpu")
        self.model.eval()

    def predict_next_token(self, 
            prompt: torch.Tensor,
            corruption_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            corruption_token_idx: Optional[list] = None,
            restoration_layer_idx: Optional[int] = None,
            restoration_token_idx: Optional[int] = None,
            restoration_point: Optional[torch.Tensor] = None,
            corrupt_att: Optional[bool] = False    
        ) -> Dict[str, Any]:
        """
        Generate the next token for a given prompt, returning a detailed decomposition of intermediate states.
        Optionally applies corruption or restoration at a specified layer and token index.

        :param prompt: The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
        :type prompt: torch.Tensor
        :param corrupt_function: Function to apply to corrupt hidden states (optional).
        :type corrupt_function: Callable[[torch.Tensor], torch.Tensor], optional
        :param corrupted_layer_idx: Index of the layer to corrupt (optional).
        :type corrupted_layer_idx: int, optional
        :param corrupted_token_idx: Index/indices of the token(s) to corrupt (optional).
        :type corrupted_token_idx: Any, optional
        :param restoration_point: Hidden state tensor to restore at a given layer (optional).
        :type restoration_point: torch.Tensor, optional
        :return: A dictionary containing intermediate model states.
        :rtype: Dict[str, Any]
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError

    def compute_embedding_std(self, subjects: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the standard deviation in the initial embeddings. Used primarily for the corruption noise scaling.
        :param subjects: The tokenized prompts' subjects to compute the embeddings from
        :type subjects: torch.Tensor
        :return: A single item tensor with the standard deviation
        :rtype: torch.Tensor
        """
        raise NotImplementedError


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
        self.emb_shape = self.model.config.n_embd
        self.device = getattr(cfg.model, "device", "cpu")
        
        self.layer_name_template = getattr(cfg.model, "layer_name_template", None)
        self.layer = getattr(cfg.model, "layer", None)

        self.corruption_token_idx = getattr(cfg.generation, "corruption_token_idx", None)
        self.noise_multiplier = getattr(cfg.generation, "corruption_noise_multiplier", 3.0)

        self.restoration_point = None

        self.hooks = []
        self.model.eval()

    def _get_module(self, module_name: str) -> torch.nn.Module:
        """
        """
        for name, module in self.model.named_modules():
            if name == module_name:
                return  module

        raise KeyError(f"{self.layer_name} not found")

    def _register_hooks(self) -> None:
        """
        """
        # Register the corruption hook
        corrupt_module = self._get_module(self.layer_name_template.format(0))
        handle = corrupt_module.register_forward_pre_hook(self._corrupt)
        self.hooks.append(handle)

        # Register the restoration hook
        restore_module = self._get_module(self.layer_name_template.format(self.layer))
        handle = restore_module.register_forward_hook(self._restore)
        self.hooks.append(handle)

    def _remove_hooks(self) -> None:
        for handle in self.hooks:
            handle.remove()

    def _corrupt(self, hidden_states):
        hidden_states[:, self.corruption_token_idx] += torch.rand_like(hidden_states[:, self.corruption_token_idx]) * self.noise_multiplier

    def _restore(self, hidden_states):
        hidden_states[:, self.restoration_token_idx] = self.restoration_point

    def compute_embedding_std(self, subjects: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the standard deviation in the initial embeddings. Used primarily for the corruption noise scaling.
        :param subjects: The tokenized prompts' subjects to compute the embeddings from
        :type subjects: torch.Tensor
        :return: A single item tensor with the standard deviation
        :rtype: torch.Tensor
        """
        raise NotImplementedError