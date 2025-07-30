"""
handlers.py
===========

Model handler classes and registry for the LLM framework.
Provides a registry and base class for implementing token-by-token generation for different model architectures.

:copyright: 2025 Jakub Res
:license: MIT

This module defines the model handler registry and base class for LLMs, as well as concrete handlers for GPT2 and Llama architectures.
Handlers provide stepwise and decomposed token prediction, as well as support for interventions and restoration experiments.

Typical usage example::

    handler = MODEL_REGISTRY["gpt2"](cfg)
    output = handler.predict_next_token_decomposed(prompt, corrupt_function, corrupted_layer_idx, corrupted_token_idx)

"""
import torch
from typing import Any, Callable, Dict, Optional, Type
from .utils import load_pretrained, load_dataset
import logging


MODEL_REGISTRY: Dict[str, Type["BaseModelHandler"]] = {}
LOGGER = logging.getLogger(__name__)


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
    def __init__(self, cfg: Any) -> None:
        """
        Initialize the model handler by loading the model and tokenizer according to the config.

        :param cfg: The configuration object.
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.model, self.tokenizer = load_pretrained(cfg)
        self.dataset = load_dataset(cfg)
        self.device = getattr(cfg.model, "device", "cpu")
        self.model.eval()

    def predict_next_tokens(self, prompt: torch.Tensor, num_of_tokens: int = 1) -> torch.Tensor:
        """
        Generate the next token(s) for a given prompt.

        :param prompt: The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
        :type prompt: torch.Tensor
        :param num_of_tokens: Number of tokens to generate. Defaults to 1.
        :type num_of_tokens: int, optional
        :return: The prompt tensor with the generated tokens appended.
        :rtype: torch.Tensor
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError

    def predict_next_token_decomposed(self, prompt: torch.Tensor, corrupt_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, corrupted_layer_idx: Optional[int] = None, corrupted_token_idx: Optional[Any] = None, restoration_point: Optional[torch.Tensor] = None) -> Dict[str, Any]:
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

    def _stepwise_loop(self, prompt: torch.Tensor, num_of_tokens: int, block_fn: Callable, final_fn: Callable, tokenizer: Any, corrupted_block_idx: Optional[int] = None) -> torch.Tensor:
        """
        Shared stepwise loop for block-by-block token generation.
        Calls block_fn for each block and final_fn for the final output.

        :param prompt: The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
        :type prompt: torch.Tensor
        :param num_of_tokens: Number of tokens to generate. Defaults to 1.
        :type num_of_tokens: int
        :param block_fn: Function to process input_ids through model blocks.
        :type block_fn: Callable
        :param final_fn: Function to process hidden states to logits.
        :type final_fn: Callable
        :param tokenizer: The tokenizer instance for EOS detection.
        :type tokenizer: transformers.PreTrainedTokenizer
        :param corrupted_block_idx: Index of block to corrupt (optional).
        :type corrupted_block_idx: int, optional
        :return: The prompt tensor with the generated tokens appended.
        :rtype: torch.Tensor
        """
        for _ in range(num_of_tokens):
            hidden_states = block_fn(prompt, corrupted_block_idx)
            logits = final_fn(hidden_states)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            prompt = torch.cat([prompt, next_token_id], dim=1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
        return prompt