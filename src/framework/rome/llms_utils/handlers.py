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
from .utils import load_pretrained

MODEL_REGISTRY: Dict[str, Type["BaseModelHandler"]] = {}

# from .utils import setup_logger
# LOGGER = setup_logger()

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

@register_model("gpt2")
class GPT2Handler(BaseModelHandler):
    """
    Handler for GPT2-style models in the LLM framework.

    Implements token-by-token generation using the GPT2 transformer architecture.
    """
    def predict_next_tokens(self, prompt: torch.Tensor, num_of_tokens: int = 1) -> torch.Tensor:
        """
        Generate the next token(s) for a given prompt using a GPT2-style model.

        :param prompt: The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
        :type prompt: torch.Tensor
        :param num_of_tokens: Number of tokens to generate. Defaults to 1.
        :type num_of_tokens: int, optional
        :return: The prompt tensor with the generated tokens appended.
        :rtype: torch.Tensor
        """
        device = prompt.device
        model = self.model
        tokenizer = self.tokenizer
        for i in range(num_of_tokens):
            with torch.no_grad():
                # Embedding and positional encoding
                input_embeds = model.transformer.wte(prompt)
                position_ids = torch.arange(prompt.shape[1], dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand_as(prompt)
                hidden_states = model.transformer.drop(model.transformer.wpe(position_ids) + input_embeds)
                # Pass through all transformer blocks
                for block in model.transformer.h:
                    hidden_states = block(hidden_states)[0]
                # Final normalization and LM head
                hidden_states = model.transformer.ln_f(hidden_states)
                logits = model.lm_head(hidden_states)
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                prompt = torch.cat([prompt, next_token_id], dim=1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        return prompt

    def predict_next_token_decomposed(self, prompt: torch.Tensor, corrupt_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, corrupted_layer_idx: Optional[int] = None, corrupted_token_idx: Optional[Any] = None, restoration_point: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Generate the next token for a given prompt, returning a detailed decomposition of intermediate states for GPT2.
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
        """
        device = prompt.device
        model = self.model
        # Initial embeddings
        position_ids = torch.arange(0, prompt.shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, prompt.shape[-1])
        token_embeds = model.transformer.wte(prompt)
        position_embeds = model.transformer.wpe(position_ids)
        hidden_states = token_embeds + position_embeds
        decomposed_outputs: Dict[str, Any] = {"initial_embedding": hidden_states.clone()}

        for layer_idx, block in enumerate(model.transformer.h):
            # Attention block
            residual = hidden_states
            # Layer norm
            hidden_states_ln1 = block.ln_1(hidden_states)
            decomposed_outputs[f"block_{layer_idx}_ln1_output"] = hidden_states_ln1.clone()
            # Multi-head attention
            attn_outputs = block.attn(hidden_states_ln1)
            attn_output = attn_outputs[0]
            # Residual connection
            hidden_states = attn_output + residual
            decomposed_outputs[f"block_{layer_idx}_attn_output"] = hidden_states.clone()
            # MLP block
            residual = hidden_states
            # Layer norm
            hidden_states_ln2 = block.ln_2(hidden_states)
            decomposed_outputs[f"block_{layer_idx}_ln2_output"] = hidden_states_ln2.clone()
            # MLP
            feed_forward_hidden_states = block.mlp(hidden_states_ln2)
            # Residual connection
            hidden_states = residual + feed_forward_hidden_states

            # Apply corruption if specified
            if corrupted_layer_idx == layer_idx:
                for token_idx in corrupted_token_idx:
                    print(f"Layer {layer_idx}, token index {token_idx} corrupted.")
                    hidden_states[:,token_idx] = corrupt_function(hidden_states[:,token_idx])

            # Apply restoration if specified
            if corrupted_layer_idx == layer_idx-1 and restoration_point is not None:
                for token_idx in corrupted_token_idx:
                    print(f"Layer {layer_idx}, token index {token_idx} restored.")
                    hidden_states[:,token_idx] = restoration_point[:,token_idx]

            decomposed_outputs[f"block_{layer_idx}_mlp_output"] = hidden_states.clone()

        # Final part
        hidden_states = model.transformer.ln_f(hidden_states)
        decomposed_outputs["final_norm_output"] = hidden_states.clone()
        logits = model.lm_head(hidden_states)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        decomposed_outputs["final_logits"] = logits
        decomposed_outputs["next_token_id"] = next_token_id
        return decomposed_outputs

    def predict_next_tokens_stepwise(self, prompt: torch.Tensor, embedding_fn: Callable[[torch.Tensor], torch.Tensor], num_of_tokens: int = 1, corrupted_block_idx: Optional[int] = None) -> torch.Tensor:
        """
        Stepwise (block-by-block) token prediction for GPT2-style models.

        This method manually passes the input through the model's embeddings and each transformer block sequentially.
        Useful for debugging, analysis, or interventions.

        :param prompt: The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
        :type prompt: torch.Tensor
        :param embedding_fn: Function to apply to hidden states for corruption/intervention.
        :type embedding_fn: Callable[[torch.Tensor], torch.Tensor]
        :param num_of_tokens: Number of tokens to generate. Defaults to 1.
        :type num_of_tokens: int, optional
        :param corrupted_block_idx: Index of block to corrupt (optional).
        :type corrupted_block_idx: int, optional
        :return: The prompt tensor with the generated tokens appended.
        :rtype: torch.Tensor
        """
        device = prompt.device
        model = self.model
        tokenizer = self.tokenizer
        def block_fn(input_ids: torch.Tensor, corrupted_block_idx: Optional[int] = None) -> torch.Tensor:
            # Embedding and positional encoding
            position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])
            token_embeds = model.transformer.wte(input_ids)
            position_embeds = model.transformer.wpe(position_ids)
            hidden_states = token_embeds + position_embeds
            # Pass through all transformer blocks
            i = 0
            for block in model.transformer.h:
                outputs = block(hidden_states)
                if corrupted_block_idx == i:
                    hidden_states = embedding_fn(outputs[0])
                else:
                    hidden_states = outputs[0]
                i += 1
            return hidden_states
        def final_fn(hidden_states: torch.Tensor) -> torch.Tensor:
            # Final normalization and LM head
            hidden_states = model.transformer.ln_f(hidden_states)
            logits = model.lm_head(hidden_states)
            return logits
        return self._stepwise_loop(prompt, num_of_tokens, block_fn, final_fn, tokenizer, corrupted_block_idx)

@register_model("llama")
class LlamaHandler(BaseModelHandler):
    """
    Handler for Llama-style models in the LLM framework.

    Implements token-by-token generation using the Llama transformer architecture.
    """
    def predict_next_tokens(self, prompt: torch.Tensor, num_of_tokens: int = 1) -> torch.Tensor:
        """
        Generate the next token(s) for a given prompt using a Llama-style model.

        :param prompt: The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
        :type prompt: torch.Tensor
        :param num_of_tokens: Number of tokens to generate. Defaults to 1.
        :type num_of_tokens: int, optional
        :return: The prompt tensor with the generated tokens appended.
        :rtype: torch.Tensor
        """
        device = prompt.device
        model = self.model
        tokenizer = self.tokenizer
        for i in range(num_of_tokens):
            with torch.no_grad():
                # Embedding and positional encoding
                input_embeds = model.model.embed_tokens(prompt)
                position_ids = torch.arange(prompt.shape[1], dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand_as(prompt)
                hidden_states = input_embeds
                # Pass through all transformer blocks
                for block in model.model.layers:
                    hidden_states = block(hidden_states, attention_mask=None)[0]
                # Final normalization and LM head
                hidden_states = model.model.norm(hidden_states)
                logits = model.lm_head(hidden_states)
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                prompt = torch.cat([prompt, next_token_id], dim=1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        return prompt

    def predict_next_token_decomposed(self, prompt: torch.Tensor) -> Dict[str, Any]:
        """
        Generate the next token for a given prompt, returning a detailed decomposition of intermediate states for Llama.

        :param prompt: The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
        :type prompt: torch.Tensor
        :return: A dictionary containing intermediate model states.
        :rtype: Dict[str, Any]
        """
        device = prompt.device
        model = self.model
        # Initial embeddings
        input_embeds = model.model.embed_tokens(prompt)
        hidden_states = input_embeds
        decomposed_outputs: Dict[str, Any] = {"initial_embedding": hidden_states.clone()}
        for i, block in enumerate(model.model.layers):
            # Attention block
            residual = hidden_states
            hidden_states_ln1 = block.input_layernorm(hidden_states)
            decomposed_outputs[f"block_{i}_ln1_output"] = hidden_states_ln1.clone()
            attn_outputs = block.self_attn(hidden_states_ln1)
            attn_output = attn_outputs[0]
            # residual connection
            hidden_states = attn_output + residual
            decomposed_outputs[f"block_{i}_attn_output"] = hidden_states.clone()
            # MLP block
            residual = hidden_states
            hidden_states_ln2 = block.post_attention_layernorm(hidden_states)
            decomposed_outputs[f"block_{i}_ln2_output"] = hidden_states_ln2.clone()
            feed_forward_hidden_states = block.mlp(hidden_states_ln2)
            # residual connection
            hidden_states = residual + feed_forward_hidden_states
            decomposed_outputs[f"block_{i}_mlp_output"] = hidden_states.clone()
        # Final part
        hidden_states = model.model.norm(hidden_states)
        decomposed_outputs["final_norm_output"] = hidden_states.clone()
        logits = model.lm_head(hidden_states)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        decomposed_outputs["final_logits"] = logits
        decomposed_outputs["next_token_id"] = next_token_id
        return decomposed_outputs

    def predict_next_tokens_stepwise(self, prompt: torch.Tensor, num_of_tokens: int = 1) -> torch.Tensor:
        """
        Stepwise (block-by-block) token prediction for Llama-style models.

        This method manually passes the input through the model's embeddings and each transformer block sequentially.
        Useful for debugging, analysis, or interventions.

        :param prompt: The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
        :type prompt: torch.Tensor
        :param num_of_tokens: Number of tokens to generate. Defaults to 1.
        :type num_of_tokens: int, optional
        :return: The prompt tensor with the generated tokens appended.
        :rtype: torch.Tensor
        """
        device = prompt.device
        model = self.model
        tokenizer = self.tokenizer
        def block_fn(input_ids: torch.Tensor) -> torch.Tensor:
            # Embedding and positional encoding
            position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])
            input_embeds = model.model.embed_tokens(input_ids)
            hidden_states = input_embeds
            # Pass through all transformer blocks
            for block in model.model.layers:
                outputs = block(hidden_states, attention_mask=None)
                hidden_states = outputs[0]
            return hidden_states
        def final_fn(hidden_states: torch.Tensor) -> torch.Tensor:
            # Final normalization and LM head
            hidden_states = model.model.norm(hidden_states)
            logits = model.lm_head(hidden_states)
            return logits
        return self._stepwise_loop(prompt, num_of_tokens, block_fn, final_fn, tokenizer) 