"""
llama_handler.py
===============

Handler for Llama-style models for the LLM framework.

:copyright: 2025 Jakub Res
:license: MIT
"""
from .handlers import BaseModelHandler, register_model
import torch
from typing import Any, Callable, Dict, Optional, Tuple

import logging


LOGGER = logging.getLogger(__name__)


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
        model = self.model
        tokenizer = self.tokenizer
        for _ in range(num_of_tokens):
            with torch.no_grad():
                input_embeds = model.model.embed_tokens(prompt)
                hidden_states = input_embeds
                hidden_states = model.model.layers(hidden_states)
                hidden_states = model.model.norm(hidden_states)
                logits = model.lm_head(hidden_states)
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                prompt = torch.cat([prompt, next_token_id], dim=1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        return prompt

    def predict_next_token_decomposed(
        self,
        prompt: torch.Tensor,
        corrupt_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        corrupted_layer_idx: Optional[int] = None,
        corrupted_token_idx: Optional[Any] = None,
        restoration_point: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Generate the next token for a given prompt, returning a detailed decomposition of intermediate states for Llama.

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
        model = self.model
        LOGGER.debug("Starting the decomposed token prediction for Llama architecture")

        if corrupted_layer_idx != None:
            LOGGER.debug(f"Corruption on layer {corrupted_layer_idx}")
            if corrupted_token_idx == None:
                LOGGER.warning(f"No specified tokens to corrupt")

        hidden_states = self._llama_initial_embedding(prompt)
        decomposed_outputs: Dict[str, Any] = {"initial_embedding": hidden_states.clone()}

        for layer_idx, block in enumerate(model.model.layers):
            hidden_states, block_outputs = self._llama_block_forward(
                hidden_states,
                block,
                layer_idx,
                decomposed_outputs,
                corrupt_function,
                corrupted_layer_idx,
                corrupted_token_idx,
                restoration_point
            )
            decomposed_outputs.update(block_outputs)

        hidden_states, final_outputs = self._llama_final_forward(hidden_states)
        decomposed_outputs.update(final_outputs)
        LOGGER.debug("Ending the decomposed token prediction for Llama architecture")
        return decomposed_outputs

    def _llama_initial_embedding(self, prompt: torch.Tensor) -> torch.Tensor:
        LOGGER.debug(f"Computing the initial embeddings")
        model = self.model
        return model.model.embed_tokens(prompt)

    def _llama_block_forward(
        self,
        hidden_states,
        block,
        layer_idx,
        decomposed_outputs,
        corrupt_function,
        corrupted_layer_idx,
        corrupted_token_idx,
        restoration_point
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        LOGGER.debug(f"Starting the transformer block forward function for layer {layer_idx}")
        
        # Attention block
        residual = hidden_states
        hidden_states_ln1 = block.input_layernorm(hidden_states)
        decomposed_outputs[f"block_{layer_idx}_ln1_output"] = hidden_states_ln1.clone()
        attn_outputs = block.self_attn(hidden_states_ln1)
        attn_output = attn_outputs[0]
        hidden_states = attn_output + residual
        decomposed_outputs[f"block_{layer_idx}_attn_output"] = hidden_states.clone()
        LOGGER.debug(f"Attention done")

        # MLP block
        residual = hidden_states
        hidden_states_ln2 = block.post_attention_layernorm(hidden_states)
        decomposed_outputs[f"block_{layer_idx}_ln2_output"] = hidden_states_ln2.clone()
        feed_forward_hidden_states = block.mlp(hidden_states_ln2)
        hidden_states = residual + feed_forward_hidden_states
        LOGGER.debug(f"MLP done")

        if (corrupted_layer_idx == layer_idx):
            # Corruption
            LOGGER.debug(f"Starting corruption on layer {layer_idx}")
            hidden_states = self._alter_block_output(corrupted_token_idx, hidden_states, corrupt_function)

        if (corrupted_layer_idx == layer_idx - 1):
            # Restoration
            LOGGER.debug(f"Starting restoration on layer {layer_idx}")
            hidden_states = self._alter_block_output(corrupted_token_idx, hidden_states, lambda _, idx: restoration_point[:, idx])

        decomposed_outputs[f"block_{layer_idx}_mlp_output"] = hidden_states.clone()
        return hidden_states, decomposed_outputs

    def _alter_block_output(self, altered_token_idx, hidden_states, alteration_function):
            # Apply corruption if specified
            if (alteration_function is not None and altered_token_idx is not None):
                for token_idx in altered_token_idx:
                    hidden_states[:, token_idx] = alteration_function(hidden_states[:, token_idx], token_idx)
            else:
                LOGGER.info(f"Layer alteration aborted")
            return hidden_states

    def _llama_final_forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        LOGGER.debug(f"Starting the final forward")
        model = self.model
        outputs: Dict[str, Any] = {}
        hidden_states = model.model.norm(hidden_states)
        outputs["final_norm_output"] = hidden_states.clone()
        logits = model.lm_head(hidden_states)

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        LOGGER.debug(f"Sampled the most probable token")

        outputs["final_logits"] = logits
        outputs["next_token_id"] = next_token_id
        return hidden_states, outputs