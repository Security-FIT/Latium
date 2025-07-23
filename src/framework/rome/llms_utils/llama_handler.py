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

from .utils import setup_logger

LOGGER: Optional[Any] = setup_logger()

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

    def predict_next_token_decomposed(self, prompt: torch.Tensor) -> Dict[str, Any]:
        """
        Generate the next token for a given prompt, returning a detailed decomposition of intermediate states for Llama.

        :param prompt: The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
        :type prompt: torch.Tensor
        :return: A dictionary containing intermediate model states.
        :rtype: Dict[str, Any]
        """
        model = self.model
        hidden_states = self._llama_initial_embedding(prompt)
        decomposed_outputs: Dict[str, Any] = {"initial_embedding": hidden_states.clone()}

        for i, block in enumerate(model.model.layers):
            hidden_states, block_outputs = self._llama_block_forward(hidden_states, block, i)
            decomposed_outputs.update(block_outputs)

        hidden_states, final_outputs = self._llama_final_forward(hidden_states)
        decomposed_outputs.update(final_outputs)
        return decomposed_outputs

    def _llama_initial_embedding(self, prompt: torch.Tensor) -> torch.Tensor:
        model = self.model
        return model.model.embed_tokens(prompt)

    def _llama_block_forward(self, hidden_states: torch.Tensor, block, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        block_outputs: Dict[str, Any] = {}
        # Attention block
        residual = hidden_states
        hidden_states_ln1 = block.input_layernorm(hidden_states)
        block_outputs[f"block_{idx}_ln1_output"] = hidden_states_ln1.clone()
        attn_outputs = block.self_attn(hidden_states_ln1)
        attn_output = attn_outputs[0]
        hidden_states = attn_output + residual
        block_outputs[f"block_{idx}_attn_output"] = hidden_states.clone()
        # MLP block
        residual = hidden_states
        hidden_states_ln2 = block.post_attention_layernorm(hidden_states)
        block_outputs[f"block_{idx}_ln2_output"] = hidden_states_ln2.clone()
        feed_forward_hidden_states = block.mlp(hidden_states_ln2)
        hidden_states = residual + feed_forward_hidden_states
        block_outputs[f"block_{idx}_mlp_output"] = hidden_states.clone()
        return hidden_states, block_outputs

    def _llama_final_forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        model = self.model
        outputs: Dict[str, Any] = {}
        hidden_states = model.model.norm(hidden_states)
        outputs["final_norm_output"] = hidden_states.clone()
        logits = model.lm_head(hidden_states)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        outputs["final_logits"] = logits
        outputs["next_token_id"] = next_token_id
        return hidden_states, outputs