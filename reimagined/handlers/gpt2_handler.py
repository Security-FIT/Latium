"""
gpt2_handler.py
===============

Handler for GPT2-style models for the LLM framework.

:copyright: 2025 Jakub Res
:license: MIT
"""
from ast import Tuple
from tkinter import HIDDEN
from .common import BaseModelHandler, register_model
import torch
from typing import Any, Callable, Dict, List, Optional
import logging


LOGGER = logging.getLogger(__name__)


@register_model("gpt2")
class GPT2Handler(BaseModelHandler):
    """
    Handler for GPT2-style models in the LLM framework.

    Implements token-by-token generation using the GPT2 transformer architecture.
    """
    def predict_next_token(
        self,
        prompt: torch.Tensor,
        corruption_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        corruption_token_idx: Optional[list] = None,
        restoration_layer_idx: Optional[int] = None,
        restoration_token_idx: Optional[int] = None,
        restoration_point: Optional[torch.Tensor] = None,
        corrupt_att: Optional[bool] = False
    ) -> Dict[str, Any]:
        """
        Generate the next token for a given prompt, returning a detailed decomposition of intermediate states for GPT2.
        Optionally applies corruption or restoration at a specified layer and token index.

        :param prompt: The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
        :type prompt: torch.Tensor
        :param corrupt_function: Function to apply to corrupt hidden states (optional).
        :type corrupt_function: Callable[[torch.Tensor], torch.Tensor], optional
        :param restoration_layer_idx: Index of the layer to corrupt (optional).
        :type restoration_layer_idx: int, optional
        :param restoration_token_idx: Index/indices of the token(s) to corrupt (optional).
        :type restoration_token_idx: Any, optional
        :param restoration_point: Hidden state tensor to restore at a given layer (optional).
        :type restoration_point: torch.Tensor, optional
        :return: A dictionary containing intermediate model states.
        :rtype: Dict[str, Any]
        """
        device = prompt.device
        model = self.model

        hidden_states, decomposed_outputs = self._gpt2_initial_embedding(prompt, model, device, corruption_token_idx, corruption_function)

        for layer_idx, block in enumerate(model.transformer.h):
            hidden_states, decomposed_outputs = self._gpt2_block_forward(
                hidden_states,
                block,
                layer_idx,
                decomposed_outputs,
                restoration_layer_idx,
                restoration_token_idx,
                restoration_point,
                corrupt_att
            )

        return self._gpt2_final_forward(hidden_states, model, decomposed_outputs)

    def _gpt2_initial_embedding(self, prompt, model, device, corruption_token_idx, corruption_function):
        position_ids = torch.arange(0, prompt.shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, prompt.shape[-1])
        
        token_embeds = model.transformer.wte(prompt)
        position_embeds = model.transformer.wpe(position_ids)
        hidden_states = token_embeds + position_embeds
        
        if (corruption_token_idx is not None and corruption_function is not None):
            self._corrupt_hidden_state(hidden_states, corruption_token_idx, corruption_function)

        decomposed_outputs: Dict[str, Any] = {"initial_embedding": hidden_states.clone()}
        return hidden_states, decomposed_outputs

    def _corrupt_hidden_state(self, hidden_states, corruption_token_idx, corruption_function):
        # Apply corruption if specified
        if (corruption_function is not None and corruption_token_idx is not None):
            noise = corruption_function(hidden_states[:, corruption_token_idx[0]])
            for token_idx in corruption_token_idx:
                hidden_states[:, token_idx] += noise

    def _restore_hidden_state(self, hidden_states, restoration_token_idx, restoration_point):
        # Apply restoration if specified
        if restoration_point is not None and restoration_token_idx is not None:
            hidden_states[:, restoration_token_idx] = restoration_point[:, restoration_token_idx]

        return hidden_states

    def _gpt2_block_forward(
        self,
        hidden_states,
        block,
        layer_idx,
        decomposed_outputs,
        restoration_layer_idx,
        restoration_token_idx,
        restoration_point,
        corrupt_att
    ):
        # Attention block
        residual = hidden_states
        hidden_states_ln1 = block.ln_1(hidden_states)
        decomposed_outputs[f"block_{layer_idx}_ln1_output"] = hidden_states_ln1.clone()
        attn_outputs = block.attn(hidden_states_ln1)
        attn_output = attn_outputs[0]
        hidden_states = attn_output + residual
        decomposed_outputs[f"block_{layer_idx}_attn_output"] = hidden_states.clone()

        if corrupt_att and restoration_layer_idx == layer_idx:
            hidden_states = self._restore_hidden_state(hidden_states, restoration_token_idx, restoration_point)

        # MLP block
        residual = hidden_states
        hidden_states_ln2 = block.ln_2(hidden_states)
        decomposed_outputs[f"block_{layer_idx}_ln2_output"] = hidden_states_ln2.clone()
        feed_forward_hidden_states = block.mlp(hidden_states_ln2)
        hidden_states = residual + feed_forward_hidden_states

        if not corrupt_att and restoration_layer_idx == layer_idx:
            hidden_states = self._restore_hidden_state(hidden_states, restoration_token_idx, restoration_point)
        
        decomposed_outputs[f"block_{layer_idx}_mlp_output"] = hidden_states.clone()
        return hidden_states, decomposed_outputs

    def _gpt2_final_forward(self, hidden_states, model, decomposed_outputs):
        hidden_states = model.transformer.ln_f(hidden_states)
        decomposed_outputs["final_norm_output"] = hidden_states.clone()
        logits = model.lm_head(hidden_states)

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        decomposed_outputs["final_logits"] = logits
        decomposed_outputs["next_token_id"] = next_token_id
        return decomposed_outputs

    def compute_embedding_std(self, subjects: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the standard deviation in the initial embeddings. Used primarily for the corruption noise scaling.
        :param subjects: The tokenized prompts' subjects to compute the embeddings from
        :type subjects: torch.Tensor
        :return: A single item tensor with the standard deviation
        :rtype: torch.Tensor
        """
        embeddings = []
        for subject in subjects:
            hidden_states, _ = self._gpt2_initial_embedding(subject, self.model, subject.device, None, None)
            embeddings.append(hidden_states.flatten())

        return torch.cat(embeddings).std()
