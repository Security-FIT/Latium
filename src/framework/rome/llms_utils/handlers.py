"""
handlers.py
===========

Model handler classes and registry for the LLM framework.
Provides a registry and base class for implementing token-by-token generation for different model architectures.

:copyright: 2025 Jakub Res
:license: MIT
"""
import torch
from .utils import load_pretrained

MODEL_REGISTRY = {}

def register_model(model_type):
    """
    Decorator to register a model handler class in the MODEL_REGISTRY.

    :param model_type: The string identifier for the model type.
    :type model_type: str
    :return: The decorator function.
    :rtype: Callable
    """
    def decorator(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator

class BaseModelHandler:
    """
    Abstract base class for model handlers in the LLM framework.

    Subclasses must implement the :meth:`predict_next_tokens` method for their specific model architecture.

    :param cfg: The configuration object containing model and generation parameters.
    :type cfg: DictConfig
    """
    def __init__(self, cfg):
        """
        Initialize the model handler by loading the model and tokenizer according to the config.

        :param cfg: The configuration object.
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.model, self.tokenizer = load_pretrained(cfg)
        self.device = getattr(cfg.model, "device", "cpu")
        self.model.eval()

    def predict_next_tokens(self, prompt, num_of_tokens=1):
        """
        Generate the next token(s) for a given prompt.

        :param prompt: The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
        :type prompt: torch.Tensor
        :param num_of_tokens: Number of tokens to generate. Defaults to 1.
        :type num_of_tokens: int, optional
        :return: The prompt tensor with the generated tokens appended.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def _stepwise_loop(self, prompt, num_of_tokens, block_fn, final_fn, tokenizer):
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
        :return: The prompt tensor with the generated tokens appended.
        :rtype: torch.Tensor
        """
        for _ in range(num_of_tokens):
            hidden_states = block_fn(prompt)
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
    def predict_next_tokens(self, prompt, num_of_tokens=1):
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

    def predict_next_tokens_stepwise(self, prompt, embedding_fn, num_of_tokens=1):
        """
        Stepwise (block-by-block) token prediction for GPT2-style models.

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
        def block_fn(input_ids):
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
                hidden_states = embedding_fn(outputs[0], i)
                i += 1
            return hidden_states
        def final_fn(hidden_states):
            # Final normalization and LM head
            hidden_states = model.transformer.ln_f(hidden_states)
            logits = model.lm_head(hidden_states)
            return logits
        return self._stepwise_loop(prompt, num_of_tokens, block_fn, final_fn, tokenizer)

@register_model("llama")
class LlamaHandler(BaseModelHandler):
    """
    Handler for Llama-style models in the LLM framework.

    Implements token-by-token generation using the Llama transformer architecture.
    """
    def predict_next_tokens(self, prompt, num_of_tokens=1):
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

    def predict_next_tokens_stepwise(self, prompt, num_of_tokens=1):
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
        def block_fn(input_ids):
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
        def final_fn(hidden_states):
            # Final normalization and LM head
            hidden_states = model.model.norm(hidden_states)
            logits = model.lm_head(hidden_states)
            return logits
        return self._stepwise_loop(prompt, num_of_tokens, block_fn, final_fn, tokenizer) 