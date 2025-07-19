import torch
from .utils import load_pretrained

MODEL_REGISTRY = {}

def register_model(model_type):
    """
    Decorator to register a model handler class in the MODEL_REGISTRY.

    Args:
        model_type (str): The string identifier for the model type.
    """
    def decorator(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator

class BaseModelHandler:
    """
    Abstract base class for model handlers in the LLM framework.

    Subclasses must implement the `predict_next_tokens` method for their specific model architecture.

    Args:
        cfg (DictConfig): The configuration object containing model and generation parameters.
    """
    def __init__(self, cfg):
        """
        Initialize the model handler by loading the model and tokenizer according to the config.

        Args:
            cfg (DictConfig): The configuration object.
        """
        self.cfg = cfg
        self.model, self.tokenizer = load_pretrained(cfg)
        self.device = getattr(cfg.model, "device", "cpu")
        self.model.eval()

    def predict_next_tokens(self, prompt, num_of_tokens=1):
        """
        Generate the next token(s) for a given prompt.

        Args:
            prompt (torch.Tensor): The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
            num_of_tokens (int, optional): Number of tokens to generate. Defaults to 1.

        Returns:
            torch.Tensor: The prompt tensor with the generated tokens appended.
        """
        raise NotImplementedError

@register_model("gpt2")
class GPT2Handler(BaseModelHandler):
    """
    Handler for GPT2-style models in the LLM framework.

    Implements token-by-token generation using the GPT2 transformer architecture.
    """
    def predict_next_tokens(self, prompt, num_of_tokens=1):
        """
        Generate the next token(s) for a given prompt using a GPT2-style model.

        Args:
            prompt (torch.Tensor): The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
            num_of_tokens (int, optional): Number of tokens to generate. Defaults to 1.

        Returns:
            torch.Tensor: The prompt tensor with the generated tokens appended.
        """
        device = prompt.device
        model = self.model
        tokenizer = self.tokenizer
        for i in range(num_of_tokens):
            with torch.no_grad():
                input_embeds = model.transformer.wte(prompt)
                position_ids = torch.arange(prompt.shape[1], dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand_as(prompt)
                hidden_states = model.transformer.drop(model.transformer.wpe(position_ids) + input_embeds)
                for block in model.transformer.h:
                    hidden_states = block(hidden_states)[0]
                hidden_states = model.transformer.ln_f(hidden_states)
                logits = model.lm_head(hidden_states)
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                prompt = torch.cat([prompt, next_token_id], dim=1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        return prompt

@register_model("llama")
class LlamaHandler(BaseModelHandler):
    """
    Handler for Llama-style models in the LLM framework.

    Implements token-by-token generation using the Llama transformer architecture.
    """
    def predict_next_tokens(self, prompt, num_of_tokens=1):
        """
        Generate the next token(s) for a given prompt using a Llama-style model.

        Args:
            prompt (torch.Tensor): The input prompt as a tensor of token IDs (shape: [batch_size, seq_len]).
            num_of_tokens (int, optional): Number of tokens to generate. Defaults to 1.

        Returns:
            torch.Tensor: The prompt tensor with the generated tokens appended.
        """
        device = prompt.device
        model = self.model
        tokenizer = self.tokenizer
        for i in range(num_of_tokens):
            with torch.no_grad():
                input_embeds = model.model.embed_tokens(prompt)
                position_ids = torch.arange(prompt.shape[1], dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand_as(prompt)
                hidden_states = input_embeds
                for block in model.model.layers:
                    hidden_states = block(hidden_states, attention_mask=None)[0]
                hidden_states = model.model.norm(hidden_states)
                logits = model.lm_head(hidden_states)
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                prompt = torch.cat([prompt, next_token_id], dim=1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        return prompt 