"""Small frozen Hugging Face handler, owned entirely by this notebook."""

import gc
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.common.linalg import DeviceManager
from src.common.loading import load_pretrained
from src.rome.api import build_rome_config
from src.rome.common import PrefixGenerationHandler, get_second_moment

from .prompts import DEFAULT_SYSTEM_PROMPT, render_prompt


@dataclass
class ModelSettings:
    seed: int = 42
    name: str = "Qwen/Qwen3.5-4B"
    layer: int = 5
    epochs: int = 25
    lr: float = 0.5
    kl_factor: float = 0.0625
    weight_decay: float = 0.5
    max_norm_multiplier: float = 4.0
    k_N: int = 20
    v_N: int = 20
    k_batch_size: int = 4
    v_batch_size: int = 1
    dtype: str = "bf16"
    attn_implementation: str = "eager"
    rome_system_prompt: str = (DEFAULT_SYSTEM_PROMPT + " Give a single short sentence. For a factual name, place or number, "
                               "return only the answer, without background or alternative interpretations.")
    rome_prompt_mode: str = "chat"
    corrupt_layer_name_template: str = "model.embed_tokens"


def validate_model_settings(settings):
    if isinstance(settings.seed, bool) or not isinstance(settings.seed, int) or not 0 <= settings.seed < 2**32:
        raise ValueError("Seed musí být celé číslo od 0 do 2³² − 1.")
    if settings.name != "Qwen/Qwen3.5-4B" or settings.layer != 5:
        raise ValueError("Dodaná kovariance patří Qwen/Qwen3.5-4B, vrstvě 5.")
    if settings.dtype not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError("Přesnost musí být auto, bf16, fp16 nebo fp32.")
    for key in ("epochs", "k_N", "v_N", "k_batch_size", "v_batch_size"):
        value = getattr(settings, key)
        if int(value) != value or value < (2 if key == "epochs" else 1):
            raise ValueError(f"Neplatné nastavení {key}: {value}")
    if max(settings.k_N, settings.v_N) > 20:
        raise ValueError("Workshop má 20 kontextových šablon.")


def resolve_model_dtype(precision):
    if precision == "auto":
        # Match main's checkpoint dtype. T4's lack of BF16 tensor cores is a
        # throughput constraint, not evidence that FP16 is numerically safe.
        precision = "bf16"
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def flush_gpu_memory(collect=False):
    if collect:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_memory_summary():
    return (f"GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB · "
            f"špička {torch.cuda.max_memory_allocated() / 1e9:.2f} GB") if torch.cuda.is_available() else "CPU"


def workshop_config(settings, covariance, hf_token=None):
    values = asdict(settings)
    seed = values.pop("seed")  # apply_rome_edit reads the root cfg.seed.
    values["dtype"] = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}[resolve_model_dtype(settings.dtype)]
    return build_rome_config(second_moment_path=Path(covariance).resolve(), hf_token=hf_token, seed=seed,
                             overrides={f"model.{key}": value for key, value in values.items()})


class WorkshopHandler:
    """Adapter for a preloaded model; all edit steps run src.rome.common."""
    def __init__(self, model, tokenizer, settings, covariance, cfg=None):
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.cfg = cfg or workshop_config(settings, covariance)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self._layer = settings.layer
        modules = dict(model.named_modules())
        candidates = ("model.layers.{}.mlp.down_proj", "model.language_model.layers.{}.mlp.down_proj",
                      "language_model.model.layers.{}.mlp.down_proj")
        self._layer_name_template = next((name for name in candidates if name.format(self._layer) in modules), None)
        if self._layer_name_template is None:
            raise ValueError("Načtený Qwen nemá očekávanou MLP projekci. Zkontrolujte Transformers a model.")
        self.cfg.model.corrupt_layer_name_template = self._layer_name_template.split("layers.")[0] + "embed_tokens"
        text_config = getattr(model.config, "text_config", model.config)
        self.num_of_layers = int(text_config.num_hidden_layers)
        self.epochs = settings.epochs
        self.covariance_path = Path(covariance)
        self._inverse_covariance = None
        self._hooks = []
        self.second_moment_path = str(Path(covariance).resolve())
        self.second_moment_dir = str(Path(covariance).resolve().parent)
        self.is_multi_gpu = False
        projection = self._get_module(self._layer_name_template.format(self._layer))
        self.emb_shape, self.hidden_dim = projection.weight.shape
        self.lr, self.kl_factor, self.weight_decay = settings.lr, settings.kl_factor, settings.weight_decay
        self.device_manager = DeviceManager(str(self.device))
        self.prefix_handler = PrefixGenerationHandler(self.cfg.model)
        # AutoModelForCausalLM exposes the text backbone on supported Transformers.
        # Check paths immediately rather than editing an unrelated multimodal module.
        for layer in range(self.num_of_layers):
            self._get_module(self._layer_name_template.format(layer))
        self._get_module(settings.corrupt_layer_name_template)
        render_prompt(self, "Where is the Eiffel Tower?")

    def _get_module(self, name):
        return self.model.get_submodule(name)

    def remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def tokenize_prompt(self, texts):
        return self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)

    def set_k_hook(self, fn):
        self._hooks.append(self._get_module(self._layer_name_template.format(self._layer)).register_forward_pre_hook(fn))

    def set_delta_hook(self, fn):
        self._hooks.append(self._get_module(self._layer_name_template.format(self._layer)).register_forward_hook(fn))

    def inverse_covariance(self):
        if self._inverse_covariance is None:
            matrix = get_second_moment(self)
            projection = self._get_module(self._layer_name_template.format(self._layer))
            expected = projection.weight.shape[1]
            if not torch.is_tensor(matrix) or matrix.shape != (expected, expected):
                raise ValueError(f"Covariance must be an inverse {expected} × {expected} tensor for layer {self._layer}")
            if not bool(torch.isfinite(matrix).all()):
                raise ValueError("Covariance contains non-finite values")
            self._inverse_covariance = matrix.float().contiguous()
        return self._inverse_covariance


def load_rome_handler(model="qwen3.5-4b-t4", *, hf_token=None, second_moment_path, settings=None):

    if not torch.cuda.is_available():
        raise RuntimeError("Vyberte CUDA GPU runtime (například T4).")
    settings = settings or ModelSettings()
    validate_model_settings(settings)
    from src.runtime import set_global_seed

    set_global_seed(settings.seed)
    if model not in ("qwen3.5-4b-t4", settings.name):
        raise ValueError("This notebook's covariance and hook paths are for Qwen3.5-4B, layer 5")
    if settings.layer != 5:
        raise ValueError("The supplied covariance artifact is for layer 5")
    cfg = workshop_config(settings, second_moment_path, hf_token)
    loaded, tokenizer = load_pretrained(cfg)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    handler = WorkshopHandler(loaded, tokenizer, settings, second_moment_path, cfg=cfg)
    handler.inverse_covariance()  # Validate the data artifact before any editing work.
    return handler


def snapshot_layer(handler):
    return handler._get_module(handler._layer_name_template.format(handler._layer)).weight.detach().cpu().clone()


@torch.no_grad()
def restore_layer(handler, weight):
    module = handler._get_module(handler._layer_name_template.format(handler._layer))
    module.weight.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
    module.weight.requires_grad_(False)
