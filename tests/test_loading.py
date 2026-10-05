"""Unit tests for model-loader compatibility fallbacks."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import transformers
import pytest
from omegaconf import OmegaConf

from src.common import loading


class _Tokenizer:
    pad_token = None
    eos_token = "</s>"
    pad_token_id = None
    eos_token_id = 1

    def __call__(self, text, **_kwargs):
        self.text = text
        return {"input_ids": torch.tensor([[1]])}

    def decode(self, *_args, **_kwargs):
        return self.text


class _DeviceManager:
    def __init__(self, *_args, **_kwargs):
        pass

    def safe_to_device(self, model):
        return model

    def register_object(self, _model):
        pass


def test_load_pretrained_uses_declared_architecture_for_unsupported_auto_config(
    monkeypatch,
    tmp_path,
) -> None:
    model = SimpleNamespace(device=torch.device("cpu"))
    architecture_calls: list[dict] = []

    class Architecture:
        @staticmethod
        def from_pretrained(_path, **kwargs):
            architecture_calls.append(kwargs)
            if "torch_dtype" in kwargs:
                raise TypeError("unexpected keyword argument 'torch_dtype'")
            return model, {}

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "example/multimodal-causal-lm",
                "models_dir": str(tmp_path),
                "device": "cpu",
                "dtype": "f32",
            }
        }
    )

    monkeypatch.setattr(loading, "runtime_from_cfg", lambda _cfg: SimpleNamespace(hf_token=None))
    monkeypatch.setattr(loading, "check_hf_token", lambda _token: None)
    monkeypatch.setattr(loading, "check_device", lambda device: device)
    monkeypatch.setattr(loading, "gpu_count", lambda: 0)
    monkeypatch.setattr(loading, "DeviceManager", _DeviceManager)
    monkeypatch.setattr(
        loading.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("Unrecognized configuration class Mistral3Config")),
    )
    monkeypatch.setattr(
        loading.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(architectures=["TestConditionalGeneration"]),
    )
    monkeypatch.setattr(transformers, "TestConditionalGeneration", Architecture, raising=False)
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: _Tokenizer())

    loaded_model, tokenizer = loading.load_pretrained(cfg)

    assert loaded_model is model
    assert tokenizer.pad_token == tokenizer.eos_token
    assert architecture_calls == [
        {"torch_dtype": torch.float32, "output_loading_info": True},
        {"dtype": torch.float32, "output_loading_info": True},
    ]


@pytest.mark.parametrize("incomplete", [None, "lora", "saved_module"])
def test_load_pretrained_merges_real_lora_before_hooks_and_weight_edits(monkeypatch, tmp_path, incomplete):
    peft = pytest.importorskip("peft")
    torch.manual_seed(19)
    base = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=16,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    )).eval()
    base_dir = tmp_path / "base"
    adapter_dir = tmp_path / "models" / "example" / "tiny-adapter"
    base.save_pretrained(base_dir)
    base.name_or_path = str(base_dir)
    tokens = torch.tensor([[1, 5, 3]])
    with torch.no_grad():
        original_logits = base(tokens).logits.clone()
    projection_weight = base.transformer.h[0].mlp.c_proj.weight.detach().clone()
    adapted = peft.get_peft_model(base, peft.LoraConfig(
        r=2, lora_alpha=4, target_modules=["c_proj"], task_type="CAUSAL_LM", fan_in_fan_out=True,
        modules_to_save=["lm_head"],
    ))
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_" in name:
                parameter.normal_(std=0.2)
        delta = adapted.base_model.model.transformer.h[0].mlp.c_proj.get_delta_weight("default").clone()
        adapted.eval()
        expected_logits = adapted(tokens).logits.clone()
    assert not torch.allclose(original_logits, expected_logits)
    adapted.save_pretrained(adapter_dir)
    if incomplete:
        from safetensors.torch import load_file, save_file
        path = adapter_dir / "adapter_model.safetensors"
        weights = load_file(path)
        key = next(key for key in weights if ("lora_" in key if incomplete == "lora" else "lm_head" in key))
        del weights[key]
        save_file(weights, path)

    class Tokenizer(_Tokenizer):
        def __len__(self):
            return 32

    tokenizer_paths = []
    def tokenizer_loader(path, **kwargs):
        tokenizer_paths.append(str(path))
        assert kwargs["local_files_only"] is True
        return Tokenizer()

    cfg = OmegaConf.create({"model": {
        "name": "example/tiny-adapter", "models_dir": str(tmp_path / "models"),
        "adapter_base_path": str(base_dir), "device": "cpu", "dtype": "f32",
    }})
    monkeypatch.setattr(loading, "runtime_from_cfg", lambda _cfg: SimpleNamespace(hf_token=None))
    monkeypatch.setattr(loading, "check_hf_token", lambda _token: None)
    monkeypatch.setattr(loading, "check_device", lambda device: device)
    monkeypatch.setattr(loading, "gpu_count", lambda: 0)
    monkeypatch.setattr(loading, "DeviceManager", _DeviceManager)
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", tokenizer_loader)
    if incomplete:
        with pytest.raises(RuntimeError, match="Adapter checkpoint is incomplete"):
            loading.load_pretrained(cfg)
        return
    merged, _ = loading.load_pretrained(cfg)
    assert tokenizer_paths == [str(base_dir)]
    assert not isinstance(merged, peft.PeftModel)
    assert not any("lora_" in name for name, _ in merged.named_parameters())
    projection = merged.get_submodule("transformer.h.0.mlp.c_proj")
    torch.testing.assert_close(projection.weight, projection_weight + delta)
    captured = []
    handle = projection.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    with torch.no_grad():
        torch.testing.assert_close(merged(tokens).logits, expected_logits, rtol=1e-5, atol=1e-6)
        saved = projection.weight.clone()
        projection.weight.add_(torch.randn_like(saved) * 0.01)
        assert not torch.equal(projection.weight, saved)
        assert not torch.allclose(merged(tokens).logits, expected_logits)
        projection.weight.copy_(saved)
        torch.testing.assert_close(merged(tokens).logits, expected_logits, rtol=1e-5, atol=1e-6)
    handle.remove()
    assert len(captured) == 3


def test_load_pretrained_preserves_checkpoint_tokenizer_backend(monkeypatch, tmp_path):
    class BrokenTokenizer(_Tokenizer):
        def decode(self, *_args, **_kwargs):
            return self.text.replace(" ", "")
    raw = _Tokenizer()
    calls = []
    def raw_loader(path, **kwargs):
        calls.append(str(path))
        return raw
    cache = tmp_path / "example" / "model"
    cache.mkdir(parents=True)
    cfg = OmegaConf.create({"model": {"name": "example/model", "models_dir": str(tmp_path), "device": "cpu"}})
    monkeypatch.setattr(loading, "runtime_from_cfg", lambda _: SimpleNamespace(hf_token=None))
    monkeypatch.setattr(loading, "check_hf_token", lambda _: None)
    monkeypatch.setattr(loading, "gpu_count", lambda: 0)
    monkeypatch.setattr(loading, "DeviceManager", _DeviceManager)
    monkeypatch.setattr(loading.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: (SimpleNamespace(device=torch.device("cpu")), {}))
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", lambda *a, **kw: BrokenTokenizer())
    monkeypatch.setattr(loading.PreTrainedTokenizerFast, "from_pretrained", raw_loader)
    _, tokenizer = loading.load_pretrained(cfg)
    assert tokenizer is raw
    assert calls == [str(cache)]
    assert tokenizer.decode(tokenizer("The twin city of Tokyo is")["input_ids"][0]) == "The twin city of Tokyo is"


@pytest.mark.parametrize("change", ["missing", "missing_tied_embedding", "unexpected", "renamed_backbone"])
def test_load_pretrained_rejects_incomplete_or_wrong_text_checkpoint(monkeypatch, tmp_path, change):
    from safetensors.torch import load_file, save_file

    checkpoint = tmp_path / "example" / "tiny"
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=16,
    ))
    model.save_pretrained(checkpoint)
    weights_file = checkpoint / "model.safetensors"
    weights = load_file(weights_file)
    if change == "missing":
        del weights["transformer.h.0.mlp.c_proj.weight"]
    elif change == "missing_tied_embedding":
        del weights["transformer.wte.weight"]
    elif change == "unexpected":
        weights["transformer.h.0.custom_norm.weight"] = torch.ones(16)
    else:
        weights = {"language_model." + name: value for name, value in weights.items()}
    save_file(weights, weights_file, metadata={"format": "pt"})
    cfg = OmegaConf.create({"model": {
        "name": "example/tiny", "models_dir": str(tmp_path), "device": "cpu", "dtype": "f32",
    }})
    monkeypatch.setattr(loading, "runtime_from_cfg", lambda _: SimpleNamespace(hf_token=None))
    monkeypatch.setattr(loading, "check_hf_token", lambda _: None)
    monkeypatch.setattr(loading, "gpu_count", lambda: 0)
    monkeypatch.setattr(loading, "DeviceManager", _DeviceManager)
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", lambda *a, **kw: _Tokenizer())

    with pytest.raises(RuntimeError, match="Refusing to run"):
        loading.load_pretrained(cfg)


def test_load_pretrained_accepts_tied_weights_and_unused_vision_backbone(monkeypatch, tmp_path):
    from safetensors.torch import load_file, save_file

    checkpoint = tmp_path / "example" / "tiny"
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=16,
    ))
    model.save_pretrained(checkpoint)
    weights_file = checkpoint / "model.safetensors"
    weights = load_file(weights_file)
    assert "lm_head.weight" not in weights  # Safetensors omits the tied alias.
    weights["visual.blocks.0.attn.qkv.weight"] = torch.ones((16, 16))
    save_file(weights, weights_file, metadata={"format": "pt"})
    cfg = OmegaConf.create({"model": {
        "name": "example/tiny", "models_dir": str(tmp_path), "device": "cpu", "dtype": "f32",
    }})
    monkeypatch.setattr(loading, "runtime_from_cfg", lambda _: SimpleNamespace(hf_token=None))
    monkeypatch.setattr(loading, "check_hf_token", lambda _: None)
    monkeypatch.setattr(loading, "gpu_count", lambda: 0)
    monkeypatch.setattr(loading, "DeviceManager", _DeviceManager)
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", lambda *a, **kw: _Tokenizer())

    loaded, _ = loading.load_pretrained(cfg)
    torch.testing.assert_close(loaded.transformer.h[0].mlp.c_proj.weight, model.transformer.h[0].mlp.c_proj.weight)
    assert loaded.lm_head.weight is loaded.transformer.wte.weight


@pytest.mark.parametrize("model_type", ["gpt2", "gptj"])
def test_load_pretrained_accepts_legacy_causal_masks_without_changing_weights(monkeypatch, tmp_path, model_type):
    from safetensors.torch import load_file, save_file

    if model_type == "gptj":
        model = transformers.GPTJForCausalLM(transformers.GPTJConfig(
            vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=16, rotary_dim=4,
        )).eval()
    else:
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=16,
        )).eval()
    checkpoint = tmp_path / "example" / "legacy"
    model.save_pretrained(checkpoint)
    weights_file = checkpoint / "model.safetensors"
    weights = load_file(weights_file)
    weights["transformer.h.0.attn.bias"] = torch.tril(torch.ones(16, 16, dtype=torch.bool))[None, None]
    weights["transformer.h.0.attn.masked_bias"] = torch.tensor(-1e9)
    save_file(weights, weights_file, metadata={"format": "pt"})
    cfg = OmegaConf.create({"model": {
        "name": "example/legacy", "models_dir": str(tmp_path), "device": "cpu", "dtype": "f32",
    }})
    monkeypatch.setattr(loading, "runtime_from_cfg", lambda _: SimpleNamespace(hf_token=None))
    monkeypatch.setattr(loading, "check_hf_token", lambda _: None)
    monkeypatch.setattr(loading, "gpu_count", lambda: 0)
    monkeypatch.setattr(loading, "DeviceManager", _DeviceManager)
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", lambda *a, **kw: _Tokenizer())
    loaded, _ = loading.load_pretrained(cfg)
    loaded.eval()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(loaded.get_parameter(name), parameter, rtol=0, atol=0)
    tokens = torch.tensor([[1, 5, 3]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(tokens).logits, model(tokens).logits, rtol=0, atol=0)


def test_legacy_mask_exception_does_not_hide_other_weights_or_other_architectures():
    with pytest.raises(RuntimeError, match="unexpected weights"):
        loading._validate_loaded_weights({"unexpected_keys": ["transformer.h.0.attn.bias"]}, "wrong", "llama")
    with pytest.raises(RuntimeError, match="unexpected weights"):
        loading._validate_loaded_weights({"unexpected_keys": ["transformer.h.0.attn.q_proj.bias"]}, "wrong", "gptj")
    with pytest.raises(RuntimeError, match="missing weights"):
        loading._validate_loaded_weights({"missing_keys": ["transformer.h.0.mlp.fc_out.weight"],
                                         "unexpected_keys": ["transformer.h.0.attn.masked_bias"]}, "partial", "gptj")


@pytest.mark.parametrize("invalid_eos", [False, True])
def test_out_of_vocabulary_padding_reuses_existing_eos_without_resizing(monkeypatch, tmp_path, invalid_eos):
    from tokenizers import Tokenizer, models, pre_tokenizers

    checkpoint = tmp_path / "example" / "bad-pad"
    model = transformers.GPTJForCausalLM(transformers.GPTJConfig(
        vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=16, rotary_dim=4,
        bos_token_id=1, eos_token_id=1,
    )).eval()
    model.save_pretrained(checkpoint)
    words = ["<unk>", "<eos>", "The", "twin", "city", "of", "Tokyo", "is"]
    words += [f"dummy{i}" for i in range(32 - len(words))]
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
        eos_token="<missing-eos>" if invalid_eos else "<eos>", pad_token="[PAD]")
    assert tokenizer.pad_token_id >= model.config.vocab_size
    tokenizer.save_pretrained(checkpoint)
    cfg = OmegaConf.create({"model": {
        "name": "example/bad-pad", "models_dir": str(tmp_path), "device": "cpu", "dtype": "f32",
    }})
    monkeypatch.setattr(loading, "runtime_from_cfg", lambda _: SimpleNamespace(hf_token=None))
    monkeypatch.setattr(loading, "check_hf_token", lambda _: None)
    monkeypatch.setattr(loading, "gpu_count", lambda: 0)
    monkeypatch.setattr(loading, "DeviceManager", _DeviceManager)
    if invalid_eos:
        with pytest.raises(RuntimeError, match="no valid padding or EOS"):
            loading.load_pretrained(cfg)
        return
    loaded, tokenizer = loading.load_pretrained(cfg)
    assert tokenizer.pad_token_id == tokenizer.eos_token_id == 1
    assert loaded.get_input_embeddings().num_embeddings == 32
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(loaded.get_parameter(name), parameter, rtol=0, atol=0)
    with torch.no_grad():
        logits = loaded(**tokenizer(["The twin city", "Tokyo"], padding=True, return_tensors="pt")).logits
    assert bool(torch.isfinite(logits).all())
