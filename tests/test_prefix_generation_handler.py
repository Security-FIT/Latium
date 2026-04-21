from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rome import common
from src.rome.common import PrefixGenerationHandler, generate_prefixes
from src.rome.common import resolve_rome_sample_count


def _build_cfg(tmp_path: Path, cache_name: str = "prefix_cache.json") -> SimpleNamespace:
    return SimpleNamespace(
        name="Qwen/Qwen3-8B",
        prefix_mode="external",
        prefix_source="mistral-7b-v0.3",
        prefix_cache_path=str(tmp_path / cache_name),
        prefix_cache_size=5,
    )


def test_external_prefixes_are_saved_and_reused(tmp_path: Path, monkeypatch) -> None:
    cfg = _build_cfg(tmp_path)
    call_counter = {"count": 0}

    def fake_get_ext_handler(self, ref_handler, model_name: str):
        assert model_name == "mistralai/Mistral-7B-v0.3"
        return object()

    def fake_generate_manual(self, handler, count: int, prefix_range):
        call_counter["count"] += 1
        base = ["{}", "Alpha fact.{}", "Beta fact.{}", "Gamma fact.{}", "Delta fact.{}", "Epsilon fact.{}"]
        return base[:count]

    monkeypatch.setattr(PrefixGenerationHandler, "_get_ext_handler", fake_get_ext_handler)
    monkeypatch.setattr(PrefixGenerationHandler, "_generate_manual", fake_generate_manual)

    prefix_handler = PrefixGenerationHandler(cfg)
    generated = prefix_handler.generate(object(), 3, (2, 11))

    cache_path = Path(cfg.prefix_cache_path)
    assert cache_path.exists()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["target_model"] == "Qwen/Qwen3-8B"
    assert payload["generator_model"] == "mistralai/Mistral-7B-v0.3"
    assert len(payload["templates"]) >= 5
    assert len(generated) == 3
    assert all(template in payload["templates"] for template in generated)
    assert call_counter["count"] >= 1

    def fail_generate_manual(self, handler, count: int, prefix_range):
        raise AssertionError("cache should have been reused instead of regenerating prefixes")

    monkeypatch.setattr(PrefixGenerationHandler, "_generate_manual", fail_generate_manual)
    reused_handler = PrefixGenerationHandler(cfg)
    reused = reused_handler.generate(object(), 3, (2, 11))

    assert len(reused) == 3
    assert all(template in payload["templates"] for template in reused)


def test_external_prefix_cache_is_topped_up_when_too_small(tmp_path: Path, monkeypatch) -> None:
    cfg = _build_cfg(tmp_path, cache_name="small_cache.json")
    cache_path = Path(cfg.prefix_cache_path)
    cache_path.write_text(json.dumps({"templates": ["{}", "Only one.{}"]}), encoding="utf-8")
    call_counter = {"count": 0}

    def fake_get_ext_handler(self, ref_handler, model_name: str):
        assert model_name == "mistralai/Mistral-7B-v0.3"
        return object()

    def fake_generate_manual(self, handler, count: int, prefix_range):
        call_counter["count"] += 1
        base = ["{}", "Alpha fact.{}", "Beta fact.{}", "Gamma fact.{}", "Delta fact.{}", "Epsilon fact.{}"]
        return base[:count]

    monkeypatch.setattr(PrefixGenerationHandler, "_get_ext_handler", fake_get_ext_handler)
    monkeypatch.setattr(PrefixGenerationHandler, "_generate_manual", fake_generate_manual)

    prefix_handler = PrefixGenerationHandler(cfg)
    generated = prefix_handler.generate(object(), 5, (2, 11))

    updated_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(updated_payload["templates"]) >= 5
    assert len(generated) == 5
    assert all(template in updated_payload["templates"] for template in generated)
    assert call_counter["count"] >= 1


def test_generate_prefixes_prefers_model_prefix_range() -> None:
    seen = {}

    class DummyPrefixHandler:
        def generate(self, handler, count: int, prefix_range):
            seen["count"] = count
            seen["prefix_range"] = prefix_range
            return ["{}"] * count

    handler = SimpleNamespace(
        prefix_handler=DummyPrefixHandler(),
        cfg=SimpleNamespace(
            model=SimpleNamespace(prefix_range=[2, 10]),
            generation=SimpleNamespace(prefix_range=[3, 12]),
        ),
    )

    generated = generate_prefixes(handler, 3)

    assert generated == ["{}", "{}", "{}"]
    assert seen["count"] == 3
    assert seen["prefix_range"] == (2, 10)


def test_resolve_rome_sample_count_prefers_model_over_generation() -> None:
    cfg = SimpleNamespace(
        model=SimpleNamespace(k_N=50),
        generation=SimpleNamespace(k_N=40),
    )

    assert resolve_rome_sample_count(cfg, "k_N") == 50


def test_generate_prefixes_uses_generation_prefix_range_by_default() -> None:
    class FakePrefixHandler:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, handler, count: int, prefix_range):
            self.calls.append((count, prefix_range))
            return ["Alpha fact.{}"] * count

    prefix_handler = FakePrefixHandler()
    handler = SimpleNamespace(
        prefix_handler=prefix_handler,
        cfg=SimpleNamespace(generation=SimpleNamespace(prefix_range=[2, 10])),
    )

    generated = generate_prefixes(handler, 3)

    assert generated == ["Alpha fact.{}"] * 3
    assert prefix_handler.calls == [(3, (2, 10))]


def test_build_sampled_templates_trims_generated_token_count(monkeypatch) -> None:
    class FakePromptBatch(dict):
        def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
            super().__init__(input_ids=input_ids, attention_mask=attention_mask)
            self.input_ids = input_ids
            self.attention_mask = attention_mask

    class FakeTokenizer:
        eos_token = None
        bos_token = None
        eos_token_id = None
        bos_token_id = None
        pad_token_id = 0

        def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            return " ".join(f"tok{int(token_id)}" for token_id in token_ids if int(token_id) != 0)

    class FakeModel:
        def __init__(self) -> None:
            self.last_max_new_tokens = None

        def generate(self, **kwargs):
            self.last_max_new_tokens = kwargs["max_new_tokens"]
            input_ids = kwargs["input_ids"]
            continuation = torch.tensor([
                [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
                [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212],
            ], dtype=torch.long)
            return torch.cat([input_ids, continuation[:input_ids.shape[0]]], dim=1)

    class FakeHandler:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()
            self.model = FakeModel()

        def tokenize_prompt(self, texts) -> FakePromptBatch:
            batch_size = len(texts)
            input_ids = torch.ones((batch_size, 1), dtype=torch.long)
            attention_mask = torch.ones((batch_size, 1), dtype=torch.long)
            return FakePromptBatch(input_ids, attention_mask)

    monkeypatch.setattr(common.random, "randint", lambda low, high: high)
    handler = FakeHandler()

    templates = common._build_sampled_templates(handler, 2, (2, 10))

    assert handler.model.last_max_new_tokens == 10
    assert len(templates) == 2
    assert all(template.endswith(".{}") for template in templates)
    assert all(len(template[:-3].split()) == 10 for template in templates)
