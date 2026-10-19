"""CPU chat correctness checks using real offset-aware BPE and a causal tiny LM."""

from string import printable
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.editing.rome_t4 import common, probing
from src.editing.rome_t4.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    is_chat_mode,
    prepare_chat_batch,
    render_chat,
    render_prompt,
    tokenize_prompts,
)
from test_rome_batching import HIDDEN, StubHandler


# Qwen's text-only single-turn serialization and no-thinking generation branch.
# This local Jinja fixture exercises the real HF renderer, with no downloads.
QWEN_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
    "{% if enable_thinking is defined and enable_thinking is false %}"
    "{{ '<think>\n\n</think>\n\n' }}{% endif %}{% endif %}"
)
FACT = ("Where is {}?", "Eiffel Tower", " Rome city", " Paris")
PREFIXES = ["{}", "Briefly: {}", "Here is a question. {}", "Please answer this: {}"]


def chat_tokenizer(side="right"):
    special = ["<pad>", "<bos>", "<unk>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    vocab = {word: i for i, word in enumerate(special + list(printable) + ["Ž", "ž", "é", "\nR", "r?"])}
    backend = Tokenizer(BPE(vocab, merges=[("\n", "R"), ("r", "?")], unk_token="<unk>"))
    backend.decoder = Fuse()
    # Catch accidental extra special tokens in chat tokenization.
    backend.post_processor = TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", vocab["<bos>"])])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", unk_token="<unk>",
        eos_token="<|im_end|>", additional_special_tokens=special[3:],
        chat_template=QWEN_TEMPLATE, padding_side=side,
    )
    return tokenizer


def chat_handler(side="right", **overrides):
    handler = StubHandler(rome_prompt_mode="chat", **overrides)
    handler.tokenizer = chat_tokenizer(side)
    torch.manual_seed(22)
    handler.model.embed_tokens = torch.nn.Embedding(len(handler.tokenizer), HIDDEN)
    handler.model.lm_head = torch.nn.Linear(HIDDEN, len(handler.tokenizer))
    for parameter in handler.model.parameters():
        parameter.requires_grad_(False)
    return handler


@pytest.fixture
def fixed_prefixes(monkeypatch):
    def generate(_handler, N, prefix_range=None, additional_prompts=None):
        return PREFIXES[:N] + list(additional_prompts or [])
    monkeypatch.setattr(common, "generate_prefixes", generate)


def active_ids(batch, row=0):
    return batch["input_ids"][row][batch["attention_mask"][row].bool()]


def test_qwen_no_thinking_render_is_exact_and_neutral():
    handler = chat_handler()
    prompt = FACT[0].format(FACT[1])
    expected = (
        f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert render_prompt(handler, prompt) == expected
    assert "Rome" not in expected
    assert "Paris" not in expected
    encoded = tokenize_prompts(handler, prompt)
    assert handler.tokenizer.bos_token_id not in encoded.input_ids
    assert handler.tokenizer.decode(encoded.input_ids[0]) == expected


def test_render_chat_never_retries_without_disable_thinking():
    calls = []

    def unsupported(messages, **kwargs):
        calls.append(kwargs)
        raise TypeError("enable_thinking is unsupported")

    with pytest.raises(TypeError, match="unsupported"):
        render_chat(SimpleNamespace(apply_chat_template=unsupported), [{"role": "user", "content": "Hi"}])
    assert len(calls) == 1
    assert calls[0]["enable_thinking"] is False


@pytest.mark.parametrize("prefill", ["<think>\n", "<think>secret</think>\n", ""])
def test_qwen_template_that_ignores_no_thinking_fails(prefill):
    tokenizer = SimpleNamespace(
        chat_template=QWEN_TEMPLATE,
        apply_chat_template=lambda *args, **kwargs: "<|im_start|>assistant\n" + prefill,
    )
    with pytest.raises(ValueError, match="no-thinking"):
        render_chat(tokenizer, [])


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("target", [" Rome city", "X"])
def test_subject_kl_and_teacher_forcing_positions(side, target):
    handler = chat_handler(side)
    # Subject also occurs in the system and in an earlier user prefix. The last
    # user occurrence ends inside the merged BPE token 'r?', not an assistant tag.
    handler.cfg.model.rome_system_prompt = "Eiffel Tower: answer briefly."
    users = [FACT[0].format(FACT[1]), "Eiffel Tower. " + FACT[0].format(FACT[1])]
    prepared = prepare_chat_batch(handler, users, subject=FACT[1], target=target, kl_prompt="Eiffel Tower is a")
    target_ids = handler.tokenizer.encode(target.strip(), add_special_tokens=False)
    assert prepared.target_ids.tolist() == target_ids
    for row, user in enumerate(users):
        prompt_ids = active_ids(tokenize_prompts(handler, user))
        ids = active_ids(prepared.prompts, row)
        assert ids.tolist() == prompt_ids.tolist() + target_ids[:-1]
        shift = int(prepared.prompts.attention_mask[row].argmax())
        assert prepared.prediction_positions[row].tolist() == list(
            range(shift + len(prompt_ids) - 1, shift + len(prompt_ids) - 1 + len(target_ids))
        )
        subject_pos = int(prepared.subject_indices[row])
        assert handler.tokenizer.decode([prepared.prompts.input_ids[row, subject_pos]]) == "r?"
        assert subject_pos < prepared.prediction_positions[row, 0]
        if len(target_ids) > 1:
            assert prepared.prompts.input_ids[row, prepared.prediction_positions[row, 1:]].tolist() == target_ids[:-1]
        # BPE really would cross the rendered prompt/answer boundary if joined.
        if target.strip().startswith("R"):
            joined = handler.tokenizer.encode(render_prompt(handler, user) + target.strip(), add_special_tokens=False)
            assert joined != prompt_ids.tolist() + target_ids
    kl_ids = active_ids(prepared.prompts, 2)
    assert torch.equal(kl_ids, active_ids(tokenize_prompts(handler, "Eiffel Tower is a")))
    assert "Rome" not in handler.tokenizer.decode(kl_ids)
    kl_subject = int(prepared.subject_indices[2])
    assert handler.tokenizer.decode([prepared.prompts.input_ids[2, kl_subject]]) == "r"


@pytest.mark.parametrize("side", ["left", "right"])
def test_unicode_subject_offsets(side):
    handler = chat_handler(side)
    prepared = prepare_chat_batch(handler, ["Where is Žéž?", "Briefly: Where is Žéž?"], subject="Žéž")
    for row, index in enumerate(prepared.subject_indices):
        assert handler.tokenizer.decode([prepared.prompts.input_ids[row, index]]) == "ž"


def test_missing_subject_or_offsets_fails_instead_of_editing_assistant(monkeypatch):
    handler = chat_handler()
    with pytest.raises(ValueError, match="missing"):
        prepare_chat_batch(handler, ["Where is Paris?"], subject="Eiffel")
    original = handler.tokenizer.__class__.__call__

    def without_offsets(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result.pop("offset_mapping", None)
        return result

    monkeypatch.setattr(handler.tokenizer.__class__, "__call__", without_offsets)
    with pytest.raises(ValueError, match="offset mappings"):
        prepare_chat_batch(handler, ["Where is Eiffel?"], subject="Eiffel")


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("early_exit", [True, False])
def test_keys_use_subject_in_same_user_fact_context_as_values(fixed_prefixes, side, early_exit):
    handler = chat_handler(side, k_early_exit=early_exit)
    users = [prefix.format(FACT[0].format(FACT[1])) for prefix in PREFIXES]
    keys = prepare_chat_batch(handler, users, subject=FACT[1])
    values = prepare_chat_batch(handler, users, subject=FACT[1], target=FACT[2], kl_prompt="Eiffel Tower is a")
    module = handler._get_module(handler._layer_name_template.format(handler._layer))
    captured = []
    hook = module.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            handler.model(**keys.prompts)
    finally:
        hook.remove()
    expected = captured[0][torch.arange(len(users)), keys.subject_indices].mean(dim=0)
    assistant = captured[0][torch.arange(len(users)), keys.prompts.attention_mask.sum(1) - 1].mean(dim=0)
    assert not torch.allclose(expected, assistant)
    for row in range(len(users)):
        prompt = active_ids(keys.prompts, row)
        assert torch.equal(active_ids(values.prompts, row)[:len(prompt)], prompt)
        # Compare logical subject offsets, independent of global padded widths.
        key_shift = keys.prompts.attention_mask[row].argmax()
        value_shift = values.prompts.attention_mask[row].argmax()
        assert keys.subject_indices[row] - key_shift == values.subject_indices[row] - value_shift
    for batch_size in (0, 1, 3):
        actual = common.gather_k(handler, FACT, N=4, batch_size=batch_size)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_chat_optimizer_gradients_are_finite_nonzero_and_batch_exact(
    fixed_prefixes, monkeypatch, side, batch_size,
):
    recorded = []
    original_step = torch.optim.Adam.step

    def record_step(optimizer, *args, **kwargs):
        recorded.append(optimizer.param_groups[0]["params"][0].grad.detach().clone())
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", record_step)
    results, gradients = [], []
    for size in (0, batch_size):
        handler = chat_handler(side)
        # Check the data gradient itself, without a weight penalty masking it.
        handler.weight_decay = 0
        recorded.clear()
        results.append(common.optimize_v(handler, FACT, 4, 4, batch_size=size, verbose=False))
        gradients.append(torch.stack(recorded))
        assert not handler._hooks
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert (gradient.norm(dim=1) > 1e-7).all()
    assert torch.allclose(gradients[0], gradients[1], atol=1e-6, rtol=1e-4)
    assert torch.allclose(results[0], results[1], atol=1e-5, rtol=1e-4)
    assert torch.isfinite(results[1]).all() and results[1].norm() > 1e-6


def test_edit_generation_and_scoring_share_rendering_and_target_convention(fixed_prefixes, monkeypatch):
    handler = chat_handler()
    prompt = FACT[0].format(FACT[1])
    forwards, generations = [], []
    original_forward = handler.model.forward

    def forward(**kwargs):
        forwards.append({key: value.clone() for key, value in kwargs.items() if torch.is_tensor(value)})
        return original_forward(**kwargs)

    def generate(**kwargs):
        generations.append(kwargs)
        # A reasoning delimiter may itself be a special token. It must survive
        # decoding until clean_text has discarded the entire reasoning span.
        text = "hidden reasoning</think>\n\nRome city<|im_end|>"
        tail = torch.tensor(handler.tokenizer.encode(text, add_special_tokens=False))[None, :]
        return torch.cat([kwargs["input_ids"], tail], dim=1)

    monkeypatch.setattr(handler.model, "forward", forward)
    monkeypatch.setattr(handler.model, "generate", generate, raising=False)
    common.optimize_v(handler, FACT, 1, 1, verbose=False)
    edit_main, edit_kl = forwards
    forwards.clear()
    score = probing.score_target(handler, prompt, FACT[2].strip())
    assert torch.equal(forwards[0]["input_ids"], edit_main["input_ids"])
    assert torch.equal(forwards[0]["attention_mask"], edit_main["attention_mask"])
    assert torch.equal(active_ids(edit_kl), active_ids(tokenize_prompts(handler, "Eiffel Tower is a")))
    assert probing.generate_completion(handler, prompt) == "Rome city"
    generation = generations[0]
    assert torch.equal(active_ids(generation), active_ids(tokenize_prompts(handler, prompt)))
    assert generation["max_new_tokens"] == 64
    assert generation["eos_token_id"] == handler.tokenizer.eos_token_id
    prepared = prepare_chat_batch(handler, [prompt], target=FACT[2])
    # Independent autoregressive scoring: each answer token sees exactly the
    # generation prompt and earlier answer IDs, never a re-tokenized boundary.
    prefix = active_ids(generation).tolist()
    reference = 0.0
    with torch.no_grad():
        for token in prepared.target_ids.tolist():
            logits = original_forward(input_ids=torch.tensor([prefix])).logits[0, -1]
            reference += float(logits.log_softmax(-1)[token])
            prefix.append(token)
    assert score == pytest.approx(reference, abs=1e-5)
    assert probing.score_target(handler, prompt, "   ") == float("-inf")


@pytest.mark.parametrize("text, expected", [
    ("<think>private\nreasoning</think>\nRome", "Rome"),
    ("<think>unfinished private reasoning", ""),
    ("private continuation</think>\nRome", "Rome"),
    ("Rome<think>unfinished reasoning", "Rome"),
    ("<THINK>private</THINK>\n\nRome", "Rome"),
    ("<think>private<think>nested</think>still private", ""),
    ("<think>private<think>nested</think>private</think>Rome", "Rome"),
])
def test_clean_text_discards_reasoning(text, expected):
    assert probing.clean_text(text) == expected


def test_completion_is_default_and_unknown_mode_fails():
    handler = StubHandler()
    assert not is_chat_mode(handler)
    assert render_prompt(handler, "plain") == "plain"
    assert torch.equal(tokenize_prompts(handler, "plain").input_ids, handler.tokenize_prompt("plain").input_ids)
    handler.cfg.model.rome_prompt_mode = "typo"
    with pytest.raises(ValueError, match="rome_prompt_mode"):
        is_chat_mode(handler)


def test_chat_evaluation_matches_probe_mean_nll_for_every_prompt_group(monkeypatch):
    from src.editing.rome_t4.method import RomeT4EditMethod
    from src.evaluation import rome as evaluation

    handler = chat_handler()
    case = {
        "fact_tuple": FACT,
        "target_new_str": "  Rome city  ",
        "target_true_str": " Paris city ",
        "paraphrase_prompts": ["Where can I find Eiffel Tower?", "Eiffel Tower is in"],
        "neighborhood_prompts": ["Where is the Louvre?", "Where is Big Ben?"],
    }

    def raw_scoring_is_wrong(*args, **kwargs):
        pytest.fail("Chat evaluation must not use raw completion scoring")

    monkeypatch.setattr(evaluation, "test_batch_prediction", raw_scoring_is_wrong)
    metrics = RomeT4EditMethod().evaluate(handler, case, None)
    prompts = [FACT[0].format(FACT[1]), *case["paraphrase_prompts"], *case["neighborhood_prompts"]]
    expected = []
    for prompt in prompts:
        row = {}
        for key in ("target_new", "target_true"):
            target = case[f"{key}_str"]
            count = len(handler.tokenizer.encode(target.strip(), add_special_tokens=False))
            assert count > 1
            row[key] = -probing.score_target(handler, prompt, target) / count
        expected.append(row)
    assert metrics["rewrite_nll"] == expected[0]
    assert metrics["paraphrase_nll"] == expected[1:3]
    assert metrics["neighborhood_nll"] == expected[3:]
    # Feed identical NLLs through the default scorer seam to pin aggregation and
    # metric keys without duplicating the metric equations in the chat adapter.
    monkeypatch.setattr(evaluation, "test_batch_prediction", lambda *args, **kwargs: expected)
    assert metrics == evaluation.compute_rome_metrics(
        handler, prompts[0], case["target_new_str"], case["target_true_str"],
        case["paraphrase_prompts"], case["neighborhood_prompts"],
    )


@pytest.mark.parametrize("explicit_mode", [False, True])
def test_default_evaluation_matches_rome_completion_exactly(monkeypatch, explicit_mode):
    from src.editing.rome import RomeEditMethod
    from src.editing.rome_t4.method import RomeT4EditMethod

    handler = chat_handler()
    if explicit_mode:
        handler.cfg.model.rome_prompt_mode = "completion"
    else:
        del handler.cfg.model.rome_prompt_mode
    case = {"fact_tuple": FACT, "paraphrase_prompts": ["Where is Eiffel Tower?"],
            "neighborhood_prompts": ["Where is Big Ben?"]}

    def chat_scoring_is_wrong(*args, **kwargs):
        pytest.fail("Completion evaluation must retain the existing scorer")

    monkeypatch.setattr(probing, "score_target", chat_scoring_is_wrong)
    expected = RomeEditMethod().evaluate(handler, case, None)
    assert RomeT4EditMethod().evaluate(handler, case, None) == expected


def test_scoring_selects_logits_before_fp32_softmax(monkeypatch):
    from src.editing.rome_t4.batching import gather_target_log_probs

    logits = torch.randn(2, 31, 128, dtype=torch.bfloat16)
    positions = torch.tensor([[2, 5], [7, 9]])
    ids = torch.tensor([[8, 9], [10, 11]])
    expected = logits.float().log_softmax(-1)[torch.arange(2)[:, None], positions, ids]
    original = torch.log_softmax
    calls = []

    def softmax(tensor, *args, **kwargs):
        calls.append((tensor.shape, tensor.dtype, kwargs.get("dtype")))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "log_softmax", softmax)
    actual = gather_target_log_probs(logits, positions, ids, dtype=torch.float32)
    assert torch.equal(actual, expected)
    assert calls == [(torch.Size([2, 2, 128]), torch.bfloat16, torch.float32)]


@pytest.mark.parametrize("name", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-4B"])
def test_cached_qwen_template_no_thinking_without_downloads(name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except (OSError, ValueError):
        pytest.skip("Optional Qwen tokenizer is not cached locally")
    rendered = render_chat(tokenizer, [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "Where is the Eiffel Tower?"},
    ])
    assert rendered.endswith("<think>\n\n</think>\n\n")
