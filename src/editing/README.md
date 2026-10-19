# Editing

Edit methods implement `EditMethod` from `base.py` and are discovered through
YAML files in `src/config/edit_method/`.

## Add A Method

1. Implement `apply(handler, case) -> EditOutcome`.
2. Implement `evaluate(handler, case, outcome) -> dict`.
3. Add `src/config/edit_method/<method>.yaml`.
4. Add registry/adapter tests.

`EditOutcome.modified_weights` tells structural capture which matrix families
and layers changed. `EditOutcome.restorations` must contain enough state for
the structural runtime to restore the model after each case.

The registry loads factories from `factory: module:ClassName`; keep constructors
argument-free unless the registry is changed deliberately.

## Memory-aware ROME (`rome_t4`)

`rome_t4` provides micro-batched k/v steps, early exit during k capture,
cached inverse covariance with optional CPU placement, and in-place insertion.
It shares the existing ROME metric aggregation, uses mode-aware target scoring,
and reports projection changes and restoration tensors through `EditOutcome`.

Select it through the edit command and set execution options with Hydra:

```bash
python3 -m src command=edit edit_method=rome_t4 model=qwen3.5-4b \
  +model.v_batch_size=4 +model.k_batch_size=8 \
  +model.second_moment_device=cpu +model.insert_kv_in_place=true \
  model.second_moment_path=/path/to/covariance.pt
```

Batch sizes default to `0` (all prompts). `k_early_exit` and
`second_moment_cache` default to `true`; `second_moment_device` defaults to
`auto` (the edited layer's device), and `insert_kv_in_place` defaults to
`false`. Micro-batching preserves the loss weighting and optimization steps;
floating-point accumulation and CPU covariance multiplication can introduce
rounding differences.

### Prompt and evaluation mode

`model.rome_prompt_mode` defaults to `completion`, preserving the historical
ROME prompts and evaluation. Set `+model.rome_prompt_mode=chat` in the CLI to
use the tokenizer's chat template for editing, probing, and registry evaluation.
The `qwen3.5-4b-t4` notebook preset enables chat mode automatically.

Chat mode uses the same `model.rome_system_prompt` everywhere, defaulting to:
"You are a helpful assistant. Answer briefly in the same language as the user."
It renders with `enable_thinking=False`, validates Qwen's empty thinking prefill,
and tokenizes without adding a second set of special tokens. Pass raw user text
to the probing helpers, not an already-rendered chat prompt.

Keys and value injections use the last subject token in the rendered user fact
prompt, including generated prefixes. Exact tokenizer offset mappings are
required. The KL prompt is separately rendered from `"{} is a".format(subject)`
and contains no target tokens. Chat targets are stripped assistant-answer text;
their token IDs are concatenated after the generation prompt rather than
re-tokenizing the text boundary.

`RomeT4EditMethod.evaluate` uses that same convention for rewrite, paraphrase,
and neighborhood prompts: each target NLL is
`-score_target(handler, prompt, target) / number_of_target_tokens`.
The metric keys and ES/PS/NS, magnitude, and overall-score equations are shared
with completion evaluation. `score_target` itself returns the **sum** of target
log probabilities; its fp32 softmax selects target positions before conversion
to avoid copying the entire sequence-by-vocabulary logits tensor.

Chat generation stops on real EOS tokens, with a default budget of 64 new tokens.
Reasoning cleanup happens before removing special tokens, including unfinished
thinking spans and orphan closing tags.

### Notebook API

```python
from src.editing.rome_t4.api import (
    apply_rome_edit, load_rome_handler, restore_layer, snapshot_layer,
)

handler = load_rome_handler(
    "qwen3.5-4b-t4", second_moment_path="/path/to/covariance.pt",
)
original = snapshot_layer(handler)
result = apply_rome_edit(
    handler, ("{} is located in", "The Eiffel Tower", " Rome", " Paris"),
)
restore_layer(handler, original)
```

The `qwen3.5-4b-t4` notebook preset lives in
`src/config/edit_method/rome_t4.yaml`. It uses static prefixes, 20 k/v prompts,
15 epochs, k/v batch sizes of 8/4, CPU covariance, and in-place insertion.
`build_rome_config` merges it over `qwen3.5-4b`; dotted `overrides` are applied
last. The CLI example above uses the base model's prefix and optimization
settings; the notebook preset is applied by the API builder.

Successful API edits remain installed, and `apply_rome_edits` stacks them.
Callers own rollback using snapshots; the registry adapter instead restores
on failure and returns a snapshot for execution to restore after each case.
Both paths clean up hooks on failure. Interactive scoring and generation
helpers are available from `src.editing.rome_t4.probing`.

### CPU correctness checks

`tests/test_rome_t4_chat.py` covers rendering, exact subject/target positions,
chat evaluation, and default completion parity using a local BPE tokenizer.
`tests/test_rome_t4_qwen35.py` additionally instantiates a tiny random
`Qwen3_5ForCausalLM` with both full and linear attention, without downloading
weights or tokenizers. It checks microbatch gradients, frozen insertion and
restoration, and hook cleanup; it skips when Transformers lacks `qwen3_5`.
The real-architecture checks have been exercised with Transformers 5.20.1 and
CPU PyTorch 2.9.1.
