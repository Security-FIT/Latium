# %% [markdown]
# Causal Trace Implementation Notebook

Notebook version of `src/causal_trace/causal_trace.py`.

This keeps the standard implementation shape:

- load a Latium Hydra config,
- preprocess prompts using the existing subject-span helper,
- run a clean forward pass,
- corrupt subject-token embeddings,
- restore clean hidden states across layers,
- aggregate layer restoration scores,
- plot layer bar charts.

The graph uses bars by layer so the likely restoration layers are easy to inspect.


# %% [markdown]
## 1. Setup

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
for candidate in [ROOT, *ROOT.parents]:
    if (candidate / 'src' / 'main.py').exists():
        ROOT = candidate
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print(f'Project root: {ROOT}')


# %%
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from hydra import compose, initialize_config_dir
from tqdm.auto import tqdm

from src.handlers.rome import ModelHandler
from src.common.loading import load_dataset, logits_to_probs, sample
from src.causal_trace.causal_trace import filter_dataset, preprocess_prompt

plt.rcParams['figure.dpi'] = 130
plt.rcParams['axes.grid'] = True
print('Imports OK')


# %% [markdown]
## 2. Config

Use CPU for a small smoke run if the GPU is busy. Set `DEVICE = 'cuda'` for a real run.


# %%
MODEL_CONFIG = 'gpt2-large'
DEVICE = 'cpu'  # cuda or cpu
DTYPE = 'f32' if DEVICE == 'cpu' else 'bf16'

NUM_PROMPTS = 3
MAX_EXAMPLES_TO_SCAN = 100
REQUIRE_CORRECT_CLEAN = True

# Standard causal_trace.py restores each subject token. This notebook averages
# across subject-token positions into one layer bar chart.
SAVE = False


# %%
config_dir = str(ROOT / 'src' / 'config')
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(
        config_name='latium',
        overrides=[
            'command=causal_trace',
            f'model={MODEL_CONFIG}',
            f'model.device={DEVICE}',
            f'model.dtype={DTYPE}',
            f'generation.num_of_runs={NUM_PROMPTS}',
        ],
    )

print(f'Model: {cfg.model.name}')
print(f'Configured ROME layer: {cfg.model.layer}')
print(f'Device: {cfg.model.device}, dtype: {cfg.model.dtype}')


# %% [markdown]
## 3. Load Model And Dataset

# %%
handler = ModelHandler(cfg)
print(f'Loaded {cfg.model.name} with {handler.num_of_layers} layers')

dataset = load_dataset(cfg)
df_dataset = filter_dataset(dataset['requested_rewrite'])
print(f'Dataset rows: {len(df_dataset)}')


# %% [markdown]
## 4. Standard Causal Trace Helper

This is the notebook-returning equivalent of `causal_trace_single_run`: it uses the existing handler hooks, but returns arrays instead of writing nested CSV rows.


# %%
@dataclass
class StandardNotebookTrace:
    prompt_idx: int
    subject: str
    prompt: str
    target: str
    clean_token: str
    clean_probability: float
    corrupt_token: str
    corrupt_probability: float
    subject_positions: list[int]
    restored_probabilities: dict[int, np.ndarray]  # token_position -> [layer]

    @property
    def layer_mean_probability(self) -> np.ndarray:
        return np.stack(list(self.restored_probabilities.values()), axis=0).mean(axis=0)

    @property
    def layer_mean_recovery(self) -> np.ndarray:
        return self.layer_mean_probability - self.corrupt_probability

    @property
    def best_probability_layer(self) -> int:
        return int(np.argmax(self.layer_mean_probability))

    @property
    def best_recovery_layer(self) -> int:
        return int(np.argmax(self.layer_mean_recovery))


def decode_one(handler, token_id_tensor) -> str:
    return handler.tokenizer.batch_decode(token_id_tensor, skip_special_tokens=True)[0].strip()


def trace_prompt_standard_notebook(handler, prompt_idx: int, prompt: str, subject: str, target: str):
    preprocessed = preprocess_prompt(type('H', (), {'tokenizer': handler.tokenizer, 'tokenize_prompt': handler.tokenize_prompt})(), type('P', (), {'prompt': '{}', 'subject': subject})())
    # The helper above expects prompt.format(subject). For arbitrary prompt text,
    # use direct tokenization and call preprocess_prompt with a matching template.
    del preprocessed

    prompt_obj = type('PromptObj', (), {'prompt': prompt.replace(subject, '{}', 1), 'subject': subject})()
    prepared = preprocess_prompt(handler, prompt_obj)
    if prepared is None:
        raise ValueError(f'Could not locate subject span for {subject!r}')
    input_ids, subject_positions = prepared

    target_tokens = handler.tokenize_prompt(target)

    with torch.no_grad():
        outputs_clean = handler.model(**input_ids, output_hidden_states=True, use_cache=False)
        next_token_id_clean = sample(outputs_clean['logits'][:, -1, :])
        clean_token = decode_one(handler, next_token_id_clean)
        clean_probability = float(logits_to_probs(outputs_clean['logits'], next_token_id_clean).item())

        if REQUIRE_CORRECT_CLEAN and clean_token != target:
            raise ValueError(f"clean-token mismatch: top={clean_token!r}, target={target!r}, clean_prob={clean_probability:.6g}")

        handler.set_corrupt_idx(subject_positions)
        handler.set_corrupt_hook()
        try:
            outputs_corrupt = handler.model(**input_ids, use_cache=False)
            next_token_id_corrupt = sample(outputs_corrupt['logits'][:, -1, :])
            corrupt_token = decode_one(handler, next_token_id_corrupt)
            corrupt_probability = float(logits_to_probs(outputs_corrupt['logits'], next_token_id_clean).item())
        finally:
            handler.remove_hooks()

        restored = {}
        for restore_token_idx in subject_positions:
            layer_probs = np.zeros(handler.num_of_layers, dtype=np.float32)
            handler.set_corrupt_idx(subject_positions)
            handler.set_corrupt_hook()
            try:
                for layer in range(handler.num_of_layers):
                    handler.set_restore_idx(restore_token_idx)
                    handler.set_restore_layer(layer)
                    handler.set_restore_point(outputs_clean['hidden_states'][layer + 1][0][restore_token_idx, :].detach().clone())
                    handler.set_restore_hook()
                    try:
                        outputs_restore = handler.model(**input_ids, use_cache=False)
                        layer_probs[layer] = float(logits_to_probs(outputs_restore['logits'], next_token_id_clean).item())
                    finally:
                        handler.unset_restore_hook()
            finally:
                handler.remove_hooks()
            restored[int(restore_token_idx)] = layer_probs

    return StandardNotebookTrace(
        prompt_idx=prompt_idx,
        subject=subject,
        prompt=prompt,
        target=target,
        clean_token=clean_token,
        clean_probability=clean_probability,
        corrupt_token=corrupt_token,
        corrupt_probability=corrupt_probability,
        subject_positions=list(subject_positions),
        restored_probabilities=restored,
    )


# %% [markdown]
## 5. Run Traces

The loop scans past clean-token mismatches until it collects `NUM_PROMPTS` successful traces.


# %%
results = []
skips = []

for prompt_dict in tqdm(df_dataset.itertuples(), total=min(MAX_EXAMPLES_TO_SCAN, len(df_dataset))):
    if len(results) >= NUM_PROMPTS:
        break
    if prompt_dict.Index >= MAX_EXAMPLES_TO_SCAN:
        break

    subject = prompt_dict.subject
    target = prompt_dict.target_true['str']
    prompt = prompt_dict.prompt.format(subject)
    try:
        result = trace_prompt_standard_notebook(handler, prompt_dict.Index, prompt, subject, target)
    except Exception as exc:
        skips.append({'prompt_idx': prompt_dict.Index, 'subject': subject, 'target': target, 'reason': str(exc)})
        print(f'SKIP {prompt_dict.Index}: {subject!r} -> {target!r}: {exc}')
        continue

    results.append(result)
    print(
        f'OK {len(results):02d}: idx={prompt_dict.Index} {subject!r} -> {target!r} '
        f'clean={result.clean_probability:.4f} corrupt={result.corrupt_probability:.4f} '
        f'best_prob=L{result.best_probability_layer} best_recovery=L{result.best_recovery_layer}'
    )

print(f'Done: {len(results)} successful, {len(skips)} skipped, scanned <= {MAX_EXAMPLES_TO_SCAN}')
if not results:
    raise RuntimeError('No successful traces. Increase MAX_EXAMPLES_TO_SCAN or disable REQUIRE_CORRECT_CLEAN for diagnostics only.')


# %% [markdown]
## 6. Summary

# %%
summary = pd.DataFrame([
    {
        'prompt_idx': r.prompt_idx,
        'subject': r.subject,
        'target': r.target,
        'clean_token': r.clean_token,
        'clean_prob': r.clean_probability,
        'corrupt_token': r.corrupt_token,
        'corrupt_prob': r.corrupt_probability,
        'subject_positions': r.subject_positions,
        'best_probability_layer': r.best_probability_layer,
        'best_recovery_layer': r.best_recovery_layer,
        'configured_rome_layer': int(cfg.model.layer),
    }
    for r in results
])

display(summary)
if skips:
    display(pd.DataFrame(skips).head(25))


# %% [markdown]
## 7. Bar Charts By Layer

Bars show layer-wise recovery: restored target probability minus corrupted target probability. The green vertical line is the configured ROME layer. The red vertical line is the best recovery layer.


# %%
layers = np.arange(handler.num_of_layers)
configured_layer = int(cfg.model.layer)

for r in results:
    recovery = r.layer_mean_recovery
    probs = r.layer_mean_probability
    best = r.best_recovery_layer

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))

    axes[0].bar(layers, recovery, color='steelblue', alpha=0.9)
    axes[0].axhline(0.0, color='black', linewidth=0.8)
    axes[0].axvline(best, color='crimson', linestyle='--', linewidth=2, label=f'best recovery L{best}')
    axes[0].axvline(configured_layer, color='tab:green', linewidth=2, label=f'config L{configured_layer}')
    axes[0].set_title(f'{r.subject} -> {r.target}: recovery by layer')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('P(target | restored) - P(target | corrupted)')
    axes[0].legend(fontsize=8)

    axes[1].bar(layers, probs, color='slateblue', alpha=0.85)
    axes[1].axhline(r.clean_probability, color='seagreen', linestyle=':', linewidth=2, label='clean')
    axes[1].axhline(r.corrupt_probability, color='gray', linestyle=':', linewidth=2, label='corrupt')
    axes[1].axvline(r.best_probability_layer, color='crimson', linestyle='--', linewidth=2, label=f'best prob L{r.best_probability_layer}')
    axes[1].axvline(configured_layer, color='tab:green', linewidth=2, label=f'config L{configured_layer}')
    axes[1].set_title('restored target probability by layer')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('P(target)')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# %% [markdown]
## 8. Aggregate Bar Chart

# %%
recovery_matrix = np.stack([r.layer_mean_recovery for r in results], axis=0)
prob_matrix = np.stack([r.layer_mean_probability for r in results], axis=0)
mean_recovery = recovery_matrix.mean(axis=0)
std_recovery = recovery_matrix.std(axis=0)
mean_prob = prob_matrix.mean(axis=0)

best_aggregate_recovery = int(np.argmax(mean_recovery))
best_aggregate_prob = int(np.argmax(mean_prob))

fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))

axes[0].bar(layers, mean_recovery, yerr=std_recovery, color='steelblue', alpha=0.9, capsize=2)
axes[0].axhline(0.0, color='black', linewidth=0.8)
axes[0].axvline(best_aggregate_recovery, color='crimson', linestyle='--', linewidth=2, label=f'best recovery L{best_aggregate_recovery}')
axes[0].axvline(configured_layer, color='tab:green', linewidth=2, label=f'config L{configured_layer}')
axes[0].set_title('Aggregate recovery by layer')
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Mean recovery')
axes[0].legend(fontsize=8)

axes[1].bar(layers, mean_prob, color='slateblue', alpha=0.85)
axes[1].axvline(best_aggregate_prob, color='crimson', linestyle='--', linewidth=2, label=f'best prob L{best_aggregate_prob}')
axes[1].axvline(configured_layer, color='tab:green', linewidth=2, label=f'config L{configured_layer}')
axes[1].set_title('Aggregate restored target probability')
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Mean P(target)')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()

aggregate_table = pd.DataFrame({
    'layer': layers,
    'mean_recovery': mean_recovery,
    'std_recovery': std_recovery,
    'mean_restored_probability': mean_prob,
}).sort_values('mean_recovery', ascending=False)
display(aggregate_table.head(10))


# %% [markdown]
## 9. Optional Save

# %%
if SAVE:
    out_dir = ROOT / 'analysis_out' / 'causal_implementation' / f'{MODEL_CONFIG}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / 'summary.csv', index=False)
    aggregate_table.to_csv(out_dir / 'aggregate_layers.csv', index=False)
    if skips:
        pd.DataFrame(skips).to_csv(out_dir / 'skipped.csv', index=False)
    np.savez_compressed(
        out_dir / 'layer_arrays.npz',
        recovery_matrix=recovery_matrix,
        probability_matrix=prob_matrix,
        mean_recovery=mean_recovery,
        mean_probability=mean_prob,
    )
    print(out_dir)
else:
    print('Set SAVE = True to write CSV and raw arrays under analysis_out/causal_implementation/')
