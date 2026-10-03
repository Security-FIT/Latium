# Causal Trace Package

The canonical method, configuration, outputs, limitations, and full
causal-to-ROME workflow are documented in
[`../../causal_tracing.md`](../../causal_tracing.md). This package README is
only an implementation entry point; it does not define a second version of the
method.

`tokenization.py` maps exact model-input token positions, `selection.py` owns
window statistics and held-out selection, and `causal_trace.py` owns model
execution and output artifacts.

## Standalone CLI

```bash
python3 -m src causal-trace model=gpt2-xl command.causal_trace.num_valid_facts=100
```

To persist a held-out-confirmed trace center as the selected model config's
ROME layer, opt in explicitly:

```bash
python3 -m src causal-trace model=gpt2-xl command.causal_trace.overwrite_model_config_layer=true
```

The config is not changed when tracing does not produce a confirmed selection.
The summary records whether the overwrite occurred and the old and new layers.

The cluster pipeline reads the confirmed center, computes matching second
moments when needed, and runs the ROME-only benchmark using runtime overrides.
It does not modify model YAML files or run structural detectors:

```bash
jobs/submit.sh causal-rome -- pipeline.model=gpt2-xl
```

## Historical token-by-block variants

Two additional commands preserve the legacy tracing experiment: corrupt subject
embeddings and restore one subject token at one whole transformer block at a
time. They are separate from the existing `causal-trace` MLP-window method.

```bash
# Original algorithm, copied unchanged from origin/legacy at 357c51b.
python3 -m src causal-kuba model=gpt2-xl generation.num_of_runs=100

# Same experiment with the implementation errors corrected.
python3 -m src causal-kuba-fix model=gpt2-xl command.legacy_trace.num_valid_facts=100

# Explicitly increase the fixed variant's paired noise draws per fact.
python3 -m src causal-kuba-fix model=gpt2-xl \
  command.legacy_trace.num_valid_facts=100 \
  command.legacy_trace.num_noise_samples=10
```

`causal-kuba` preserves the original algorithm, including its target/span checks,
unpaired noise and historical hook behavior. It uses the current shared model
loader and handler, rather than recreating the historical runtime. Its default
is 100,000 successful runs; set `generation.num_of_runs` explicitly. The wrapper
creates the output directory, with CSVs under `analysis_out/causal_kuba` by
default; `generation.filename` overrides the filename prefix.

`causal-kuba-fix` uses deterministic paired noise for the corrupted baseline and
all restorations, validates unique subject spans, captures actual block outputs
instead of potentially normalized hidden states, restores token rows with the
correct shape/device/dtype, and cleans up its temporary hooks even on failure.
It measures the target's **first continuation token**, including when the full
target contains multiple tokens. Each draw shares one noise vector across the
subject tokens, as in the legacy protocol.

Fixed-variant options live in `command.legacy_trace.*`. The default is 100 facts
and one noise draw per fact. `noise_std` is an absolute standard deviation;
when null it uses `model.corruption_noise_multiplier`, falling back to three
times the embedding-weight standard deviation only if that value is missing.
Outputs are per-run directories under `analysis_out/causal_kuba_fix`, containing
`summary.json`, per-fact JSON, `traces.csv`, and the token/layer `profile.csv`.
Use `command.legacy_trace.output_dir` to change the destination. The summary
records rejected facts and marks incomplete cohorts `insufficient_valid_facts`.

Neither variant selects a ROME layer automatically, modifies model YAML,
computes covariance, or performs a ROME edit. A whole-block restoration peak
is a candidate for testing, not proof of the best MLP weight-editing site.
