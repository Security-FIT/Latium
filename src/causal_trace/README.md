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
