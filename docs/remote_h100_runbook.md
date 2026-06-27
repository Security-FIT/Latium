# Remote GPU Capture Runbook

The staged pipeline keeps model-dependent work on the GPU host and allows
artifact-only analysis on any machine with the repository environment.

## GPU Host

Check second moments:

```bash
python -m src command=structural/validate_cov \
  structural.run.models='[gpt2-large,qwen3-4b]' \
  structural.validate_cov.fail_missing=true
```

Capture one or more methods:

```bash
python -m src command=structural/capture \
  structural.run.models='[gpt2-large,qwen3-4b]' \
  structural.run.edit_methods='[rome]' \
  structural.run.n_tests=50 \
  structural.run.start_idx=30 \
  structural.capture.profile=paper \
  structural.capture.matrix_features.feature_set=paper \
  structural.run.output_dir=analysis_out \
  structural.run.run_id=n50-s30
```

This stage performs ROME evaluation and saves the primitives needed by the
common detectors. It does not run artifact-only detection.

Use `structural.capture.profile=spectral` when only spectral replay is needed,
`matrix` when only scalar matrix features are needed, and `none` when only edit
execution/evaluation JSON is required. Use `full` only when later analyses need
attention, matrix-anomaly, or bottom-rank primitives.

## Transfer

Copy the entire Run Root so `manifest.json` and all relative paths remain
together:

```bash
rsync -av user@gpu-host:/path/to/Latium/analysis_out/n50-s30/ \
  analysis_out/n50-s30/
```

Do not copy selected JSON files without the manifest.

## Analysis Host

Run artifact-only analyses:

```bash
python -m src command=structural/analyze \
  structural.analyze.run_root=analysis_out/n50-s30 \
  structural.analysis.preset=paper
```

Render selected outputs:

```bash
python -m src command=graphs/run \
  graphs.run_root=analysis_out/n50-s30 \
  graphs.renderer_preset=structural-paper
```

To rerun one detector after changing its Implementation:

```bash
python -m src command=structural/analyze \
  structural.analyze.run_root=analysis_out/n50-s30 \
  structural.analysis.preset=none \
  structural.analysis.enable='[spectral]' \
  structural.run.force=true
```

No model checkpoint or safetensors file is required for analysis. Only the run
root is transferred.
