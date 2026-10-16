# Saved inputs for metrics and graphs

`gram-artifacts.zip` contains the experiment results used for metrics and graphs:

| Directory | Runs | Description |
| --- | ---: | --- |
| `classic-gram-merged` | 17 | 1,000 independent edits per base model |
| `finetune-covariance` | 744 | One edit per finetuned checkpoint, with its own covariance |
| `finetune-no-cov` | 790 | One edit per checkpoint, using base-model covariance |

The paper evaluates 17 base models and finetuned checkpoints with
checkpoint-specific covariance. The base-covariance cohort supports the
checkpoint selection analysis and comparison between covariance settings.

Each run contains baseline and ROME execution records, baseline and edited
GRAM captures and analyses, per-case ROME ES/PS/NS, configured edit layers,
detected layers and full layer scores. Configurations, fact selections,
checkpoint indexes and provenance support independent inspection of the
reported numbers.

`INDEX.json` at the archive root lists file sizes and SHA-256 digests.
`SOURCE-INDEX.json/csv` describe source-file provenance. The scripts resolve
input paths relative to the extracted archive; computation metadata can also
contain the absolute paths used during the experiments.

To run GPU experiments, follow the main README and use the repository's
model configurations and fact manifests.

## Compute metrics (Python standard library only)

From the repository root:

```bash
mkdir -p analysis_out/graph-inputs
python3 -m zipfile -e reproduction/gram-artifacts.zip analysis_out/graph-inputs
python3 scripts/summarize_rome_gram.py \
  analysis_out/graph-inputs/classic-gram-merged \
  analysis_out/graph-inputs/finetune-covariance \
  analysis_out/graph-inputs/finetune-no-cov \
  --by-family > analysis_out/results.csv
```

Omit `--by-family` for individual checkpoint rows. All completed edits count,
including unsuccessful ROME edits. ES/PS/NS and rates are percentages; S is
the harmonic mean of aggregate ES, PS and NS, before rounding. MAE uses actual
layer distances. Lift divides mean edited target score by mean baseline target
score. Unedited FP counts each baseline once.

## Regenerate figures without downloading models

Use the normal installed Latium environment, or a CPU environment:

```bash
python3 -m venv .venv-graphs
. .venv-graphs/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy matplotlib
python scripts/render_saved_results.py
```

The command verifies archive and input checksums, extracts into
`analysis_out/saved-results/inputs/`, and writes separate outputs:

- `per-checkpoint.csv` and `per-family.csv`: metrics for all models/campaigns.
- `native-renderer/`: existing Latium GRAM reports for every classic model.
- `figures/`: baseline/edited profiles with mean, min–max and quartile range;
  configured edit layers; linear/log finetuned profiles; success/FP charts;
  and a four-panel `gram_profiles` figure. Every plot has PDF and PNG versions.
- `verification.json`: input verification and aggregate counts.

Set `--output PATH` to choose a different output directory. `bundle.json`
records the archive checksum and size.

The four paper notebooks and their figure definitions are in
[notebooks/paper/](../notebooks/paper/README.md). Run
`python scripts/reproduce_paper.py` after installing their requirements to
rebuild the notebook inputs from this ZIP, execute the checks and generate the
paper PDF/PNG figures and LaTeX tables.
