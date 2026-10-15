# Saved inputs for metrics and graphs

`gram-artifacts.zip` contains the accepted, merged experiment results:

| Directory | Runs | Description |
| --- | ---: | --- |
| `classic-gram-merged` | 18 | 1,000 independent edits per base model |
| `finetune-covariance` | 744 | One edit per finetuned checkpoint, with its own covariance |
| `finetune-no-cov` | 790 | One edit per checkpoint, using base-model covariance |

The paper uses 17 base models; the additional Gemma run remains available in
the archive. Own-covariance results are the final finetuned experiment. The
base-covariance cohort is included for comparison and selection analysis.

The original directory layout and experiment data are retained: baseline and
ROME execution records, baseline and edited GRAM captures and analyses, cases,
per-case ROME ES/PS/NS, configured edit layers, detected layers, full layer
scores, configurations, checkpoint indexes, selections and provenance. These
files allow independent inspection of the reported numbers, beyond the CSV
summaries. Scientific payloads are copied byte for byte, without reducing
their fields or keeping only aggregate values.

Only causal tracing, logs, generated reports/images and duplicate code/job
archives are omitted. Run manifests remove entries for these omitted artifacts;
all other data is retained. `SOURCE-INDEX.json/csv` preserve original source
catalogs, which can include references to omitted files. Original absolute
paths in metadata are also preserved for provenance; the supplied scripts
resolve checkpoint inputs relative to the extracted archive.

No model weights or covariance tensors are included. GPU experiments can be
repeated using the main README and the repository's model/fact manifests.

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

Set `--output PATH` to choose a different output directory. Original archived
inputs are never modified. `INDEX.json` inside the archive lists every input's
SHA-256; `bundle.json` records the archive checksum and size.
