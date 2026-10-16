# Paper notebooks

These four notebooks compute the paper's tables and figures from the experiment
data in `reproduction/gram-artifacts.zip`. Notebook 03 defines the plot colors,
shading, panel layout and typesetting. Notebook 02 checks the four DeepSeek
exceptions to the baseline-layer rule and computes MAE from actual distances.

| Notebook | Output |
| --- | --- |
| `01_rome_efficacy.ipynb` | ROME efficacy, Tables 1–2 and fact-cohort checks |
| `02_detection_base_models.ipynb` | Base-model detection, Table 3 and localization checks |
| `03_figure_score_profiles.ipynb` | Paper Figure 3 and supplementary profiles of all 17 models |
| `04_finetuned_checkpoints.ipynb` | Own-covariance finetunes, Table 4 and selection checks |

## Run everything from the original artifacts

From the repository root, in the installed Latium environment:

```bash
python -m pip install -r notebooks/paper/requirements.txt
python scripts/reproduce_paper.py
```

For a fresh CPU-only environment, install CPU PyTorch first as described in
`reproduction/README.md`, then install the requirements above. No GPU, model
weights or causal-tracing files are required.

The command verifies/extracts the ZIP, computes the notebook input tables from
the raw execution and GRAM records, and executes all four notebooks. A failed
cell or comparison fails the command.

Results are written to `analysis_out/paper/`:

- `outputs/gram_profiles.pdf` and `.png`: the four-panel paper figure.
- `outputs/gram_profiles_all_models.pdf` and `.png`: all 17 paper models.
- `outputs/table2_rows.tex`, `table3_rows.tex`, `table4_rows.tex`: Tables 2–4.
- `outputs/checks_*.csv`: every checked paper value and its computed result.
- `notebooks/`: executed notebooks and HTML exports.
- `verification.json`: summary of all notebook checks.

PDF and PNG versions of the paper figures are available in `figures/`.
Use `--output PATH` for another output
directory. If `render_saved_results.py` has already finished, reuse it with
`python scripts/reproduce_paper.py --saved-results analysis_out/saved-results`.

For interactive use, run `render_saved_results.py` first and open the notebooks.
Their default input directory is `analysis_out/saved-results/paper-inputs/` and
their default output directory is `analysis_out/paper/outputs/`. Environment
variables `CSF_GRAPHS_DIR` and `CSF_PAPER_OUTPUT` override those locations.
