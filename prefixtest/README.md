## Prefixtest

This folder contains the prefix/template variability experiment and its local
visualization outputs.

Included here:

- `experiment.py`: the prefix/template variability experiment runner
- `run_remote.sh`: remote upload/run/fetch helper for the experiment
- `prefixtest.ipynb`: thin notebook visualizer
- `prefixtest_support.py`: plotting and artifact-loading helpers
- `artifacts/`: selected local JSON/CSV artifacts for this experiment
- `output/`: generated graphs and summary tables to keep with the experiment

Shared dependencies remain outside this folder:

- `structural_benchmark.py`
- `src/`
- `notebooks/paper_graphs_support.py`
- `notebooks/new-gen/`

### Local usage

```bash
python prefixtest/experiment.py --model Qwen/Qwen3-8B --case-idx 0
```

### Remote usage

```bash
RUN_NAMES="self_short self_long self_with_k_hints template_short template_alt_subject external_fact_target external_fact_relation_long external_fact_contrast_long" \
./prefixtest/run_remote.sh "Qwen/Qwen3-8B" 0
```

### Notebook usage

Open `prefixtest/prefixtest.ipynb` and run the cells. It will load the latest
artifact from either `prefixtest/artifacts/` or `analysis_out/`.
