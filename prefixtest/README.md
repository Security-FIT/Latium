## Prefixtest

This folder contains the prefix/template variability experiment and its local
visualization outputs.

Included here:

- `run_remote.sh`: remote upload/run/fetch helper for the experiment
- `prefixtest.ipynb`: thin notebook visualizer
- `artifacts/`: selected local JSON/CSV artifacts for this experiment
- `output/`: generated graphs and summary tables to keep with the experiment

Canonical implementation now lives under:

- `src.experiments.prefix_variability/`
- `src.graphs/prefix.py`
- `src.graphs/`

### Local usage

```bash
python -m src.experiments.prefix_variability.cli --model Qwen/Qwen3-8B --case-idx 0
```

### Remote usage

```bash
RUN_NAMES="self_short self_long self_with_k_hints template_short template_alt_subject external_fact_target external_fact_relation_long external_fact_contrast_long" \
./prefixtest/run_remote.sh "Qwen/Qwen3-8B" 0
```

### Notebook usage

Open `prefixtest/prefixtest.ipynb` and run the cells. It will load the latest
artifact from either `prefixtest/artifacts/` or `analysis_out/`.
