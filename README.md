# Latium

Latium applies ROME model edits and uses GRAM to locate the edited layer.
Developed by Security@FIT in collaboration with Red Hat Research.

Main contributors: **Matej Olexa** and **Jakub Res**.

## Install

Use Linux, Python 3.11 or newer, and an NVIDIA GPU with enough memory for the
chosen model. These commands work in an ordinary SSH session:

```bash
git clone --depth 1 --branch CSF-submission https://github.com/Security-FIT/Latium.git
cd Latium
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your GPU driver if the default build
does not suit your machine. Public models need no authentication. For gated
or private models, obtain access and set `HF_TOKEN` or run `hf auth login`.
Set `HF_HOME` to change the Hugging Face download cache.

## Run ROME and GRAM

Start with GPT-2 Medium and one CounterFact edit:

```bash
python -m src second-moment model=gpt2-medium model.second_moment_target_samples=32
python -m src structural run structural=gram \
  'structural.run.models=[gpt2-medium]' structural.run.n_tests=1 \
  structural.run.run_id=example structural.render.enabled=true
```

The first command downloads the model and Wikitext, computes covariance
statistics, and saves them in `data/second_moment_stats/`. The second downloads
CounterFact, analyzes the unedited model, applies ROME, runs GRAM and renders
the results. Original weights are restored between independent edits.

The 32-text covariance setting is a quick example. Omit
`model.second_moment_target_samples=32` to use 100,000 texts. Increase
`structural.run.n_tests` to edit more facts; `structural.run.start_idx` selects
the starting dataset index. Use a new run ID for a separate run.

Model configurations live in `src/config/model/`. For another configured model,
replace `gpt2-medium` in both commands. Saved covariance is found automatically.
If several covariance files exist, select one with `second_moment_path` in the
model YAML.
ROME optimizer and prefix-generation defaults are in
`src/config/model_base/default.yaml`; the edit layer is set in each model YAML.

## Model configurations

Use a config key from this table in both commands above:
`model=gpt2-large` and `'structural.run.models=[gpt2-large]'`, for example.
Each key links to its YAML configuration.

| Config key | Model |
|---|---|
| [gpt2-medium](src/config/model/gpt2-medium.yaml) | `openai-community/gpt2-medium` |
| [gpt2-large](src/config/model/gpt2-large.yaml) | `openai-community/gpt2-large` |
| [gpt2-xl](src/config/model/gpt2-xl.yaml) | `gpt2-xl` |
| [gpt-j-6b](src/config/model/gpt-j-6b.yaml) | `EleutherAI/gpt-j-6B` |
| [opt-6.7b](src/config/model/opt-6.7b.yaml) | `facebook/opt-6.7b` |
| [llama2-7b](src/config/model/llama2-7b.yaml) | `NousResearch/Llama-2-7b-hf` |
| [falcon-7b](src/config/model/falcon-7b.yaml) | `tiiuae/falcon-7b` |
| [deepseek-7b-base](src/config/model/deepseek-7b-base.yaml) | `deepseek-ai/deepseek-llm-7b-base` |
| [mistral-7b-v0.1](src/config/model/mistral-7b-v0.1.yaml) | `mistralai/Mistral-7B-v0.1` |
| [mistral-7b-v0.3](src/config/model/mistral-7b-v0.3.yaml) | `mistralai/Mistral-7B-v0.3` |
| [ministral-3-8b](src/config/model/ministral-3-8b.yaml) | `mistralai/Ministral-3-8B-Base-2512` |
| [qwen3-4b](src/config/model/qwen3-4b.yaml) | `Qwen/Qwen3-4B` |
| [qwen3-8b](src/config/model/qwen3-8b.yaml) | `Qwen/Qwen3-8B` |
| [qwen3.5-4b](src/config/model/qwen3.5-4b.yaml) | `Qwen/Qwen3.5-4B` |
| [gemma-4-12b](src/config/model/gemma-4-12b.yaml) | `google/gemma-4-12B` |
| [granite4-micro](src/config/model/granite4-micro.yaml) | `ibm-granite/granite-4.0-micro` |
| [granite-4.1-8b](src/config/model/granite-4.1-8b.yaml) | `ibm-granite/granite-4.1-8b-base` |
| [olmo-3-1025-7b](src/config/model/olmo-3-1025-7b.yaml) | `allenai/Olmo-3-1025-7B` |

For a finetuned checkpoint, copy its architecture's model YAML, set `name` to
the checkpoint's Hub ID or local directory, and choose its edit layer. To load
a LoRA adapter, set `name` to its downloaded directory and `adapter_base_path`
to its downloaded base checkpoint directory; the adapter is merged before
editing and analysis. Compute covariance using this configuration for the
finetuned model's own statistics, or set `second_moment_path` to compatible
base-model statistics to reuse them.

## Paper experiments

### The same 1,000 facts

To use the same 1,000 CounterFact facts, retrieve the saved manifest from `main`:

Use the generated 100,000-text covariance file for the structural run. If you
previously computed the 32-text example, set `second_moment_path` in the model
YAML to the new file's printed path before running the structural command.

```bash
git fetch origin main:refs/remotes/origin/main
mkdir -p inputs
git show origin/main:manifests/counterfact_seed42_n1000.json \
  > inputs/counterfact_seed42_n1000.json

MODEL=gpt2-medium
python -m src second-moment model="$MODEL" model.second_moment_target_samples=100000
python -m src structural run structural=gram \
  "structural.run.models=[$MODEL]" \
  structural.run.case_index_file=inputs/counterfact_seed42_n1000.json \
  structural.run.start_idx=0 structural.run.n_tests=1000 \
  "structural.run.run_id=paper-$MODEL" structural.render.enabled=true
```

Replace `MODEL` with any config key above to run the same facts on another model.
The manifest supplies the saved fact IDs and order. `start_idx` is a position
in that manifest; use `start_idx=0, n_tests=500` and `start_idx=500, n_tests=500`
with distinct run IDs to split it into two batches.

### Finetuned checkpoints

The saved selections in [manifests/paper/](manifests/paper/) contain checkpoint
and adapter-base revisions, edit layers, and one assigned fact per checkpoint.
They cover up to 100 finetunes per family: 790 evaluated checkpoints for base
covariance, and the 747 selected checkpoints for own covariance. The shared
[100-fact manifest](manifests/paper/rome-facts-100.json) supplies their assigned
facts; this is a separate selection from the classic 1,000 facts.

Run a family with either covariance mode:

```bash
python -m src.finetunes --family gpt2-medium --covariance base
python -m src.finetunes --family gpt2-medium --covariance own
```

Replace `gpt2-medium` with a family listed in
[finetunes-base](manifests/paper/finetunes-base/) or
[finetunes-own](manifests/paper/finetunes-own/). Add `--limit 1` to run only the
first saved checkpoint. No new checkpoint discovery or fact sampling is used.
These commands use the current ROME settings and self-generated prefixes.

Base covariance is computed once for the family; own covariance is computed
on each finetuned checkpoint. Both use 100,000 texts. Each checkpoint gets its
own unedited GRAM baseline, one assigned ROME edit, edited GRAM and graphs.
Downloaded weights are removed after each checkpoint. Results are saved under
`analysis_out/finetunes-<mode>-<family>-<timestamp>/runs/<checkpoint>/`;
`--output PATH` selects a new output directory. Failed checkpoints are recorded
in `failures.json`, and remaining checkpoints continue.

## Outputs and rendering

The example writes `analysis_out/example/`. It contains execution records,
baseline and edited GRAM results, per-case ROME scores, and resolved settings.
Plots and JSON/CSV reports are under `graphs/gram-report/`.

Rebuild plots from a saved run without loading the model:

```bash
python -m src graphs run analysis_out/example
```

ROME success means an efficacy score of 1. GRAM accuracy requires the selected
layer to equal the configured edit layer exactly. Baseline FP means the
unedited checkpoint selects its future configured edit layer.

## Causal tracing

```bash
python -m src causal-trace model=gpt2-medium
```

Results are written under `analysis_out/causal_trace/`. See
[causal_tracing.md](causal_tracing.md) for method settings and output details.
Run `python -m src` for other commands.
