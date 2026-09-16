# W&B monitoring

The structural pipeline has optional experiment tracking through a small provider boundary in
`src/tracking.py`. Tracking is disabled by default, so existing commands and programmatic callers do
not import or initialize W&B.

## Enable it

Install the requirements, authenticate once with `wandb login`, and add these Hydra overrides to a
structural run:

```bash
python -m src structural run \
  structural.tracking.provider=wandb \
  structural.tracking.project=latium \
  structural.tracking.entity=YOUR_ENTITY \
  structural.tracking.run_name=gpt2-large-detector-check
```

Use `structural.tracking.mode=offline` on a worker without network access, then run `wandb sync` on the
generated W&B directory later. Optional settings are `structural.tracking.group`,
`structural.tracking.tags`, and `structural.tracking.heartbeat_seconds`.

## What is recorded

One W&B run spans capture, edits, analysis, and rendering. Its latest values answer the operational
questions directly:

| Question | W&B fields |
| --- | --- |
| Is the process running? | `monitor/running`, `monitor/status`, `monitor/heartbeat`, `monitor/uptime_seconds` |
| Could it be stuck? | `monitor/seconds_since_activity`, `monitor/stage`, `monitor/substage` |
| Which edit is active? | `progress/edit`, `progress/edit_total`, `context/case_id`, `context/model`, `context/plan` |
| Is a detector or localizer active? | `context/analysis`, `analysis/index`, `analysis/status` |
| What is ROME doing? | `rome/loss`, `rome/prediction_loss`, `rome/kl_divergence`, `rome/weight_decay`, `rome/delta_norm`, and final update norms |

The heartbeat is emitted by a background thread even while one edit or detector call is taking a long
time. `monitor/seconds_since_activity` measures time since the pipeline last advanced or emitted a ROME
optimization metric. A rising value with a continuing heartbeat means the process is alive inside the
same operation; a stopped heartbeat means the worker or process is no longer reporting.

W&B also captures the existing Python logs, including the standard ROME prefix and optimization output.
The tracker deliberately avoids recording prompts, subjects, target text, or the Hugging Face token.

## Branch compatibility

The instrumentation points are shared by `feat/paper-improvements` and `simplify-code`:

1. `src/structural/runner.py` owns the W&B run lifecycle.
2. `src/structural/execution/edit_execution.py` records edit progress.
3. `src/rome/common.py` records the existing ROME optimization values.
4. `src/structural/analysis/runtime.py` records every registered analysis by identifier.

Paper-only detector and localizer variants require no W&B-specific extension. They pass through the
same analysis registry and appear under `context/analysis`. New long-running work should only call
`current_tracker().set_state(...)` at the start of a stage and `current_tracker().log(...)` for useful
numeric results.

## Suggested W&B workspace

Create four panels:

1. A run table with `monitor/status`, `monitor/stage`, `monitor/substage`, `progress/edit`,
   `progress/edit_total`, and `context/analysis`.
2. A line chart for `monitor/heartbeat` and `monitor/seconds_since_activity`.
3. A line chart for the `rome/*` loss values, grouped by run.
4. A run table with `edit/success` and the final `analysis/*` case counts.

The first implementation does not send automatic stuck alerts because the expected duration of model
loads, edits, and localizers differs substantially by model. Once normal durations are known, a W&B
automation can alert on a chosen `monitor/seconds_since_activity` threshold.
