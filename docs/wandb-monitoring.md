# W&B monitoring

W&B tracking is optional and disabled by default. Install the project requirements and authenticate
the machine once:

```bash
pip install -r requirements.txt
wandb login
```

Enable it for a structural run:

```bash
python3 -m src structural run \
  'structural.run.models=[gpt2-large]' \
  structural.tracking.provider=wandb \
  structural.tracking.project=latium \
  structural.tracking.entity=YOUR_ENTITY \
  structural.tracking.run_name=gpt2-large-check
```

Use `structural.tracking.mode=offline` without network access, then upload the resulting directory with
`wandb sync <offline-run-directory>`.

## What to watch

- **Liveness:** `monitor/status`, `monitor/heartbeat`, `monitor/stage`, and
  `monitor/seconds_since_activity`.
- **Current CounterFact row:** `counterfact/index`, `counterfact/case_id`, `progress/edit`, and
  `progress/edit_total`.
- **Current rewrite:** `counterfact/fact_tuple`, `counterfact/subject`, `counterfact/target_true`,
  `counterfact/target_new`, `counterfact/original_text`, and `counterfact/edited_text`.
- **Detector/localizer:** `context/analysis`, `analysis/status`, `analysis/success_rate`,
  `analysis/successes`, and `analysis/evaluated`.
- **Per-method summary:** `analysis/methods/<method>/success_rate`.
- **Paper experiment methods:** `analysis/experiments/<method>/success_rate`.

Localization success means that the predicted layer equals the edited layer. Binary detection success
means that an edited execution is detected and a baseline execution is not detected.

The heartbeat continues during a long edit or analysis. If the heartbeat advances while
`monitor/seconds_since_activity` rises, the process is alive but has not finished its current operation.
If the heartbeat stops, the process is no longer reporting.

Optional settings are `structural.tracking.group`, `structural.tracking.tags`, and
`structural.tracking.heartbeat_seconds`.
