# W&B tracking

Tracking defaults to `none`. Authenticate with `wandb login`, then enable it:

```bash
python -m src structural run \
  'structural.run.models=[gpt2-large]' \
  structural.tracking.provider=wandb \
  structural.tracking.project=latium
```

Gram fleet runners accept `--tracking wandb`. For offline structural runs, set
`structural.tracking.mode=offline`, then run `wandb sync <run-directory>`.

| Fields | Meaning |
|---|---|
| `monitor/status`, `monitor/heartbeat`, `monitor/stage` | Process status and current stage. |
| `monitor/seconds_since_activity` | Time since an operation completed. |
| `progress/edit`, `progress/edit_total` | Current edit and total selected facts. |
| `counterfact/*` | Dataset row, case ID, subject and rewrite targets. |
| `analysis/success_rate`, `analysis/methods/<method>/success_rate` | Analysis accuracy and per-method summaries. |

Optional settings: `structural.tracking.entity`, `group`, `run_name`, `tags`
and `heartbeat_seconds`.
