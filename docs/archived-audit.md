# Live MetaCentrum artifact audit — 12 October 2026

Both accounts were accessed live over SSH. Remote operations were read-only. The exact selected paths, file sizes and SHA-256 byte hashes are retained in this folder.

## Coverage

| Campaign | Account | Completed ROME + baseline GRAM + edited GRAM | Accepted causal tracing |
|---|---|---:|---:|
| Base models | olexamatej | 18,000 edits, 18 × 1,000 | 1,700 profiles, 17 × 100 |
| Fine-tunes/adapters using base covariance | olexamatej | 790 checkpoint workflows | 766 checkpoint traces |
| Fine-tunes/adapters using own covariance | xsajko01 | 744 of 747 frozen checkpoints | 727 checkpoint traces |

## What was verified

- Every selected workflow was resolved through its saved run manifest. The six primary JSON outputs are baseline execution, baseline normal-GRAM capture and analysis, ROME execution, and edited normal-GRAM capture and analysis.
- Selected primary outputs were parsed and checked against manifest identities, statuses, content hashes, configuration hashes and input references. Complete ROME case IDs were checked for paired edited GRAM capture and analysis records.
- Base-model selection uses corrected DeepSeek/GPT-J/Ministral first halves and the 24-case Qwen3.5 repair. The original Qwen error records are retained as history; the repair replaces them by case ID.
- Accepted checkpoint traces were selected from final saved trace states, checked against their manifests, and their referenced raw JSON/summary/plot files were checked for existence.
- Own-covariance live file bytes were compared to all 5,191 artifacts in authoritative-artifacts.csv: 5,191 matched.
- 744 own-covariance computation/deletion receipts were checked for checkpoint revision, fresh computation, 100,000 target samples, and deletion. All 81 tasks have empty covariance/download leftover files.
- 21,831 distinct JSON files were read. Unexpected integrity issues in completed selection: 0.

## Exact locations

Base-model roots:

```text
/storage/brno2/home/olexamatej/Latium-gram-n100-95fc3f7-20261003/analysis_out/gram-first500-20261003
/storage/brno2/home/olexamatej/Latium-gram-retry-d7ec7ab-20261004/analysis_out/gram-first500-retry-20261004
/storage/brno2/home/olexamatej/Latium-base-brno2-097ce88-20261009/results/next500
/storage/brno2/home/olexamatej/Latium-qwen35-retry24-097ce88-20261010/results
```

Fine-tune/base-covariance roots:

```text
/storage/brno2/home/olexamatej/Latium-shared-facts-20261005
/storage/brno2/home/olexamatej/Latium-quota-retry-20261006
/storage/brno2/home/olexamatej/Latium-scratch-remainder-20261007-155240
/storage/brno2/home/olexamatej/Latium-remaining-eligible-20261008-171001
/storage/brno2/home/olexamatej/Latium-remaining-lgm-20261009-095423
/storage/brno2/home/olexamatej/Latium-remaining-olmo-20261009-101748
```

Own-covariance root:

```text
/storage/brno2/home/xsajko01/Latium-owncov-20261010/fleet-worked747-40gb-r2/runs/
```

Base causal-tracing root (select legacy-fixed only):

```text
/storage/brno2/home/olexamatej/runs/causal-comparison-20261002-expanded
```

## Limits and gaps

- The checkpoint campaigns do not contain 100 completed fine-tunes per family. See family-coverage.csv for observed counts. Gemma has base-model artifacts but no selected checkpoint workflow in either checkpoint campaign.
- Base tracing has 100 selected profiles for 17 families. Missing comparable fixed-legacy family: deepseek-7b-base.
- Checkpoint causal traces use a separate tracing fact pool. They are pre-ROME measurements; no post-ROME causal traces were found in the selected manifests and output paths inspected.
- The 747-checkpoint own-covariance fleet reuses a frozen subset from the earlier evaluated campaign; the campaigns must remain separate experiments.
- Own-covariance tensors and downloaded weights were removed after processing. The retained artifacts are computation logs/state, matrix filenames, cleanup checksums, and deletion receipts. The original tensor bytes cannot be recovered from these receipts.
- ROME efficacy failures are evaluated outcomes and remain in the completed workflow counts. They are different from download/runtime failures.
- Base unedited GRAM is a model-level control. The selected files retain 37 batch/repair baseline outputs for 18 base model families; those are not 18,000 independent controls.
- This inventory audits the permanent Brno2 locations identified by the supplied guides. It is not an exhaustive search of every MetaCentrum storage backend or expired scratch directory.

Own-covariance download failures:

- `CompassioninMachineLearning/Olmo-3-7b-final-CPT-10k-urban-density-dataset` — RepositoryNotFoundError: 404 Client Error. (Request ID: Root=1-6acaf076-43aff91b7dffa87566301912;26dabe7d-3cc2-41d7-aade-ffe46761a615)
- `CompassioninMachineLearning/Olmo-3-7b-intermediate-epoch-3-CPT-10k-urban-density-dataset` — RepositoryNotFoundError: 404 Client Error. (Request ID: Root=1-6acaf078-1ab452bd4d7cf7ff1176c541;0c4f27fb-e71c-45a9-b9da-00b2c4627387)
- `michaljach/jet` — RepositoryNotFoundError: 404 Client Error. (Request ID: Root=1-6acafec5-55a73475398fab3003241435;76941e3e-ba03-4a21-971c-0c017459f3c3)

Own-covariance workflows with unavailable tracing:

- `netrias/alcohol0_bacteria100_harmonization_gpt2large` (gpt2-large)
- `netrias/alcohol75_bacteria25_harmonization_gpt2large` (gpt2-large)
- `netrias/alcohol50_bacteria50_harmonization_gpt2large` (gpt2-large)
- `netrias/alcohol25_bacteria75_harmonization_gpt2large` (gpt2-large)
- `payelb/GPT2L_full` (gpt2-large)
- `TomData/GPT2-review` (gpt2-medium)
- `trojblue/gpt2-prompt-upscaler-v1` (gpt2-medium)
- `dakwi/chessgpt-medium-l` (gpt2-medium)
- `CodeferSystem/GPT2-Hacker-password-generator-Medium` (gpt2-medium)
- `jarod0411/zinc10M_gpt2-medium_SMILES_step1` (gpt2-medium)
- `Pomilon/LEMA-llama-2-7b` (llama2-7b)
- `predibase/dbpedia` (mistral-7b-v0.1)
- `predibase/covid` (mistral-7b-v0.1)
- `hin123123/theralingua-mistral-7b-word` (mistral-7b-v0.3)
- `klcsp/mistral-7b-0.3-gpt4o_100k_closedqa-fft` (mistral-7b-v0.3)
- `klcsp/mistral-7b-0.3-gpt4o_100k_summarize-fft` (mistral-7b-v0.3)
- `klcsp/mistral-7b-0.3-gpt4o_100k_coding-fft` (mistral-7b-v0.3)

## Per-family coverage

| Family | Base edits | Base trace profiles | Shared checkpoint workflows | Shared traces | Owncov attempted | Owncov workflows | Owncov traces |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-7b-base | 1000 | 0 | 8 | 8 | 8 | 8 | 8 |
| falcon-7b | 1000 | 100 | 37 | 37 | 36 | 36 | 36 |
| gemma-4-12b | 1000 | 100 | 0 | 0 | 0 | 0 | 0 |
| gpt-j-6b | 1000 | 100 | 10 | 10 | 10 | 10 | 10 |
| gpt2-large | 1000 | 100 | 50 | 43 | 48 | 48 | 43 |
| gpt2-medium | 1000 | 100 | 66 | 59 | 63 | 63 | 58 |
| gpt2-xl | 1000 | 100 | 23 | 23 | 21 | 21 | 21 |
| granite-4.1-8b | 1000 | 100 | 4 | 4 | 4 | 4 | 4 |
| granite4-micro | 1000 | 100 | 59 | 59 | 57 | 57 | 57 |
| llama2-7b | 1000 | 100 | 20 | 18 | 18 | 18 | 17 |
| ministral-3-8b | 1000 | 100 | 2 | 2 | 2 | 2 | 2 |
| mistral-7b-v0.1 | 1000 | 100 | 89 | 87 | 72 | 72 | 70 |
| mistral-7b-v0.3 | 1000 | 100 | 98 | 92 | 85 | 85 | 81 |
| olmo-3-1025-7b | 1000 | 100 | 72 | 72 | 72 | 70 | 70 |
| opt-6.7b | 1000 | 100 | 3 | 3 | 3 | 3 | 3 |
| qwen3-4b | 1000 | 100 | 86 | 86 | 85 | 85 | 85 |
| qwen3-8b | 1000 | 100 | 82 | 82 | 82 | 82 | 82 |
| qwen3.5-4b | 1000 | 100 | 81 | 81 | 81 | 80 | 80 |

## Audit files

- summary.json: coverage and validation totals.
- selected-campaigns.json: all selected run/checkpoint records and exact primary paths.
- checkpoint-workflows.csv: paired six-artifact selection per evaluated checkpoint.
- family-coverage.csv: per-family completed/attempted/trace counts.
- live-file-inventory.csv: exact paths, roles, byte sizes and live SHA-256 hashes.
- authoritative-checkpoints.csv / authoritative-artifacts.csv: source own-covariance indexes copied from the cluster.
- selected-output-sources-20261011.json / selected-output-validation-20261011.json: source base-model selection and prior validation copied from the cluster.
- audit_remote.py and run_remote.py: the read-only audit and SSH runner.

Check timestamps (UTC):

```json
{
  "olexamatej": {
    "started_at": "2026-10-12T12:10:38.053208+00:00",
    "finished_at": "2026-10-12T12:13:58.840262+00:00"
  },
  "xsajko01": {
    "started_at": "2026-10-12T12:10:38.184054+00:00",
    "finished_at": "2026-10-12T12:14:07.051473+00:00"
  }
}
```

## Retained historical failures and partial workflow

The four observations below are the repaired original Qwen execution status and the incomplete iko-01/iko-002 workflow. No unexpected hash/identity/input/file errors were found in the completed selection.

```json
[
  {
    "path": "/storage/brno2/home/olexamatej/Latium-gram-n100-95fc3f7-20261003/analysis_out/gram-first500-20261003/models/qwen3.5-4b/run/plans/qwen3.5-4b/cf_7ce8d8044134_m0000-0500_r01/methods/rome/execution.json",
    "issue": "primary_artifact_incomplete",
    "status": "error"
  },
  {
    "path": "/storage/brno2/home/olexamatej/Latium-shared-facts-20261005/analysis_out/finetuned-restart-first40-shared-20261005/gpt2-medium/models/fleet_iko-01_iko-002/run/plans/fleet_iko-01_iko-002/cf_4f9009281b60_m0030-0031_r01/methods/rome/captures/gram-localization.json",
    "issue": "primary_artifact_incomplete",
    "status": "error"
  },
  {
    "path": "/storage/brno2/home/olexamatej/Latium-shared-facts-20261005/analysis_out/finetuned-restart-first40-shared-20261005/gpt2-medium/models/fleet_iko-01_iko-002/run/plans/fleet_iko-01_iko-002/cf_4f9009281b60_m0030-0031_r01/methods/rome/analysis/detection/gram-localization/44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a.json",
    "issue": "primary_artifact_incomplete",
    "status": "unavailable"
  },
  {
    "path": "/storage/brno2/home/olexamatej/Latium-shared-facts-20261005/analysis_out/finetuned-restart-first40-shared-20261005/gpt2-medium/models/fleet_iko-01_iko-002/run",
    "issue": "completed_edit_missing_paired_gram",
    "case_id": "18518"
  }
]
```

## Additional retained evidence

- The older checkpoint campaign also retains one partial ROME/GRAM workflow: iko-01/iko-002 (GPT-2 Medium), case 18518. ROME produced non-finite weights/NLL metrics and its edited GRAM capture failed with "Editable projection contains non-finite values". Its baseline and causal trace exist, but it does not belong in the 790 completed workflows.
- Across the same roots, 786 accepted checkpoint traces exist: 766 paired with complete workflows, one for that partial workflow, and 19 trace-only checkpoint identities. The extra 20 must not inflate the main complete-workflow trace count.
- The original Qwen3.5 first-half run has no graphs/gram-report directory. Its raw outputs remain usable for the 476 completed cases; the repaired subset and second half have their own reports. Overall there are 36 saved base batch/repair graph directories for 37 selected run manifests.
- All 747/747 frozen own-covariance checkpoint identities and assigned ROME facts match the earlier successful checkpoint campaign.

The auxiliary check reread all 1,571 selected manifests to inspect registered causal-trace phases. It found zero registered post-ROME causal traces. All six standard GRAM report files exist for each of the 790 complete shared workflows and 744 own-covariance workflows. The only missing selected base report is the original Qwen3.5 first-half report listed above. Every evaluated own-covariance checkpoint has its saved YAML configuration, and no tensor remains in its permanent covariance directory. See auxiliary-validation.json.
