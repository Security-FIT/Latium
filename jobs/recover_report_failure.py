#!/usr/bin/env python3
"""Repair report-only failures from verified saved artifacts without model loading."""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def stamp():
    return datetime.now(timezone.utc).isoformat()

def backup(path):
    destination = path.with_name(path.name + '.before-report-hotfix')
    if not destination.exists():
        destination.write_bytes(path.read_bytes())


def require(condition, message):
    # Recovery checks must also run when Python is invoked with -O.
    if not condition:
        raise ValueError(message)

def recover(root, *, graphs=True):
    from jobs import gram_fleet
    from jobs.finetuned_gram_fleet import rome_case
    from jobs.finetuned_trace import trace_implementation_hash
    from jobs.paper_fleet import run_logged, manifest_records
    from src.counterfact_selection import load_case_manifest
    from src.gram_experiment import write_json
    from src.graphs.gram import report_experiment
    from src.results import content_hash
    root = Path(root).resolve()
    receipt = {'kind': 'report-only-recovery', 'started_at': stamp(),
               'scientific_recomputation': False, 'recovered': [], 'errors': []}
    if not (root / 'checkpoints.json').is_file():
        return receipt
    selection = json.loads((root / 'checkpoints.json').read_text())
    entries = {entry['key']: entry for entry in selection['models']}
    cohort = load_case_manifest(root / 'cases.json')
    catalog = json.loads((root / 'experiment.json').read_text())
    tracing_hash = trace_implementation_hash()
    for path in sorted(root.glob('models/*/fleet-batches/*/state.json')):
        state = json.loads(path.read_text())
        if state.get('status') != 'failed' or state.get('error') != 'RuntimeError: Gram batch failed; see batches/*/state.json and fleet.log':
            continue
        key = path.parents[2].name
        try:
            entry = entries[key]
            require(all(state.get(k) == entry.get(k) for k in ('model_id', 'revision', 'fact_position', 'fact_case_id')),
                    'Checkpoint identity mismatch')
            require(state.get('trace_implementation') == tracing_hash, 'Tracing implementation mismatch')
            position = entry['fact_position']
            require(cohort['case_ids'][position] == entry['fact_case_id'], 'Assigned fact mismatch')
            covariance = state['covariance']; computation = covariance['computation']
            samples = selection['identity']['covariance_samples']
            require(selection['identity']['covariance_source'] == 'finetuned', 'Recovery requires finetuned covariance receipts')
            require(covariance['removed'] and covariance['target_samples'] == samples, 'Covariance lifecycle mismatch')
            require(computation['source'] == 'computed' and computation['target_samples'] == samples,
                    'Covariance computation mismatch')
            require(computation['checkpoint_revision'] == entry['revision'], 'Covariance revision mismatch')
            require(covariance['files'] and all(f['bytes'] > 0 and len(f['sha256']) == 64 for f in covariance['files']),
                    'Missing covariance digest receipts')
            batch = f'm{position:04d}-{position+1:04d}'
            inner_path = root / 'models' / key / 'batches' / batch / 'state.json'
            inner = json.loads(inner_path.read_text())
            require(inner.get('status') == 'failed' and 'gram-report failed' in inner.get('error', ''),
                    'Failure is not limited to report rendering')
            model_run = root / catalog['models'][key]['run_root']
            plan_id = catalog['models'][key]['batches'][batch]['plan_id']
            gram_fleet.verify_batch(model_run, key, plan_id, cohort, position, position+1)
            case = rome_case(model_run, plan_id)
            trace = state.get('trace', {})
            if trace.get('status') == 'complete':
                artifact = model_run / 'causal-kuba-fix' / f"m{trace['accepted_position']:04d}" / 'artifact.json'
                payload = json.loads(artifact.read_text())
                require(payload['status'] == 'complete', 'Tracing artifact is incomplete')
                _, records = manifest_records(model_run)
                matches = [r for r in records if r['path'] == str(artifact.relative_to(model_run))]
                require(len(matches) == 1 and matches[0]['content_hash'] == content_hash(payload),
                        'Tracing artifact content hash mismatch')
                require(all((model_run / p).is_file() for p in payload['summary']['outputs']),
                        'Tracing output is missing')
            if graphs:
                run_logged([sys.executable, '-m', 'src', 'graphs', 'run', str(model_run),
                            'graphs.renderer_preset=gram-report'], stage='gram-report-recovery', model=key)
            recovered_at = stamp()
            proof = {'recovered_at': recovered_at, 'original_status': state['status'],
                     'original_error': state['error'], 'scientific_artifacts_verified': True,
                     'covariance_recomputed': False, 'model_redownloaded': False,
                     'original_pbs_exit_status_preserved': True}
            backup(inner_path); backup(path)
            inner.update(status='complete', completed_at=recovered_at,
                         report_recovery={**proof, 'original_error': inner['error']})
            inner.pop('error', None); write_json(inner_path, inner)
            success = bool(case['edit']['success'])
            state.update(status='complete' if success else 'rome_failed', rome_success=success,
                         accepted_position=position, completed_at=recovered_at, report_recovery=proof)
            state.pop('error', None)
            for attempt in state.get('attempts', []):
                if attempt.get('position') == position:
                    attempt.update(status='complete' if success else 'rome_failed', execution=case,
                                   report_recovery=proof, completed_at=recovered_at)
                    if success: attempt.pop('error', None)
                    else: attempt['error'] = 'Efficacy evaluation did not pass'
            write_json(path, state)
            receipt['recovered'].append({'key': key, 'model_id': entry['model_id'],
                                         'fact_case_id': entry['fact_case_id'], 'status': state['status'],
                                         'state_path': str(path.relative_to(root))})
        except Exception as exc:
            receipt['errors'].append({'key': key, 'error': f'{type(exc).__name__}: {exc}'})
    if receipt['recovered']:
        report_experiment(root, graphs=graphs)
        budget_path = root / 'budget-summary.json'
        if budget_path.is_file():
            summary = json.loads(budget_path.read_text()); backup(budget_path)
            recovered = {r['model_id']: r for r in receipt['recovered']}
            for row in summary.get('checkpoints', []):
                if row['model_id'] in recovered:
                    row['report_recovery'] = {'original_returncode': row.get('returncode'),
                        'original_state_status': row.get('state_status'), 'recovery_returncode': 0,
                        'scientific_recomputation': False}
                    row['evaluated'] = True; row['state_status'] = recovered[row['model_id']]['status']
                    state_path = root / recovered[row['model_id']]['state_path']
                    row['rome_success'] = json.loads(state_path.read_text())['rome_success']
            summary['report_recovery'] = {'original_status': summary['status'], 'recovered_at': stamp(),
                                          'recovered_checkpoints': len(receipt['recovered']),
                                          'pbs_exit_status_unchanged': True}
            summary['status'] = ('needs_continuation' if summary.get('pending') else
                                 'complete_with_failures' if any(not c.get('evaluated') for c in summary['checkpoints']) else 'complete')
            write_json(budget_path, summary)
    receipt['finished_at'] = stamp()
    write_json(root / 'report-hotfix-recovery.json', receipt)
    return receipt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root'); parser.add_argument('--no-graphs', action='store_true')
    args = parser.parse_args(); result = recover(args.run_root, graphs=not args.no_graphs)
    print(json.dumps({'recovered': len(result['recovered']), 'errors': result['errors'],
                      'scientific_recomputation': False}))
    raise SystemExit(bool(result['errors']))
