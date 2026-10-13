"""Report recovery must never promote incomplete or changed science artifacts."""
import json

import pytest

from jobs import recover_report_failure as recovery
from jobs.finetuned_trace import trace_implementation_hash
from src.gram_experiment import register_batch
from test_gram_workflow import cohort_file, write_batch


def failed_report(tmp_path):
    _, cohort, _ = cohort_file(tmp_path, n=10)
    root = tmp_path / 'run'
    key, position = 'fleet_owner_model', 3
    entry = {'key': key, 'model_id': 'owner/model', 'revision': 'a' * 40,
             'fact_position': position, 'fact_case_id': cohort['case_ids'][position]}
    register_batch(root, cohort, {key: {'config_hash': 'frozen'}}, position, position + 1, {'workflow': 'gram'})
    model_run = root / 'models' / key / 'run'
    write_batch(model_run, key, cohort, position, position + 1)
    (root / 'checkpoints.json').write_text(json.dumps({
        'identity': {'covariance_source': 'finetuned', 'covariance_samples': 123}, 'models': [entry]}))
    inner = root / 'models' / key / 'batches' / 'm0003-0004' / 'state.json'
    outer = root / 'models' / key / 'fleet-batches' / 'm0000-0001' / 'state.json'
    inner.parent.mkdir(parents=True, exist_ok=True)
    outer.parent.mkdir(parents=True)
    inner.write_text(json.dumps({'status': 'failed', 'error': 'RuntimeError: gram-report failed (1)'}))
    state = {**entry, 'status': 'failed',
             'error': 'RuntimeError: Gram batch failed; see batches/*/state.json and fleet.log',
             'trace_implementation': trace_implementation_hash(),
             'trace': {'status': 'rejected'},
             'covariance': {'removed': True, 'target_samples': 123,
                            'files': [{'bytes': 12, 'sha256': 'b' * 64}],
                            'computation': {'source': 'computed', 'target_samples': 123,
                                            'checkpoint_revision': entry['revision']}}}
    outer.write_text(json.dumps(state))
    (root / 'budget-summary.json').write_text(json.dumps({
        'status': 'complete_with_failures', 'pending': [], 'checkpoints': [
            {'model_id': entry['model_id'], 'evaluated': False, 'returncode': 1, 'state_status': 'failed'}]}))
    return root, model_run, inner, outer


def test_recovery_uses_verified_artifacts_without_model_loading(tmp_path, monkeypatch):
    root, _, inner, outer = failed_report(tmp_path)
    from jobs import paper_fleet
    monkeypatch.setattr(paper_fleet, 'run_logged', lambda *a, **kw: pytest.fail('Scientific stages must not run'))
    result = recovery.recover(root, graphs=False)
    assert len(result['recovered']) == 1 and not result['errors']
    assert json.loads(inner.read_text())['status'] == 'complete'
    assert json.loads(outer.read_text())['status'] == 'complete'
    assert json.loads(outer.with_name(outer.name + '.before-report-hotfix').read_text())['status'] == 'failed'
    summary = json.loads((root / 'budget-summary.json').read_text())
    assert summary['status'] == 'complete' and summary['checkpoints'][0]['evaluated']
    assert summary['checkpoints'][0]['returncode'] == 1
    assert summary['checkpoints'][0]['report_recovery']['original_returncode'] == 1
    assert not recovery.recover(root, graphs=False)['recovered']


@pytest.mark.parametrize('damage', ['missing', 'changed', 'nonfinite', 'case_error', 'revision', 'science_failure'])
def test_recovery_leaves_dirty_runs_failed(tmp_path, damage):
    root, model_run, inner, outer = failed_report(tmp_path)
    if damage in ('missing', 'changed', 'nonfinite', 'case_error'):
        from src.results import content_hash
        manifest_path = model_run / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        record = next(r for r in manifest['artifacts'].values() if r['kind'] == 'capture' and r.get('edit_method') == 'rome')
        artifact = model_run / record['path']
        if damage == 'missing':
            artifact.unlink()
        else:
            payload = json.loads(artifact.read_text())
            if damage == 'case_error':
                payload['cases'][0]['status'] = 'error'
            else:
                payload['cases'][0]['data']['value'] = float('nan') if damage == 'nonfinite' else 9
            artifact.write_text(json.dumps(payload))
            if damage != 'changed':
                record['content_hash'] = content_hash(payload)
                manifest_path.write_text(json.dumps(manifest))
    elif damage == 'revision':
        state = json.loads(outer.read_text())
        state['revision'] = 'c' * 40
        outer.write_text(json.dumps(state))
    else:
        inner.write_text(json.dumps({'status': 'failed', 'error': 'RuntimeError: gram failed (1)'}))
    originals = (inner.read_bytes(), outer.read_bytes())
    result = recovery.recover(root, graphs=False)
    assert result['errors'] and not result['recovered']
    assert (inner.read_bytes(), outer.read_bytes()) == originals


def test_failed_graph_rerender_does_not_promote_states(tmp_path, monkeypatch):
    root, _, inner, outer = failed_report(tmp_path)
    from jobs import paper_fleet
    def failed_render(*a, **kw):
        raise RuntimeError('renderer still failed')
    monkeypatch.setattr(paper_fleet, 'run_logged', failed_render)
    result = recovery.recover(root)
    assert result['errors'] and not result['recovered']
    assert json.loads(inner.read_text())['status'] == 'failed'
    assert json.loads(outer.read_text())['status'] == 'failed'
