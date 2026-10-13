import json
from pathlib import Path
import sys
import time
import pytest
from jobs import run_budgeted_finetunes as budget


def test_guard_requires_room_for_checkpoint_and_transfer():
    assert budget.should_start(3720, 3600, 0)
    assert not budget.should_start(3719, 3600, 0)
    assert not budget.should_start(5000, 3600, 4000)
    assert budget.should_start(6120, 3600, 4000)


def test_timeout_kills_model_process_group(tmp_path):
    pidfile=tmp_path/'child.pid'
    code=('import os,subprocess,sys,time; '
          'p=subprocess.Popen([sys.executable,"-c","import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(120)"]); '
          'open(sys.argv[1],"w").write(str(p.pid)); time.sleep(120)')
    start=time.time()
    result,expired=budget.run_process([sys.executable,'-c',code,str(pidfile)],1)
    assert result==124 and expired and time.time()-start<20
    pid=int(pidfile.read_text())
    proc=Path(f'/proc/{pid}/stat')
    assert not proc.exists() or proc.read_text().split()[2]=='Z'


def fake_run(tmp_path, owner=True):
    root=tmp_path/'run';downloads=tmp_path/'downloads'/'owned';key='fleet_owner_model'
    root.mkdir();downloads.mkdir(parents=True)
    entry={'key':key,'model_id':'owner/model','revision':'a'*40,'fact_position':3,'fact_case_id':5226}
    (root/'checkpoints.json').write_text(json.dumps({'identity':{'covariance_source':'finetuned','download_root':str(downloads)},'models':[entry]}))
    (downloads/'owner.json').write_text(json.dumps({'run_root':str(root if owner else tmp_path/'other')}))
    weights=downloads/key;weights.mkdir();(weights/'model.safetensors').write_bytes(b'owned weights')
    cov=root/'models'/key/'covariance';cov.mkdir(parents=True);(cov/'own.pt').write_bytes(b'covariance')
    state=root/'models'/key/'fleet-batches'/'m0000-0001'/'state.json';state.parent.mkdir(parents=True);state.write_text(json.dumps({'status':'running'}))
    return root,downloads,key,state


def test_interruption_removes_covariance_and_owned_weights_with_provenance(tmp_path):
    root,downloads,key,state=fake_run(tmp_path)
    budget.cleanup_owned(root,downloads.parent)
    assert not (downloads/key).exists()
    assert not (root/'models'/key/'covariance').exists()
    record=json.loads((root/'models'/key/'covariance-lifecycle.json').read_text())
    assert record['removed'] and record['files'][0]['sha256'] and record['checkpoint_revision']=='a'*40
    assert json.loads(state.read_text())['status']=='failed'
    budget.cleanup_owned(root,downloads.parent)
    assert json.loads((root/'models'/key/'covariance-lifecycle.json').read_text())['files']==record['files']


def test_cleanup_rejects_foreign_owner_without_deleting_anything(tmp_path):
    root,downloads,key,state=fake_run(tmp_path,owner=False)
    with pytest.raises(ValueError,match='unowned'):
        budget.cleanup_owned(root,downloads.parent)
    assert (downloads/key/'model.safetensors').exists()
    assert (root/'models'/key/'covariance'/'own.pt').exists()


def test_frozen_command_preserves_owncov_and_separate_fact_manifests(tmp_path):
    batch={'family':'qwen3-8b','array_index':1,'models':[{}],'prefix_cache':'inputs/prefix_cache/qwen3.json'}
    command=budget.make_command(batch,tmp_path/'run',tmp_path/'downloads','server:results/')
    for argument in ['--finetuned-covariance','--no-keep-covariance','--no-keep-downloads','--causal-kuba-fix']:
        assert argument in command
    assert command[command.index('--trace-case-index-file')+1].endswith('trace-facts.json')
    assert command[command.index('--case-index-file')+1].endswith('rome-facts.json')
    assert command[command.index('--models-manifest')+1].endswith('001.json')


def test_direct_manifest_command_accepts_custom_paths_and_sample_count(tmp_path):
    batch = {'family': 'gpt2-xl', 'models': [{}], 'models_manifest': str(tmp_path / 'frozen.json'),
             'case_index_file': str(tmp_path / 'rome.json'),
             'trace_case_index_file': str(tmp_path / 'trace.json')}
    command = budget.make_command(batch, tmp_path / 'run', tmp_path / 'downloads', 'server:results/', samples=321)
    assert command[command.index('--models-manifest') + 1] == str(tmp_path / 'frozen.json')
    assert command[command.index('--covariance-samples') + 1] == '321'


@pytest.mark.parametrize('work_hours,worker_status,expected_status,expected_code', [
    ('1', 'rome_failed', 'needs_continuation', 75),
    ('2', 'rome_failed', 'complete', 0),
    ('2', 'failed', 'complete_with_failures', 1)])
def test_direct_budget_keeps_assignments_and_defers_without_backfilling(tmp_path, monkeypatch,
                                                                      work_hours, worker_status, expected_status, expected_code):
    root = tmp_path / 'run'
    manifest = tmp_path / 'frozen.json'
    entry = {'key': 'fleet_owner_model', 'model_id': 'owner/model', 'revision': 'a' * 40,
             'fact_position': 3, 'fact_case_id': 5226}
    if worker_status == 'failed':
        entry['selection_error'] = 'Unsupported frozen checkpoint'
    manifest.write_text(json.dumps({'models': [entry]}))
    calls, cleanup = [], []
    monkeypatch.setattr(budget.time, 'time', lambda: 1000)
    monkeypatch.setattr(budget.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(budget, 'cleanup_owned', lambda *a: cleanup.append(a))
    monkeypatch.setattr(budget.subprocess, 'run', lambda *a, **kw: None)
    def fake_process(command, timeout_seconds):
        calls.append(command)
        if '--prepare-only' in command:
            budget.write_json(root / 'checkpoints.json', {'models': [entry]})
        else:
            budget.write_json(root / 'models' / entry['key'] / 'fleet-batches' / 'm0000-0001' / 'state.json',
                              {'status': worker_status, 'rome_success': False})
        return int('--prepare-only' not in command and worker_status == 'failed'), False
    monkeypatch.setattr(budget, 'run_process', fake_process)
    code = budget.main(['--models-manifest', str(manifest), '--base-model', 'gpt2-xl',
                        '--case-index-file', str(tmp_path / 'rome.json'),
                        '--trace-case-index-file', str(tmp_path / 'trace.json'),
                        '--run-root', str(root), '--download-root', str(tmp_path / 'downloads'),
                        '--artifact-root', str(tmp_path / 'results'), '--job-start-epoch', '1000',
                        '--work-hours', work_hours])
    summary = json.loads((root / 'budget-summary.json').read_text())
    assert code == expected_code and summary['status'] == expected_status
    assert cleanup
    if expected_code == 75:
        assert len(calls) == 1 and summary['pending'][0]['fact_case_id'] == 5226
        assert not summary['checkpoints']
    else:
        assert len(calls) == 2 and not summary['pending']
        assert summary['checkpoints'][0]['evaluated'] == (worker_status != 'failed')
        assert summary['checkpoints'][0]['rome_success'] is False
        assert summary['checkpoints'][0]['fact_case_id'] == 5226
