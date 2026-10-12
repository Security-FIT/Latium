#!/usr/bin/env python3
"""Run frozen checkpoints with an absolute PBS budget and checkpoint cleanup."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK_SECONDS = 19.5 * 3600


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def stop_group(process):
    """Stop the worker and all its model/trace subprocesses before cleanup."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # The parent can exit before a GPU child. Kill the entire original group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=15)
    # SIGKILL delivery is asynchronous; do not remove files while children exit.
    deadline = time.monotonic() + 5
    while True:
        live = False
        for stat in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = stat.read_text().rsplit(') ', 1)[1].split()
                if int(fields[2]) == process.pid and fields[0] not in ('Z', 'X'):
                    live = True
                    break
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        if not live:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Worker process group {process.pid} did not stop')
        time.sleep(0.05)


def run_process(command, timeout_seconds):
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        return process.wait(timeout=timeout_seconds), False
    except subprocess.TimeoutExpired:
        stop_group(process)
        return 124, True
    except BaseException:
        stop_group(process)
        raise


def cleanup_owned(run_root, download_parent, samples=None):
    """Reuse the tested provenance cleanup; verify ownership before deletion."""
    from jobs import finetuned_gram_fleet as fleet
    root = Path(run_root).resolve()
    selection_path = root / 'checkpoints.json'
    if not selection_path.is_file():
        return
    selection = json.loads(selection_path.read_text())
    samples = samples or selection['identity'].get('covariance_samples', 100000)
    if selection['identity']['covariance_source'] != 'finetuned':
        raise ValueError('Budget runner only cleans dedicated finetuned covariance')
    if selection['identity'].get('download_layout') == 'shared-revision-v1':
        raise ValueError('Budget runner must not clean retained shared downloads')
    parent = Path(download_parent).resolve()
    downloads = Path(selection['identity']['download_root'])
    if downloads.is_symlink() or not downloads.resolve().is_relative_to(parent):
        raise ValueError('Download path escaped the dedicated job directory')
    if downloads.exists():
        marker = downloads / 'owner.json'
        if not marker.is_file() or json.loads(marker.read_text()) != {'run_root': str(root)}:
            raise ValueError('Cannot clean an unowned checkpoint directory')
    for entry in selection['models']:
        record = fleet.remove_checkpoint_covariance(root, entry, samples)
        state_path = root / 'models' / entry['key'] / 'fleet-batches' / 'm0000-0001' / 'state.json'
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            state['covariance'] = record
            if state.get('status') == 'running':
                state.update(status='failed', error='Worker interrupted by job budget or termination')
            write_json(state_path, state)
        if downloads.exists():
            fleet.remove_download(downloads / entry['key'], downloads)
    if downloads.exists():
        fleet.remove_download(downloads / 'bases', downloads)


def make_command(batch, run_root, download_root, artifact_root, *, inputs_root=None, samples=100000):
    inputs = Path(inputs_root) if inputs_root is not None else ROOT / 'inputs'
    manifest = (Path(batch['models_manifest']) if batch.get('models_manifest') else
                inputs / 'batches' / f"{batch['array_index']:03d}.json")
    command = [sys.executable, str(ROOT / 'jobs/finetuned_gram_fleet.py'),
               '--base-model', batch['family'], '--models-manifest', str(manifest),
               '--model-count', str(len(batch['models'])), '--run-root', str(run_root),
               '--artifact-root', artifact_root, '--download-root', str(download_root),
               '--case-index-file', str(batch.get('case_index_file', inputs / 'rome-facts.json')),
               '--trace-case-index-file', str(batch.get('trace_case_index_file', inputs / 'trace-facts.json')),
               '--case-start', '0', '--case-stop', '1', '--causal-kuba-fix',
               '--tracking', 'none', '--no-keep-downloads', '--finetuned-covariance',
               '--no-keep-covariance', '--covariance-samples', str(samples)]
    if batch.get('prefix_cache'):
        command += ['--prefix-cache-file', str(ROOT / batch['prefix_cache'])]
    return command


def should_start(remaining, predicted, observed):
    required = max(predicted, observed * 1.5) + 120
    return remaining >= required


def terminate(signum, frame):
    raise InterruptedError(f'Received signal {signum}')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--array-index', type=int)
    parser.add_argument('--plan', help='Optional frozen array plan; inputs default to its directory')
    parser.add_argument('--models-manifest', help='Assigned checkpoint manifest for a single job')
    parser.add_argument('--base-model')
    parser.add_argument('--case-index-file')
    parser.add_argument('--trace-case-index-file')
    parser.add_argument('--prefix-cache-file')
    parser.add_argument('--work-hours', type=float, default=WORK_SECONDS / 3600,
                        help='Absolute worker cutoff including runtime staging; leave room for artifact transfer')
    parser.add_argument('--per-checkpoint-budget-minutes', type=float, default=60)
    parser.add_argument('--covariance-samples', type=int, default=100000)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--download-root', required=True)
    parser.add_argument('--artifact-root')
    parser.add_argument('--job-start-epoch', type=float)
    parser.add_argument('--cleanup-only', action='store_true')
    args = parser.parse_args(argv)
    root = Path(args.run_root).resolve()
    if args.cleanup_only:
        cleanup_owned(root, args.download_root)
        from jobs.recover_report_failure import recover
        result = recover(root)
        print(json.dumps({'report_hotfix_recovered': len(result['recovered']), 'report_hotfix_errors': result['errors']}), flush=True)
        return int(bool(result['errors']))
    if args.job_start_epoch is None or not args.artifact_root:
        parser.error('job-start-epoch and artifact-root are required')
    if (not all(math.isfinite(v) and v > 0 for v in
                (args.work_hours, args.per_checkpoint_budget_minutes, args.covariance_samples))
            or not math.isfinite(args.job_start_epoch)):
        parser.error('work-hours, checkpoint budget and covariance-samples must be positive')
    if args.plan:
        if args.models_manifest or args.array_index is None:
            parser.error('plan requires array-index and cannot be combined with models-manifest')
        plan_path = Path(args.plan).resolve()
        plan = json.loads(plan_path.read_text())
        matches = [b for b in plan['batches'] if b['array_index'] == args.array_index]
        if len(matches) != 1:
            parser.error('array-index must identify exactly one frozen batch')
        batch, inputs_root = matches[0], plan_path.parent
    else:
        if not all((args.models_manifest, args.base_model, args.case_index_file, args.trace_case_index_file)):
            parser.error('a single job requires models-manifest, base-model and both fact manifests')
        manifest = Path(args.models_manifest).resolve()
        frozen = json.loads(manifest.read_text())
        batch = {'family': args.base_model, 'array_index': args.array_index,
                 'models_manifest': str(manifest), 'models': frozen['models'],
                 'case_index_file': str(Path(args.case_index_file).resolve()),
                 'trace_case_index_file': str(Path(args.trace_case_index_file).resolve()),
                 'per_checkpoint_budget_minutes': args.per_checkpoint_budget_minutes}
        if args.prefix_cache_file:
            batch['prefix_cache'] = str(Path(args.prefix_cache_file).resolve())
        inputs_root = manifest.parent
    if not batch['models'] or any('fact_position' not in e or 'fact_case_id' not in e for e in batch['models']):
        parser.error('budgeted jobs require nonempty frozen checkpoints with fixed fact assignments')
    deadline = args.job_start_epoch + args.work_hours * 3600
    base_command = make_command(batch, root, args.download_root, args.artifact_root,
                                inputs_root=inputs_root, samples=args.covariance_samples)
    summary = {'job_id': os.environ.get('PBS_JOBID'), 'array_index': args.array_index,
               'family': batch['family'], 'pbs_array_index': os.environ.get('PBS_ARRAY_INDEX'), 'job_start_epoch': args.job_start_epoch,
               'worker_deadline_epoch': deadline, 'work_cutoff_hours': args.work_hours,
               'checkpoints': [], 'pending': [], 'status': 'running'}
    write_json(root / 'budget-summary.json', summary)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    observed = 0
    attempted = set()
    failures = False
    interrupted = False
    try:
        prepare_time = min(600, max(1, deadline - time.time()))
        code, expired = run_process(base_command + ['--prepare-only'], prepare_time)
        if code:
            raise RuntimeError(f'Frozen checkpoint preparation failed: exit {code}, timeout={expired}')
        selection = json.loads((root / 'checkpoints.json').read_text())
        # Unsupported frozen members are recorded as failed by the worker;
        # retain their ranks and continue with other checkpoints in the job.
        for position, entry in enumerate(selection['models']):
            predicted = batch['per_checkpoint_budget_minutes'] * 60
            remaining = deadline - time.time()
            if not should_start(remaining, predicted, observed):
                print(f"Budget guard: deferring checkpoint {position}; remaining={remaining:.0f}s", flush=True)
                break
            started = time.time()
            # Allow unusually slow checkpoints, but never cross the absolute job cutoff.
            allowance = min(remaining - 120, max(7200, predicted * 2))
            command = base_command + ['--checkpoint-start', str(position), '--checkpoint-stop', str(position + 1)]
            attempted.add(position)
            print(f"Checkpoint {position + 1}/{len(selection['models'])}: {entry['model_id']}; timeout={allowance:.0f}s", flush=True)
            code, expired = run_process(command, allowance)
            cleanup_owned(root, args.download_root)
            state_path = root / 'models' / entry['key'] / 'fleet-batches' / 'm0000-0001' / 'state.json'
            state = json.loads(state_path.read_text()) if state_path.is_file() else {}
            elapsed = time.time() - started
            valid = code == 0 and state.get('status') in ('complete', 'rome_failed')
            failures |= not valid
            if valid:
                observed = max(observed, elapsed)
            summary['checkpoints'].append({'position': position, 'model_id': entry['model_id'],
                'revision': entry['revision'], 'fact_position': entry['fact_position'],
                'fact_case_id': entry['fact_case_id'], 'elapsed_seconds': elapsed,
                'returncode': code, 'timed_out': expired, 'state_status': state.get('status'),
                'evaluated': valid, 'rome_success': state.get('rome_success')})
            write_json(root / 'budget-summary.json', summary)
            subprocess.run(['rsync', '-a', '--timeout=60', str(root) + '/', args.artifact_root.rstrip('/') + '/'],
                           check=True, timeout=120)
    except Exception as exc:
        failures = True
        interrupted = True
        summary['error'] = f'{type(exc).__name__}: {exc}'
        print(summary['error'], flush=True)
    finally:
        cleanup_owned(root, args.download_root)
        reported = {row['position'] for row in summary['checkpoints']}
        for position in sorted(attempted - reported):
            summary['checkpoints'].append({'position': position, **batch['models'][position],
                'evaluated': False, 'state_status': 'interrupted', 'error': summary.get('error')})
        summary['pending'] = [{'position': p, **e} for p, e in enumerate(batch['models']) if p not in attempted]
        summary['status'] = 'needs_continuation' if summary['pending'] else 'complete_with_failures' if failures else 'complete'
        summary['finished_epoch'] = time.time()
        summary['elapsed_hours'] = (time.time() - args.job_start_epoch) / 3600
        write_json(root / 'budget-summary.json', summary)
    return 75 if summary['pending'] else 1 if failures or interrupted else 0


if __name__ == '__main__':
    raise SystemExit(main())
