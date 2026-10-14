import hashlib
import json
from pathlib import Path
import torch
from src.structural.config import StructuralBenchmarkConfig
from src.structural.runner import run_structural_benchmark
from src.structural.execution import edit_execution
from src.submission.artifacts import read_run
from src.submission.run import model_config_directory
from src.submission.render import render_saved_run

source = Path('results/smoke-classic-medium/runs/gpt2-medium/first500')
receipt = json.loads((source / 'reproduction.json').read_text())
assert receipt['status'] == 'complete' and receipt['validation_subset']
checks = []
original_restore = edit_execution.restore

def checked_restore(handler, outcome):
    parameters = {name: handler._get_module(name).weight for name in outcome.restorations} if outcome else {}
    original_restore(handler, outcome)
    for name, parameter in parameters.items():
        restored = handler._get_module(name).weight
        assert restored is parameter
        assert not restored.requires_grad
        assert torch.equal(restored, outcome.restorations[name])
        raw = restored.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        checks.append({'module': name, 'sha256': hashlib.sha256(raw).hexdigest(),
                       'device': str(restored.device), 'parameter_preserved': True,
                       'weights_restored_exactly': True, 'requires_grad': restored.requires_grad})

edit_execution.restore = checked_restore
output = Path('results/cluster-restoration-probe')
with model_config_directory(source / 'config/model'):
    cfg = StructuralBenchmarkConfig(models=('gpt2-medium',), n_tests=2, start_idx=0,
        case_index_file=str(Path('manifests/CSF-final/classic-facts.json').resolve()),
        output_dir=output, run_id='run', seed=0, analysis_preset='gram-localization',
        render_graphs=False, fail_on_missing_second_moment=True, force=True)
    run_structural_benchmark(cfg)
root = output / 'run'
_, artifacts = read_run(root)
cases = [case for p in artifacts.values() if p['kind'] == 'execution' and p['run']['edit_method'] == 'rome'
         for case in p['cases']]
assert len(cases) == len(checks) == 2
assert all(case['status'] == 'complete' for case in cases)
assert len({check['sha256'] for check in checks}) == 1
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.json')}
render_saved_run(root, output / 'rendered')
assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.json')}
result = {'validation_subset': True, 'covariance_samples': 32, 'case_ids': [c['case_id'] for c in cases],
          'restorations': checks, 'source_unchanged_by_rendering': True,
          'rendered_cases': json.loads((output / 'rendered/cases.json').read_text())}
Path('results/cluster-validation/gpu-restoration.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'restored_edits': len(checks), 'source_unchanged_by_rendering': True}))
