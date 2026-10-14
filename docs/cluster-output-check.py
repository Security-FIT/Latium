import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image
from pypdf import PdfReader
import torch
from src.submission.artifacts import read_run

results = Path('results')
records = []
primary_count = 0
for path in sorted(results.glob('smoke-*/runs/*/*/reproduction.json')):
    r = json.loads(path.read_text())
    assert r['status'] == 'complete' and r['validation_subset']
    root = Path(r['run_root'])
    _, artifacts = read_run(root)
    primary_count += len(artifacts)
    assert len(artifacts) == 6
    assert all(p['status'] == 'complete' and all(c['status'] == 'complete' for c in p['cases'])
               for p in artifacts.values())
    config = r['resolved_model_config']
    cov = Path(config['second_moment_path'])
    assert hashlib.sha256(cov.read_bytes()).hexdigest() == r['covariance']['sha256']
    matrix = torch.load(cov, map_location='cpu', weights_only=True)
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    assert matrix.dtype == torch.float32 and torch.isfinite(matrix).all()
    identity = r['covariance']['identity']
    assert identity['samples'] == r['covariance']['count'] == 32
    if r['campaign'] == 'finetune-own':
        assert identity['model'] == r['entry']['model_id']
        assert identity['revision'] == r['entry']['revision']
        assert r['covariance']['fresh_computation']
    rows = json.loads((root.parent / 'rendered/cases.json').read_text())
    assert len(rows) == 1
    row = rows[0]
    assert row['target_layer'] == config['layer']
    assert row['exact'] == (row['predicted_layer'] == row['target_layer'])
    assert row['baseline_fp'] == (row['baseline_selected_layer'] == row['target_layer'])
    assert row['rome_success'] == (row['metrics']['efficacy_score'] >= 1.)
    if r['campaign'] != 'classic':
        assert row['case_id'] == str(r['entry']['fact_case_id'])
        assert row['cohort_position'] == r['entry']['fact_position']
    records.append({'campaign': r['campaign'], 'family': r['entry']['family'], 'model_id': r['entry']['model_id'],
        'checkpoint_type': r['entry'].get('checkpoint_type', 'base'), 'checkpoint_revision': config['checkpoint_revision'],
        'adapter_base_revision': config.get('adapter_base_revision'), 'dtype': config['dtype'], 'seed': r['seed'],
        'case_id': row['case_id'], 'cohort_position': row['cohort_position'], 'cohort_hash': row['cohort_hash'],
        'target_layer': row['target_layer'], 'gram_selected_layer': row['predicted_layer'],
        'gram_exact': row['exact'], 'baseline_selected_layer': row['baseline_selected_layer'], 'baseline_fp': row['baseline_fp'],
        'efficacy_score': row['metrics']['efficacy_score'], 'paraphrase_score': row['metrics']['paraphrase_score'],
        'neighborhood_score': row['metrics']['neighborhood_score'], 'rome_success': row['rome_success'],
        'covariance_identity': identity, 'covariance_sha256': r['covariance']['sha256'], 'covariance_shape': list(matrix.shape),
        'covariance_dtype': str(matrix.dtype), 'source': r['source'], 'run_root': str(root), 'status': r['status']})
    del matrix
assert len(records) == 8
for family in ['gpt2-medium', 'gpt2-large', 'qwen3.5-4b']:
    paired = [r for r in records if r['family'] == family and r['campaign'] != 'classic']
    assert len(paired) == 2
    assert len({r['covariance_sha256'] for r in paired}) == 2

figures = []
roots = [Path('/workspace/csf-rendered'), Path('/workspace/csf-render-one'), results / 'cluster-restoration-probe/rendered']
roots += [Path(r['run_root']).parent / 'rendered' for r in records]
for root in roots:
    for p in sorted(root.rglob('*')):
        if p.suffix == '.png':
            with Image.open(p) as img:
                img.load()
                assert img.width > 100 and img.height > 100
        elif p.suffix == '.svg':
            ET.parse(p)
        elif p.suffix == '.pdf':
            pdf = PdfReader(p, strict=True)
            assert len(pdf.pages) > 0
            for page in pdf.pages:
                assert page.mediabox.width > 100 and page.mediabox.height > 100
                assert page.extract_text()
        else:
            continue
        figures.append({'path': str(p), 'format': p.suffix[1:], 'bytes': p.stat().st_size,
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
assert len(figures) == 102
gpu_restore = json.loads((results / 'cluster-validation/gpu-restoration.json').read_text())
assert len(gpu_restore['restorations']) == 2 and gpu_restore['source_unchanged_by_rendering']
for campaign in ['base', 'own', 'classic']:
    payload = json.loads((results / f'cluster-validation/render-new-{campaign}.json').read_text())
    assert payload['validation_subset']
    assert payload['checkpoints'] == (1 if campaign == 'classic' else 3)
    rows = json.loads((Path('/workspace/csf-rendered') / campaign / 'native/cases.json').read_text())
    assert all(len(row['checkpoint_revision']) == 40 for row in rows)
    expected = {(r['model_id'], r['case_id']): r['checkpoint_revision'] for r in records}
    assert all(row['checkpoint_revision'] == expected[(row['model_id'], row['case_id'])] for row in rows)
    if campaign == 'classic':
        summary = json.loads(Path('/workspace/csf-rendered/classic/family-summary.json').read_text())
        assert summary[0]['baseline_checkpoints'] == 1
report = {'validation_subset': True, 'renderer_revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
          'environment': json.loads((results / 'cluster-validation/environment.json').read_text()),
          'smoke_workflows': len(records), 'primary_artifacts_verified': primary_count, 'covariance_samples': 32,
          'rome_successes': sum(r['rome_success'] for r in records), 'gram_exact': sum(r['gram_exact'] for r in records),
          'baseline_fp': sum(r['baseline_fp'] for r in records), 'figures_verified': len(figures),
          'records': records, 'gpu_restoration': gpu_restore, 'figures': figures,
          'limitations': ['Smoke covariance uses 32 texts, not the 100000-text paper protocol.',
                          'Qwen3.5 uses reference PyTorch linear-attention/convolution kernels; optional accelerated packages are absent.',
                          'Only these three model families were exercised on this fresh cluster. Historical inputs and saved campaign results remain unchanged.']}
(results / 'cluster-validation/smoke-results.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: report[k] for k in ['smoke_workflows','primary_artifacts_verified','rome_successes','gram_exact','baseline_fp','figures_verified']},indent=2))
