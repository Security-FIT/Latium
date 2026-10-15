#!/usr/bin/env python3
"""Build notebook inputs from the artifact ZIP and execute all paper notebooks."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO / 'analysis_out/paper')
    parser.add_argument('--saved-results', type=Path,
                        help='Reuse an existing render_saved_results.py output instead of rebuilding it')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    saved = args.saved_results.resolve() if args.saved_results else output / 'saved-results'
    if not args.saved_results:
        subprocess.run([sys.executable, str(REPO / 'scripts/render_saved_results.py'),
                        '--output', str(saved)], check=True)
    if not (saved / 'paper-inputs/data/classic-layer-profile-statistics.csv').exists():
        parser.error('Notebook inputs missing; run without --saved-results to build them')

    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    notebooks = REPO / 'notebooks/paper'
    os.environ['CSF_GRAPHS_DIR'] = str(saved / 'paper-inputs')
    os.environ['CSF_PAPER_OUTPUT'] = str(output / 'outputs')
    executed = output / 'notebooks'
    executed.mkdir(exist_ok=True)
    # Use this interpreter, rather than an unrelated user/global python3 kernel.
    with tempfile.TemporaryDirectory(prefix='latium-paper-kernel-') as temp:
        kernels = Path(temp)
        spec = kernels / 'latium-paper'
        spec.mkdir()
        (spec / 'kernel.json').write_text(json.dumps(dict(
            argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            display_name='Latium paper', language='python')))
        for source in sorted(notebooks.glob('0*.ipynb')):
            nb = nbformat.read(source, as_version=4)
            manager = KernelManager(kernel_name='latium-paper',
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
            client = NotebookClient(nb, km=manager, timeout=600, allow_errors=False,
                resources={'metadata': {'path': str(notebooks)}})
            client.execute(cleanup_kc=True)
            nbformat.write(nb, executed / source.name)
            html, _ = HTMLExporter().from_notebook_node(nb)
            (executed / source.with_suffix('.html').name).write_text(html, encoding='utf-8')
            print(f'Passed: {source.name}', flush=True)

    checks = []
    for path in sorted((output / 'outputs').glob('checks_*.csv')):
        with path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert rows and all(r['ok'] == 'True' for r in rows), path
        checks.append(dict(file=path.name, checks=len(rows), passed=len(rows)))
    assert len(checks) == 4, checks
    result = dict(passed=True, notebooks=4, checks=checks, total_checks=sum(c['checks'] for c in checks),
                  input_directory=str(saved / 'paper-inputs'), output_directory=str(output))
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
