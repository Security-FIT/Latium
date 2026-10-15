#!/usr/bin/env python3
"""Regenerate metrics and figures from original saved experiment data, on CPU."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from summarize_rome_gram import read, runs, observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO / 'analysis_out/saved-results')
    args = parser.parse_args()
    output = args.output.resolve()
    bundle = REPO / 'reproduction/gram-artifacts.zip'
    metadata = read(REPO / 'reproduction/bundle.json')
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != metadata['sha256']:
        raise ValueError('Bundle checksum mismatch')
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / 'inputs'
    inputs.mkdir(exist_ok=True)
    # Reject links/traversal, and never extract outside the dedicated input tree.
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            if member.is_dir() or not (inputs / member.filename).resolve().is_relative_to(inputs.resolve()):
                raise ValueError(f'Unsafe archive member: {member.filename}')
            dest = inputs / member.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(member))
    index = read(inputs / 'INDEX.json')
    for entry in index['files']:
        if hashlib.sha256((inputs / entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f"Input checksum mismatch: {entry['path']}")
    roots = [inputs / name for name in index['campaign_runs']]
    for name, extra in (('per-checkpoint.csv', []), ('per-family.csv', ['--by-family'])):
        with (output / name).open('w', newline='', encoding='utf-8') as stream:
            subprocess.run([sys.executable, str(REPO / 'scripts/summarize_rome_gram.py'),
                            *map(str, roots), *extra], stdout=stream, check=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from src.graphs.gram import case_rows, write_report

    grouped = defaultdict(list)
    totals = {}
    for root in roots:
        campaign_rows = []
        for manifest, family, model, revision in runs(root):
            executions, analyses = [], []
            for record in read(manifest)['artifacts'].values():
                if record['kind'] != 'execution' and not (
                    record['kind'] == 'analysis' and record['producer'] == 'gram-localization'
                ):
                    continue
                payload = read(manifest.parent / record['path'])
                (executions if record['kind'] == 'execution' else analyses).append(payload)
            rows, controls = case_rows(executions, analyses)
            # Also validates completeness, duplicates and all target-layer scores.
            checked = list(observations(manifest))
            assert len(rows) == len(checked)
            for row in rows:
                row['checkpoint'] = model
            campaign_rows.extend(rows)
            grouped[root.name, family].extend(rows)
            if root.name == 'classic-gram-merged':
                write_report(output / 'native-renderer' / family, rows, controls,
                             formats=('pdf', 'png'))
        totals[root.name] = dict(n=len(campaign_rows),
            rome_success=sum(r['rome_success'] for r in campaign_rows),
            gram_exact=sum(r['exact'] for r in campaign_rows))

    plt.rcParams.update({'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})

    def panel(axis, campaign, family):
        rows = grouped[campaign, family]
        layers = sorted(map(int, rows[0]['layer_scores']))
        assert all(sorted(map(int, r['layer_scores'])) == layers for r in rows)
        edited = np.array([[r['layer_scores'][str(l)] for l in layers] for r in rows])
        baseline = np.array([[r['baseline_layer_scores'][str(l)] for l in layers] for r in rows])
        axis.fill_between(layers, edited.min(0), edited.max(0), color='#f2b3ad', alpha=.45, label='Edited min–max')
        axis.fill_between(layers, np.quantile(edited, .25, axis=0), np.quantile(edited, .75, axis=0),
                          color='#c43d36', alpha=.15, label='Edited middle 50%')
        axis.plot(layers, edited.mean(0), color='#c43d36', label='Edited mean')
        axis.plot(layers, baseline.mean(0), color='black', label='Unedited' if len(rows) == 1000 else 'Unedited mean')
        if campaign != 'classic-gram-merged':
            axis.fill_between(layers, baseline.min(0), baseline.max(0), color='gray', alpha=.12, label='Unedited min–max')
        for target in sorted({int(r['target_layer']) for r in rows}):
            axis.axvline(target, color='gray', ls=':', lw=1, label=f'Edit layer {target}')
        exact = 100 * sum(r['exact'] for r in rows) / len(rows)
        axis.set(title=f'{family}\nexact {exact:.1f}% · n={len(rows)}', xlabel='Layer', ylabel='GRAM score')
        axis.set_ylim(bottom=0)
        axis.grid(axis='y', alpha=.2)

    figures = output / 'figures'
    for campaign, family in grouped:
        dest = figures / campaign
        dest.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        panel(ax, campaign, family)
        ax.legend(fontsize=7)
        fig.tight_layout()
        for fmt in ('pdf', 'png'):
            fig.savefig(dest / f'{family}.{fmt}', dpi=200)
        if campaign != 'classic-gram-merged':
            ax.set_yscale('log')
            for fmt in ('pdf', 'png'):
                fig.savefig(dest / f'{family}-log.{fmt}', dpi=200)
        plt.close(fig)

    # Four models used in the main paper figure; the archive also retains Gemma.
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, family in zip(axes, ('qwen3-8b', 'gpt2-medium', 'mistral-7b-v0.3', 'olmo-3-1025-7b')):
        panel(ax, 'classic-gram-merged', family)
    handles, labels = axes[0].get_legend_handles_labels()
    labels[-1] = 'Configured edit layer'
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .1, 1, 1))
    for fmt in ('pdf', 'png'):
        fig.savefig(figures / f'gram_profiles.{fmt}', dpi=200)
    plt.close(fig)

    for campaign in ('finetune-covariance', 'finetune-no-cov'):
        with (output / 'per-family.csv').open(newline='') as stream:
            stats = [r for r in csv.DictReader(stream) if r['campaign'] == campaign]
        x = np.arange(len(stats))
        fig, ax = plt.subplots(figsize=(12, 5))
        for offset, field, label in ((-.25, 'rome_pct', 'ROME success'), (0, 'gram_pct', 'GRAM exact'),
                                     (.25, 'baseline_fp_pct', 'Unedited FP')):
            ax.bar(x + offset, [float(r[field]) for r in stats], .25, label=label)
        ax.set_xticks(x, [r['family'] for r in stats], rotation=65, ha='right')
        ax.set(ylabel='Percent', ylim=(0, 105))
        ax.legend()
        fig.tight_layout()
        for fmt in ('pdf', 'png'):
            fig.savefig(figures / f'{campaign}-outcomes.{fmt}', dpi=200)
        plt.close(fig)
    (output / 'verification.json').write_text(json.dumps(dict(
        verified_input_files=len(index['files']), totals=totals), indent=2) + '\n')
    print(json.dumps(dict(output=str(output), totals=totals), indent=2))


if __name__ == '__main__':
    main()
