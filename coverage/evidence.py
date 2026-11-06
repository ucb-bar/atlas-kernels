"""Recover retained model-bench artifacts from immutable, hash-checked Git objects.

No model execution or network access is performed. A normal clone contains the
historical commit; shallow clones must fetch that commit before recovery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('tinyllama', 'smolvla')
VARIANTS = ('smol-projection-results', 'tiny-fused-results')
DOCUMENT_FIELDS = ('model', 'coverage', 'coverage_sha256', 'coverage_status', 'workers',
                   'launch_checks', 'unmapped', 'uncovered', 'npu_model')
ROW_FIELDS = ('key', 'family', 'params', 'count', 'flops', 'modules', 'dtypes', 'phases',
              'shape', 'variant', 'generator_words', 'program_words', 'instructions',
              'functional_seconds', 'timed_seconds', 'subset_seconds', 'golden',
              'bit_exact', 'full_extent_exact', 'padding_differences',
              'padding_signed_zero_only', 'cosine', 'relative_error', 'max_abs_error',
              'model_cycles', 'cycles_source', 'timed_output_equal', 'status', 'total_cycles')


def summarize(document):
    """Keep full precision for every reported measurement, without run diagnostics."""
    summary = {'schema': 'atlas_model_bench_summary_v1'}
    summary.update((key, document[key]) for key in DOCUMENT_FIELDS)
    summary['command'] = ('make bench-model MODEL=' + document['model']
                          + ' OUT=out/bench/' + document['model']
                          + ' NPU_MODEL=... MODEL_PYTHON=... ATLAS_MLIR_BIN=... WORKERS=4')
    summary['backend'] = {'min_tier': document['backend']['min_tier']}
    summary['notes'] = [
        'Retained full-shape measurements at atlas-kernels '
        '`80c5f27d4bcded9c1ca7618a17ffcdbae2be7a39` and atlas-mlir '
        '`b8dca4e65459d24cb74a403569b46e8325c4c489`. '
        'Each launch kind has fresh functional and timed execution; no resume or '
        'row-subset cycle estimate enters the totals. These are historical results, '
        'not measurements of subsequent source changes.',
        'The [artifact manifest](MODEL-BENCH.artifact.json) authenticates the full JSON '
        'and this [compact summary](MODEL-BENCH.summary.json). '
        'Run `make evidence` to recover the full JSON and regenerate reports under '
        '`out/evidence/` without simulations. See the '
        '[evidence instructions](../../EVIDENCE.md) for recovery and fresh measurement.',
    ]
    if document['model'] == 'smolvla':
        summary['notes'].append(
            'The original smol_fusion workers failed before numerical execution due '
            'to an unrelated tools snapshot on PYTHONPATH. Only that family was '
            'rerun in a fresh directory with PYTHONPATH cleared. Original and retry '
            'commands, environment overrides, and source hashes remain in the full JSON.')
    summary['launches'] = []
    for row in document['launches']:
        compact = {key: row[key] for key in ROW_FIELDS if key in row}
        if row.get('backend'):
            compact['backend'] = {'tier': row['backend']['tier']}
        summary['launches'].append(compact)
    return summary


def verify(data, expected, label):
    if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
        raise ValueError(f'{label}: byte length or SHA-256 mismatch')


def artifact_bytes(manifest, repo, label):
    source = manifest['git']
    try:
        blob = subprocess.run(
            ['git', '-C', str(repo), 'rev-parse', f"{source['commit']}:{source['path']}"],
            check=True, capture_output=True, text=True).stdout.strip()
        if blob != source['blob']:
            raise ValueError(f'{label}: Git blob identity mismatch')
        data = subprocess.run(['git', '-C', str(repo), 'cat-file', 'blob', blob],
                              check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError(f"{label}: historical Git objects unavailable; fetch commit "
                         f"{source['commit']} from your atlas-kernels remote") from error
    verify(data, manifest['artifact'], label)
    return data


def write_artifact(target, data):
    # Never silently overwrite a locally modified evidence artifact.
    if target.exists() and target.read_bytes() != data:
        raise FileExistsError(f'{target}: existing artifact differs')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def summarize_variant(document):
    return {key: value for key, value in document.items() if key not in
            ('compiler_tools', 'traces', 'rtl_receipt', 'compiler_functional_joins', 'rtl')}


def recover_variant(name, out, repo=ROOT):
    manifest = json.loads((ROOT / 'coverage' / f'{name}.json').read_text())
    data = artifact_bytes(manifest, repo, name)
    if summarize_variant(json.loads(data)) != manifest['summary']:
        raise ValueError(f'{name}: compact measurements differ from full artifact')
    target = out / 'variants' / f'{name}.json'
    write_artifact(target, data)
    return target


def recover(model, out, repo=ROOT):
    directory = ROOT / 'coverage' / 'examples' / model
    manifest = json.loads((directory / 'MODEL-BENCH.artifact.json').read_text())
    data = artifact_bytes(manifest, repo, model)
    summary_data = (directory / 'MODEL-BENCH.summary.json').read_bytes()
    verify(summary_data, manifest['summary'], f'{model} summary')
    summary = json.loads(summary_data)
    if summarize(json.loads(data)) != summary:
        raise ValueError(f'{model}: compact measurements differ from full artifact')
    destination = out / model
    target = destination / 'MODEL-BENCH.json'
    write_artifact(target, data)
    for name in ('MODEL-BENCH.artifact.json', 'MODEL-BENCH.summary.json'):
        (destination / name).write_bytes((directory / name).read_bytes())
    return summary, destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'out' / 'evidence')
    args = parser.parse_args(argv)
    from bench_model import render
    for model in MODELS:
        summary, destination = recover(model, args.out)
        instructions = os.path.relpath(ROOT / 'coverage' / 'EVIDENCE.md', destination)
        report = render(summary).replace('(../../EVIDENCE.md)', f'({instructions})')
        (destination / 'MODEL-BENCH.md').write_text(report)
        print(f'{model}: authenticated JSON and report in {destination}')
    for name in VARIANTS:
        print(f'{name}: authenticated JSON in {recover_variant(name, args.out)}')


if __name__ == '__main__':
    main()
