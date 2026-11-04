"""Check the added attention dimensions and RMSNorm widths through the model bench.

    python coverage/bench_shapes.py --model-root NPU_MODEL --atlas-mlir-bin BIN --out OUT

Each case executes both functional RTL numerics and the Core cycle model, checks compiler
words at stream-verified or better, and compares with the family's FP32 reference.
Neither family has an independent Atlas-arithmetic golden. The acceptance limits are
cosine >= 0.98 and relative L2 error <= 0.20 on seeded, normally distributed inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bench_model


def inventory():
    rows = []
    for dim in (128, 256):
        for mask in ('none', 'causal'):
            for fp8 in (False, True):
                for queries, keys in ((32, 64), (1, 65)):
                    rows.append(dict(
                        phase='validation', module=f'd{dim}_{mask}_{fp8}_{queries}', source='shape-sweep',
                        op='aten.scaled_dot_product_attention.default',
                        inputs=[[[1, 2, queries, dim], 'bfloat16'], [[1, 1, keys, dim], 'bfloat16'],
                                [[1, 1, keys, dim], 'bfloat16']],
                        outputs=[[[1, 2, queries, dim], 'bfloat16']], count=1,
                        flops_each=8 * queries * keys * dim, where='atlas:attention', rule=0,
                        launch={'family': 'attention', 'per': {'op': 'aten.scaled_dot_product_attention.default'},
                                'mask': mask, 'output_fp8': fp8}))
    for width in (32, 960, 1024, 2304, 4096, 7136):
        for physical_rows in (32, 128):
            rows.append(dict(
                phase='validation', module=f'norm_{physical_rows}_{width}', source='shape-sweep',
                op='aten.rms_norm.default',
                inputs=[[[1, physical_rows, width], 'bfloat16'], [[width], 'bfloat16']],
                outputs=[[[1, physical_rows, width], 'bfloat16']], count=1, flops_each=0,
                where='atlas:rmsnorm', rule=1,
                launch={'family': 'rmsnorm', 'per': {'op': 'aten.rms_norm.default'}}))
    return dict(schema='atlas.coverage.summary.v1', status='COMPLETE', models={
        'shape_sweep': dict(status='COMPLETE', kit_launches={}, bench={}, atlas_rows=rows, unmapped=[])})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--atlas-mlir-bin', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    coverage = args.out / 'coverage.json'
    coverage.write_text(json.dumps(inventory(), indent=1) + '\n')
    command = ['--coverage', str(coverage), '--model-root', str(args.model_root), '--out', str(args.out),
               '--workers', str(args.workers), '--atlas-mlir-bin', str(args.atlas_mlir_bin),
               '--min-tier', 'stream-verified',
               '--command', 'python coverage/bench_shapes.py --model-root NPU_MODEL --atlas-mlir-bin BIN --out OUT',
               '--note', 'New-shape acceptance: cosine >= 0.98, relative L2 error <= 0.20; both families lack an '
                         'independent Atlas-arithmetic golden. All cycle counts are full launches.']
    if args.resume:
        command.append('--resume')
    if args.markdown:
        command.extend(['--markdown', str(args.markdown)])
    status = bench_model.main(command)
    document = json.loads((args.out / 'MODEL-BENCH.json').read_text())
    assert len(document['launches']) == 28
    for row in document['launches']:
        assert row['status'] == 'ok', row
        assert row['timed_output_equal'], row
        assert row['backend']['tier'] in ('verified', 'stream-verified'), row
        assert row['cosine'] >= 0.98 and row['relative_error'] <= 0.20, row
    return status


if __name__ == '__main__':
    raise SystemExit(main())
