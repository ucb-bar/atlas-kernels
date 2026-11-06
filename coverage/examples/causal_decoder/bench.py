r"""Export Merlin's small decoder, place it with this directory's starter mapping, and bench its Atlas launches.

MERLIN_ROOT=/path/to/merlin EXPORT_PYTHON=/path/to/export/bin/python MODEL_PYTHON=/path/to/model/bin/python \
  NPU_MODEL=/path/to/npu-model python3 -B coverage/examples/causal_decoder/bench.py \
    --atlas-mlir-bin /path/to/atlas-mlir/build/bin --min-tier stream-verified --out /path/to/output

This is the "Starting a new model" path in the top README, end to end: export_ops.py --loader (EXPORT_PYTHON needs
torch and NumPy), build_table.py --allow-unmapped with mapping.yaml here (the driver Python needs PyYAML), and
bench_model.py on npu_model 47b8599 (MODEL_PYTHON needs Python >=3.14, torch, NumPy, and npu_model).
Exports, tables, and job results stay under --out; only MODEL-BENCH.md is written next to this script.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).absolute().parent
COVERAGE = HERE.parents[1]
COMMAND = ('MERLIN_ROOT=/path/to/merlin EXPORT_PYTHON=/path/to/export/bin/python '
           'MODEL_PYTHON=/path/to/model/bin/python NPU_MODEL=/path/to/npu-model '
           'python3 -B coverage/examples/causal_decoder/bench.py '
           '--atlas-mlir-bin /path/to/atlas-mlir/build/bin --min-tier stream-verified --out "$OUT"')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--merlin-root', type=Path, default=os.environ.get('MERLIN_ROOT'))
    parser.add_argument('--export-python', default=os.environ.get('EXPORT_PYTHON'))
    parser.add_argument('--model-python', default=os.environ.get('MODEL_PYTHON'))
    parser.add_argument('--torch-python', default=os.environ.get('TORCH_PYTHON'),
                        help='compatibility fallback for unset export and model interpreters')
    parser.add_argument('--atlas-mlir-bin', type=Path, default=os.environ.get('ATLAS_MLIR_BIN'))
    parser.add_argument('--min-tier', choices=('verified', 'stream-verified'), default='stream-verified')
    parser.add_argument('--model-root', type=Path, default=os.environ.get('NPU_MODEL'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, default=HERE / 'MODEL-BENCH.md')
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    args.export_python = args.export_python or args.torch_python or sys.executable
    args.model_python = args.model_python or args.torch_python
    if args.merlin_root is None or args.model_root is None:
        parser.error('set MERLIN_ROOT and NPU_MODEL, or pass --merlin-root and --model-root')
    if not args.model_python:
        parser.error('set MODEL_PYTHON (Python >=3.14 with torch, NumPy, and npu_model), or pass --model-python')
    if args.atlas_mlir_bin is None:
        parser.error('set ATLAS_MLIR_BIN, or pass --atlas-mlir-bin')
    args.out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    fixture, table = args.out / 'causal_decoder.json', args.out / 'table'
    steps = [
        [args.export_python, '-B', str(COVERAGE / 'export_ops.py'), 'causal_decoder', '--loader',
         str(args.merlin_root / 'examples/workloads/causal_decoder/loader.py'), '--out', str(args.out / 'export'), '--fixture', str(fixture)],
        [sys.executable, '-B', str(COVERAGE / 'build_table.py'), '--inventory', f'causal_decoder={fixture}',
         '--mapping', str(HERE / 'mapping.yaml'), '--allow-unmapped', '--out', str(table)],
        [args.model_python, '-B', str(COVERAGE / 'bench_model.py'), '--coverage', str(table / 'coverage.json'),
         '--model-root', str(args.model_root), '--out', str(args.out / 'bench'), '--markdown', str(args.markdown),
         '--workers', str(args.workers), '--atlas-mlir-bin', str(args.atlas_mlir_bin),
         '--min-tier', args.min_tier, '--command', COMMAND,
         '--note', 'Placement comes from `coverage/examples/causal_decoder/mapping.yaml`, which maps only ops an existing family '
                   'computes and cites why; the attention core stays unmapped because this decoder uses head dimension 8, while '
                   'the family supports 64, 128, and 256. The export is `export_ops.py causal_decoder --loader merlin:examples/workloads/causal_decoder/loader.py`, '
                   'the same command as `coverage/examples/causal_decoder/REPORT.md`, which lists the decoder with no placement rules.'],
    ]
    for step in steps:
        subprocess.run(step, env=environment, check=True)
    print(f'Bench report: {args.markdown}')


if __name__ == '__main__':
    main()
