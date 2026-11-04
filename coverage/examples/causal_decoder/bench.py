"""Export Merlin's small decoder, place it with this directory's starter mapping, and bench its Atlas launches.

MERLIN_ROOT=/path/to/merlin TORCH_PYTHON=/path/to/venv/bin/python NPU_MODEL=/path/to/npu-model \
  python3 -B coverage/examples/causal_decoder/bench.py --out /path/to/output

This is the "Starting a new model" path in the top README, end to end: export_ops.py --loader (TORCH_PYTHON needs
torch and NumPy), build_table.py --allow-unmapped with mapping.yaml here (the driver Python needs PyYAML), and
bench_model.py on npu_model 47b8599 (TORCH_PYTHON again). Exports, tables, and job results stay under --out; only
MODEL-BENCH.md is written next to this script.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).absolute().parent
COVERAGE = HERE.parents[1]
COMMAND = ('MERLIN_ROOT=/path/to/merlin TORCH_PYTHON=/path/to/venv/bin/python NPU_MODEL=/path/to/npu-model '
           'python3 -B coverage/examples/causal_decoder/bench.py --out "$OUT"')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--merlin-root', type=Path, default=os.environ.get('MERLIN_ROOT'))
    parser.add_argument('--torch-python', default=os.environ.get('TORCH_PYTHON', sys.executable))
    parser.add_argument('--model-root', type=Path, default=os.environ.get('NPU_MODEL'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, default=HERE / 'MODEL-BENCH.md')
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    if args.merlin_root is None or args.model_root is None:
        parser.error('set MERLIN_ROOT and NPU_MODEL, or pass --merlin-root and --model-root')
    args.out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    fixture, table = args.out / 'causal_decoder.json', args.out / 'table'
    steps = [
        [args.torch_python, '-B', str(COVERAGE / 'export_ops.py'), 'causal_decoder', '--loader',
         str(args.merlin_root / 'examples/workloads/causal_decoder/loader.py'), '--out', str(args.out / 'export'), '--fixture', str(fixture)],
        [sys.executable, '-B', str(COVERAGE / 'build_table.py'), '--inventory', f'causal_decoder={fixture}',
         '--mapping', str(HERE / 'mapping.yaml'), '--allow-unmapped', '--out', str(table)],
        [args.torch_python, '-B', str(COVERAGE / 'bench_model.py'), '--coverage', str(table / 'coverage.json'),
         '--model-root', str(args.model_root), '--out', str(args.out / 'bench'), '--markdown', str(args.markdown),
         '--workers', str(args.workers), '--command', COMMAND,
         '--note', 'Placement comes from `coverage/examples/causal_decoder/mapping.yaml`, which maps only ops an existing family '
                   'computes and cites why; the attention core stays unmapped because attention v2 requires head dimension 64 and this '
                   'decoder uses 8. The export is `export_ops.py causal_decoder --loader merlin:examples/workloads/causal_decoder/loader.py`, '
                   'the same command as `coverage/examples/causal_decoder/REPORT.md`, which lists the decoder with no placement rules.'],
    ]
    for step in steps:
        subprocess.run(step, env=environment, check=True)
    print(f'Bench report: {args.markdown}')


if __name__ == '__main__':
    main()
