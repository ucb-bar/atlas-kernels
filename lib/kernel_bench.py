"""Shared bench driver: run each kernel's reference cases on npu_model and report exactness, fidelity, and cycles.

Every kernels/<family>/reference.py defines CASES (one representative real TinyLlama or SmolVLA shape per family
and model, plus variants) and build(case, rng), which returns a Case: the generated program, its staged arena,
the Atlas-arithmetic golden output where lib/golden has one, and the FP32 PyTorch reference of what the kernel
computes on the unquantized inputs. For each case the driver

- executes the program with lib/atlas_execution.Executor (npu_model 47b8599, numerics="rtl") and compares the
  output bytes with the golden (bit-exactness),
- reports fidelity of the decoded output against the FP32 reference (cosine similarity and relative error
  ||y - ref|| / ||ref||, the radiance-kernels precision table's measures), and
- with --timed, runs the same program and arena on the npu_model Core cycle model and reports its cycle count
  (npu_model model cycles, not RTL cycles), requiring the timed output to equal the functional one.

    python3 lib/kernel_bench.py --model-root NPU_MODEL --out OUT --timed [--family projection_gemm ...]
    python3 lib/kernel_bench.py --report OUT

A run ends by writing kernels/<family>/RESULTS.md and the combined coverage/RESULTS.md from the JSON results in
OUT; --report rewrites them from existing results.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
import atlas_kernels_path  # noqa: E402,F401
from atlas_vector_arithmetic import decode_bf16, decode_fp8  # noqa: E402
from smol_weight_relocate import DEFAULT_ARENA, relocate_assembly  # noqa: E402

DRAM_BASE = 0x90000000


@dataclass
class Case:
    source: str
    arena: bytes
    output_offset: int
    output_bytes: int
    reference: np.ndarray
    decode: object
    golden: bytes | None = None
    golden_name: str | None = None
    meta: dict = field(default_factory=dict)


# Tile codecs: the layouts atlas-mlir tools/tiny_full/fusion_layout.py stages for these programs.
def pack_bf16_tiles(value):
    value = np.asarray(value, dtype='<u2')
    m, n = value.shape
    return value.reshape(m // 32, 32, n // 32, 2, 16).transpose(0, 2, 3, 1, 4).copy().tobytes()


def unpack_bf16_tiles(data, m, n):
    return np.frombuffer(data, dtype='<u2').reshape(m // 32, n // 32, 2, 32, 16).transpose(0, 3, 1, 2, 4).reshape(m, n).copy()


def pack_fp8_tiles(value):
    value = np.asarray(value, dtype=np.uint8)
    m, n = value.shape
    return value.reshape(m // 32, 32, n // 32, 32).transpose(0, 2, 1, 3).copy().tobytes()


def unpack_fp8_tiles(data, m, n):
    return np.frombuffer(data, dtype=np.uint8).reshape(m // 32, n // 32, 32, 32).transpose(0, 2, 1, 3).reshape(m, n).copy()


def pack_fp8_weights(value):
    value = np.asarray(value, dtype=np.uint8)
    k, n = value.shape
    return value.reshape(k // 32, 32, n // 32, 32).transpose(2, 0, 1, 3).copy().tobytes()


_POSITIVE = np.array([c for c in range(1, 127) if (c >> 3) & 15], dtype=np.uint8)  # normal finite codes, ascending
_VALUES = np.zeros(256)
_VALUES[_POSITIVE] = decode_fp8(_POSITIVE)


def encode_fp8(values):
    """Nearest FP8 E4M3 code (ties to the even code), saturating at 448; values below half the smallest
    normal flush to +0, matching the device decode that flushes exponent-zero codes."""
    values = np.asarray(values, dtype=np.float64)
    magnitude = np.minimum(np.abs(values), 448.0)
    grid = _VALUES[_POSITIVE]
    upper = np.clip(np.searchsorted(grid, magnitude), 1, len(grid) - 1)
    lower = upper - 1
    pick_upper = (grid[upper] - magnitude < magnitude - grid[lower]) | (
        (grid[upper] - magnitude == magnitude - grid[lower]) & (_POSITIVE[upper] % 2 == 0))
    codes = np.where(pick_upper, _POSITIVE[upper], _POSITIVE[lower])
    codes = np.where(magnitude < grid[0] / 2, 0, codes).astype(np.uint8)
    return np.where((values < 0) & (codes != 0), codes | 0x80, codes).astype(np.uint8)


def fp8_scale(values):
    """Smallest power-of-two exponent e with max|values| / 2**e <= 448 (the capture's compile-time scale rule)."""
    peak = float(np.max(np.abs(values)))
    return int(np.ceil(np.log2(peak / 448.0))) if peak > 0 else 0


def quantize(values):
    exponent = fp8_scale(values)
    return encode_fp8(np.ldexp(values, -exponent)), exponent


def bf16_words(values):
    """FP32 values to BF16 words with round-to-nearest-even."""
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((words + np.uint32(0x7FFF) + ((words >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_values(words):
    return decode_bf16(np.asarray(words, dtype=np.uint16)).astype(np.float32)


def fp8_values(codes, exponent=0):
    return np.ldexp(decode_fp8(np.asarray(codes, dtype=np.uint8)), exponent).astype(np.float32)


def relocate(source):
    return relocate_assembly(source, DEFAULT_ARENA, DRAM_BASE)


def projection_operands(spec, rng):
    """Seeded FP32 activations N(0, 1) and weights N(0, 0.02**2) at the case's physical shape, with logical
    rows beyond spec['logical_rows'] zero (decode padding), and their FP8 codes and power-of-two scales."""
    m, k, n = spec['shape']
    rows = spec.get('logical_rows', m)
    a = np.zeros((m, k), dtype=np.float32)
    a[:rows] = rng.standard_normal((rows, k), dtype=np.float32)
    w = (0.02 * rng.standard_normal((k, n))).astype(np.float32)
    return a, w, quantize(a), quantize(w)


def stage_projection(m, k, n, a_codes, w_codes, rescale, epilogue=None, extras=b'', postpass=None):
    """Generated projection program and arena, as atlas-mlir tools/tiny_full/fusion.py prepare_projection stages
    them: A tiles, B tiles, a 256-byte guard, the poisoned output extent, a guard, then epilogue extras."""
    from atlas_program import assembly
    output_offset = m * k + k * n + 256
    extent = epilogue.output_bytes if epilogue else 2 * m * n
    arena = pack_fp8_tiles(a_codes) + pack_fp8_weights(w_codes) + b'\xa5' * 256 + b'\xcd' * extent + b'\x5a' * 256 + extras
    source = assembly(m, k, n, output_offset, rescale, traffic_reuse=True, epilogue=epilogue)
    if postpass is not None:
        source = postpass(source, epilogue)
    return relocate(source), arena, output_offset, extent


def fidelity(actual, reference):
    a, r = np.asarray(actual, dtype=np.float64).ravel(), np.asarray(reference, dtype=np.float64).ravel()
    cosine = float(a @ r / (np.linalg.norm(a) * np.linalg.norm(r)))
    return {'cosine': cosine, 'relative_error': float(np.linalg.norm(a - r) / np.linalg.norm(r)),
            'max_abs_error': float(np.max(np.abs(a - r)))}


def timed_cycles(model_root, source, arena, max_cycles):
    """npu_model Core cycle count for one program and arena (the atlas-mlir tools/tiny_full/d1-model.py method)."""
    sys.path.insert(0, str(model_root))
    import torch
    from npu_model.configs import isa_definition  # noqa: F401
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.hardware import Core
    from npu_model.util.converter import input_to_program
    import io

    class QuietLogger:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    program = input_to_program(io.StringIO(source))
    program.dram_base = DRAM_BASE
    config = DefaultHardwareConfig()
    config.arch_state_config = replace(config.arch_state_config, numerics='rtl', randomize_init=True, init_seed=42)
    program.memory_regions = [(0, torch.frombuffer(bytearray(arena), dtype=torch.uint8))]
    core = Core(config=config, logger=QuietLogger())
    try:
        core.load_program(program)
        while not core.is_finished() and core.cycle_count < max_cycles:
            core.tick()
        if not core.is_finished():
            raise TimeoutError(f'Core did not finish within {max_cycles} cycles')
        state = core.arch_state
        return core.cycle_count, bytes(state.read_dram(DRAM_BASE, len(arena)).tolist())
    finally:
        core.close()


def families():
    return sorted(path.parent.name for path in (ROOT / 'kernels').glob('*/reference.py'))


def load_reference(family):
    """Each family's reference.py is loaded by path; the shared basename is never imported by bare name."""
    spec = importlib.util.spec_from_file_location(f'{family}_reference', ROOT / 'kernels' / family / 'reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(family, index, model_root, out, timed, max_cycles):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import torch
    torch.set_num_threads(1)
    from atlas_execution import Executor
    module = load_reference(family)
    spec = module.CASES[index]
    rng = np.random.default_rng(spec.get('seed', 0))
    case = module.build(spec, rng)
    started = time.perf_counter()
    execution = Executor(model_root).execute(case.source, case.arena, case.output_offset, case.output_bytes,
                                             max_instructions=50_000_000)
    output = execution.output
    result = {'family': family, 'case': spec['name'], 'model': spec['model'], 'shape': spec['shape'],
              'note': spec.get('note', ''), **case.meta, 'program_words': execution.program_words,
              'instructions': execution.instructions, 'functional_seconds': round(time.perf_counter() - started, 1),
              'model_revision': execution.model['revision'], 'numerics': execution.model['numerics'],
              'golden': case.golden_name, 'bit_exact': None if case.golden is None else output == case.golden,
              **fidelity(case.decode(output), case.reference)}
    if case.golden is not None:
        result['golden_fidelity'] = fidelity(case.decode(case.golden), case.reference)
    if timed:
        started = time.perf_counter()
        cycles, arena = timed_cycles(model_root, case.source, case.arena, max_cycles)
        result.update(model_cycles=cycles, timed_seconds=round(time.perf_counter() - started, 1),
                      timed_output_equal=arena[case.output_offset:case.output_offset + case.output_bytes] == output)
    path = out / f'{family}-{spec["name"]}.json'
    path.write_text(json.dumps(result, indent=1) + '\n')
    return result


def fmt(value, digits=5):
    return '' if value is None else f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def table(rows):
    lines = ['| case | model | shape | golden | bit-exact | cosine | rel-err | model cycles | instructions |',
             '| --- | --- | --- | --- | :-: | ---: | ---: | ---: | ---: |']
    for row in rows:
        exact = 'n/a' if row['bit_exact'] is None else 'exact' if row['bit_exact'] else 'DIFFERS'
        cycles = f"{row['model_cycles']:,}" if 'model_cycles' in row else 'not run'
        lines.append(f"| {row['case']} | {row['model']} | {'x'.join(map(str, row['shape']))} | {row['golden'] or 'none'} | "
                     f"{exact} | {fmt(row['cosine'])} | {100 * row['relative_error']:.3f}% | {cycles} | "
                     f"{row['instructions']:,} |")
    return lines


def report(out):
    rows = [json.loads(path.read_text()) for path in sorted(Path(out).glob('*.json'))]
    header = ['Generated by `python3 lib/kernel_bench.py --model-root NPU_MODEL --out OUT --timed` (10,675 CPU-seconds in all, 1,722 for the longest case, '
              'so 13 workers finish in about 30 minutes; `--report OUT` rewrites the tables from saved results). Inputs are '
              'seeded synthetic tensors at the real model shape; outputs run on npu_model 47b8599 with numerics="rtl". '
              'Bit-exact compares the output bytes with the lib/golden Atlas-arithmetic reference where one exists. '
              'Cosine and rel-err (||y - ref|| / ||ref||) compare the decoded output with the FP32 PyTorch reference '
              'on the unquantized inputs, so they include FP8 input quantization. Model cycles are npu_model Core '
              'cycle-model counts, not RTL cycles.', '']
    combined = ['# Kernel bench results', '', *header]
    for family in sorted({row['family'] for row in rows}):
        selected = [row for row in rows if row['family'] == family]
        lines = [f'# {family} bench results', '', *header, *table(selected), '']
        lines += [f"- {row['case']}: {row['note']}" for row in selected if row['note']]
        (ROOT / 'kernels' / family / 'RESULTS.md').write_text('\n'.join(lines).rstrip('\n') + '\n')
        combined += [f'## {family}', '', *table(selected), '']
    (ROOT / 'coverage' / 'RESULTS.md').write_text('\n'.join(combined).rstrip('\n') + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--family', action='append')
    parser.add_argument('--case', action='append', help='run only these case names')
    parser.add_argument('--timed', action='store_true')
    parser.add_argument('--max-cycles', type=int, default=200_000_000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.report:
        return report(args.report)
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(family, index) for family in (args.family or families())
            for index, spec in enumerate(load_reference(family).CASES) if not args.case or spec['name'] in args.case]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run, family, index, args.model_root, args.out, args.timed, args.max_cycles): (family, index)
                   for family, index in jobs}
        for future in as_completed(futures):
            row = future.result()
            print(json.dumps({key: row.get(key) for key in ('family', 'case', 'bit_exact', 'cosine', 'relative_error',
                                                            'model_cycles', 'functional_seconds', 'timed_seconds')}),
                  flush=True)
    report(args.out)


if __name__ == '__main__':
    main()
