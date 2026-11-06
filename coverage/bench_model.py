"""Bench every device launch a model's coverage places on Atlas, at the model's real shapes, on npu_model.

Input is the coverage.json that coverage/build_table.py writes (with or without --allow-unmapped). Its atlas_rows
and the mapping rules' launch hints become distinct device launches: projection panels (plain projection_gemm or a
fused family's program), attention launches, and RMSNorm launches, each with its count in the model. For each
distinct launch the script

- checks known shape limits, reporting unsupported shapes as no-variant, then generates the program with the
  family's kernels/<family>/gen.py (a subprocess whose failures are reported as errors),
- when --atlas-mlir-bin is explicit, translates the program to machine IR through lib/atlas_backend.py, runs
  atlas-opt's stream and timing verifiers, and requires atlas-emit's words to equal npu_model's assembler words
  at --min-tier (default stream-verified),
- stages seeded synthetic inputs at that shape and runs the program on npu_model 47b8599 with numerics="rtl"
  (lib/atlas_execution.Executor), comparing the output bytes with the independent lib/golden reference where the
  family has one, and the decoded output with an FP32 PyTorch reference (cosine, relative error),
- runs the same program and arena on the npu_model Core cycle model (npu_model model cycles, not RTL cycles) and
  requires the timed output to equal the functional output.

It writes MODEL-BENCH.md and MODEL-BENCH.json into --out. Cycle totals are per-launch model cycles times the launch
count; they ignore host time and launch overhead.

    python coverage/bench_model.py --coverage COVERAGE_JSON [--model NAME] --model-root NPU_MODEL --out OUT [--workers 32]
    python coverage/bench_model.py --plan-only --coverage COVERAGE_JSON --out OUT
    python coverage/bench_model.py --report OUT [--markdown PATH]

Exit status is nonzero when an independent golden or a timed/functional comparison differs, or a job fails for a
reason other than a missing generator variant.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
import hashlib
from importlib import metadata
import json
from math import prod as product
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).absolute().parents[1]
PROJECTION_OPS = ('aten.linear.default', 'aten.conv2d.default', 'aten.conv2d.padding')
GOLDEN_FAMILIES = {'projection_gemm': 'ordered_matmul', 'rne_add': 'ordered_matmul + exact sum rounded to BF16',
                   'rope': 'ordered_matmul + RoPE RNE formula'}
SLOW_SECONDS = 1200  # the 20-minute threshold the report flags


class NoVariant(Exception):
    """A deliberate capability check excludes this shape or staging policy."""


# ----------------------------------------------------------------------------------------------- launch plan

def pad32(value):
    return -(-value // 32) * 32


def row_matches(match, row):
    """Subset of build_table.matches without PyYAML: op, module, source, phase, component, dtype, and out."""
    output = row['outputs'][0] if row.get('outputs') else [[], '']
    for key, value in (match or {}).items():
        values = value if isinstance(value, list) else [value]
        if key == 'dtype':
            if output[1] not in values:
                return False
        elif key == 'out':
            if list(output[0]) != list(value):
                return False
        elif key in ('op', 'module', 'source', 'phase', 'component'):
            if not any(fnmatchcase(row.get(key, ''), pattern) for pattern in values):
                return False
        else:
            raise ValueError(f'unknown launch match field {key}')
    return True


def hints(row):
    value = row.get('launch')
    return [] if value is None else value if isinstance(value, list) else [value]


def row_hint(row):
    """The hint covering this row (its rows filter matches), or None."""
    covering = [hint for hint in hints(row) if row_matches(hint.get('rows'), row)]
    if len(covering) > 1:
        raise ValueError(f"row {row['module']} {row['op']} is covered by {len(covering)} launch hints")
    return covering[0] if covering else None


def projection_geometry(row):
    """Logical (M, K, N) of one Linear or Conv2d invocation."""
    out = row['outputs'][0][0]
    weight = row['inputs'][1][0]
    if row['op'] == 'aten.linear.default':
        return product(out[:-1]), weight[1], weight[0]
    return out[0] * product(out[2:]), product(weight[1:]), weight[0]


def panels(m, k, n, policy):
    """Physical panels [(logical_rows, M, K, N, logical_k, logical_n, multiplicity)] of one call."""
    rows_cap = policy.get('panel_rows', 128)
    widths = [n]
    for entry in policy.get('panel_columns', []):
        if 'max_k' not in entry or pad32(k) <= entry['max_k']:
            step = entry['n']
            widths = [min(step, n - start) for start in range(0, n, step)]
            break
    result = Counter()
    for start in range(0, m, rows_cap):
        rows = min(rows_cap, m - start)
        for width in widths:
            result[(rows, pad32(rows), pad32(k), pad32(width), k, width)] += 1
    return result


def mask_pattern(hint, rows, keys):
    masks = hint.get('masks', {})
    return masks.get(f'{rows}x{keys}', masks.get('default', 'all_visible'))


def attention_params(row, hint):
    out = row['outputs'][0][0]
    if len(out) != 4:
        raise ValueError('attention benchmark requires a rank-four output')
    if row['op'] == 'aten.scaled_dot_product_attention.default':
        shapes = [item[0] for item in row['inputs'][:3]]
        if len(shapes) != 3 or any(len(shape) != 4 for shape in shapes):
            raise ValueError('attention benchmark requires rank-four Q, K, and V')
        query_shape, key_shape, value_shape = shapes
        if len({shape[0] for shape in (*shapes, out)}) != 1:
            raise ValueError('attention Q, K, V, and output batch sizes must match')
        if key_shape != value_shape or query_shape[3] != key_shape[3] or query_shape != out:
            raise ValueError('attention Q, K, V, and output geometry is incompatible with the benchmark')
        _, heads, rows, head_dim = query_shape
        keys, kv_heads = key_shape[2], hint.get('kv_heads', key_shape[1])
    else:
        if any(len(item[0]) == 4 and item[0][0] != out[0] for item in row['inputs']):
            raise ValueError('attention input and output batch sizes must match')
        _, heads, rows, keys = out
        kv_heads, head_dim = hint.get('kv_heads', heads), hint.get('head_dim', 64)
    if out[0] != 1:
        raise ValueError(f'attention benchmark supports batch size 1 only; got {out[0]}')
    mask = hint.get('mask', 'none')
    params = dict(rows=rows, keys=keys, heads=heads, kv_heads=kv_heads, head_dim=head_dim, mask=mask,
                  output_fp8=bool(hint.get('output_fp8', False)))
    if mask == 'causal':
        params['query_start'] = keys - rows
    if mask == 'arbitrary':
        params['mask_pattern'] = mask_pattern(hint, rows, keys)
    return params


def launch_key(family, params):
    return family + ':' + json.dumps(params, sort_keys=True, separators=(',', ':'))


def rmsnorm_eps(row, hint):
    """Keep a captured epsilon, or require the mapping to supply decomposed norm policy."""
    def number(value):
        if isinstance(value, bool):
            raise ValueError('RMSNorm eps must be positive and finite')
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError('RMSNorm eps must be positive and finite') from None
        if not math.isfinite(result) or result <= 0:
            raise ValueError('RMSNorm eps must be positive and finite')
        return result

    captured = number(row['eps']) if 'eps' in row else None
    selected = number(hint['eps']) if 'eps' in hint else captured if captured is not None else 1e-5
    if captured is not None and selected != captured:
        raise ValueError(f"RMSNorm eps mismatch for {row['module']}: inventory {captured}, mapping {selected}")
    return selected


def plan(model_summary):
    """Distinct device launches with counts, the rows each covers, and rows that produce no launch."""
    rows = model_summary.get('atlas_rows')
    if rows is None:
        raise ValueError('coverage.json has no atlas_rows; regenerate it with this checkout of coverage/build_table.py')
    policy = model_summary.get('bench', {})
    launches = {}
    covered, uncovered = set(), []

    def add(family, params, count, row, flops):
        key = launch_key(family, params)
        entry = launches.setdefault(key, dict(key=key, family=family, params=params, count=0, flops=0,
                                              modules=[], dtypes=[], phases=[]))
        entry['count'] += count
        entry['flops'] += flops
        for field_name, value in (('modules', row['module'] or '(top)'), ('dtypes', row['outputs'][0][1] if row['outputs'] else ''),
                                  ('phases', row['phase'])):
            if value not in entry[field_name]:
                entry[field_name].append(value)

    # Fusions: (family, hint, module globs, phases of the hint's rows).
    fusions = []
    for index, row in enumerate(rows):
        hint = row_hint(row)
        if hint and hint.get('fuse_into'):
            covered.add(index)
            item = next((f for f in fusions if f[1] is hint or f[1] == hint), None)
            if item is None:
                fusions.append((hint['family'], hint, hint['fuse_into'], {row['phase']}))
            else:
                item[3].add(row['phase'])
    for index, row in enumerate(rows):
        hint = row_hint(row)
        if row['op'] in PROJECTION_OPS and row['where'] == 'atlas:projection_gemm' and 'component' not in row:
            covered.add(index)
            matched = [f for f in fusions if row['phase'] in f[3] and any(fnmatchcase(row['module'], g) for g in f[2])]
            if len(matched) > 1:
                raise ValueError(f"{row['module']} ({row['phase']}) is the target of {len(matched)} fusions")
            family, fusion = (matched[0][0], matched[0][1]) if matched else ('projection_gemm', {})
            m, k, n = projection_geometry(row)
            for (logical_rows, pm, pk, pn, lk, ln), multiplicity in panels(m, k, n, policy).items():
                params = dict(m=pm, k=pk, n=pn, rows=logical_rows, logical_k=lk, logical_n=ln)
                if family == 'epilogue':
                    params.update(steps=fusion['steps'], output=fusion.get('output', 'bf16'))
                    if fusion.get('consumer_width'):
                        params['consumer_width'] = fusion['consumer_width']
                elif family == 'smol_fusion':
                    params.update(operations=fusion.get('operations', ['bias', 'gelu_tanh']),
                                  output=fusion.get('output', 'fp8'), module=row['module'].replace('*', '0'))
                add(family, params, row['count'] * multiplicity, row, 2 * logical_rows * lk * ln * row['count'] * multiplicity)
            continue
        if hint and hint.get('per') is not None:
            covered.add(index)
            if row_matches(hint['per'], row):
                if hint['family'] == 'attention':
                    params = attention_params(row, hint)
                elif hint['family'] == 'rmsnorm':
                    out = row['outputs'][0][0]
                    params = dict(rows=product(out[:-1]), m=pad32(product(out[:-1])), width=hint.get('width', out[-1]),
                                  eps=rmsnorm_eps(row, hint))
                else:
                    raise ValueError(f"no per-row launch rule for family {hint['family']}")
                add(hint['family'], params, row['count'], row, 0)
            continue
        if hint is not None:
            covered.add(index)
    # Rows inside a launch: same rule and module as a per-row or fused hint, or a fused component (fc1 bias).
    for index, row in enumerate(rows):
        if index in covered:
            continue
        family = row['where'].split(':', 1)[1]
        if 'component' in row and any(f[0] == family and any(fnmatchcase(row['module'], g) for g in f[2]) for f in fusions):
            continue
        if row['flops_each'] == 0 and row_hint(row) is None and hints(row):
            continue
        uncovered.append(row)
    # FLOPs of per-row launches: every Atlas row of that rule, module, and phase.
    for entry in launches.values():
        if entry['flops'] == 0:
            entry['flops'] = sum(row['flops_each'] * row['count'] for row in rows
                                 if row['where'] == f"atlas:{entry['family']}" and row['module'] in entry['modules']
                                 and row['phase'] in entry['phases'] and launch_matches_shape(entry, row))
    ordered = sorted(launches.values(), key=lambda e: (e['family'], e['key']))
    return ordered, uncovered


def launch_matches_shape(entry, row):
    """Attribute an attention or norm row to the launch whose geometry its output carries."""
    out = row['outputs'][0][0] if row['outputs'] else []
    p = entry['params']
    if entry['family'] == 'attention':
        return len(out) == 4 and out[2] == p['rows'] and out[3] in (p['keys'], p['head_dim'])
    if entry['family'] == 'rmsnorm':
        return (len(out) >= 2 and product(out[:-1]) == p['rows']
                and rmsnorm_eps(row, row_hint(row) or {}) == p.get('eps', 1e-5))
    return True


def describe(entry):
    p, family = entry['params'], entry['family']
    if family == 'attention':
        text = f"{p['rows']}x{p['keys']} {p['heads']}/{p['kv_heads']} d{p['head_dim']} {p['mask']}"
        if p.get('mask_pattern'):
            text += f" ({p['mask_pattern']})"
        return text + (' FP8 out' if p['output_fp8'] else '')
    if family == 'rmsnorm':
        return (f"{p['m']}x{p['width']}" + (f" ({p['rows']} rows)" if p['rows'] != p['m'] else '')
                + f" eps={p.get('eps', 1e-5):g}")
    text = f"{p['m']}x{p['k']}x{p['n']}"
    logical = (p['rows'], p['logical_k'], p['logical_n'])
    if logical != (p['m'], p['k'], p['n']):
        text += f" (logical {'x'.join(map(str, logical))})"
    if family == 'epilogue':
        text += ' ' + '+'.join(p['steps']) + (' FP8 out' if p['output'] == 'fp8' else '')
    if family == 'smol_fusion':
        text += ' ' + '+'.join(p['operations']) + (' FP8 out' if p['output'] == 'fp8' else '')
    return text


# ----------------------------------------------------------------------------------------------- execution

@dataclass
class Built:
    source: str
    arena: bytes
    output_offset: int
    output_bytes: int
    reference: object
    decode: object
    golden: object = None  # callable returning the golden output bytes
    variant: str = ''
    generator: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    logical: object = None  # raw output bytes -> words the model consumes (padding rows and columns removed)


def seed(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def generator(family, arguments, work, model_root):
    """Run kernels/<family>/gen.py; unexpected failures must fail the benchmark."""
    out = Path(tempfile.mkdtemp(prefix=f'{family}-', dir=work))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(model_root), ATLAS_NPU_MODEL_ROOT=str(model_root))
    command = [sys.executable, '-B', str(ROOT / 'kernels' / family / 'gen.py'), '--out', str(out), '--name', 'program', *arguments]
    process = subprocess.run(command, capture_output=True, text=True, env=environment)
    text = f"kernels/{family}/gen.py {' '.join(shlex.quote(a) for a in arguments)}"
    if process.returncode:
        lines = [line for line in process.stderr.strip().splitlines() if line.strip()]
        raise RuntimeError(f"{text}: {lines[-1] if lines else f'exit {process.returncode}'}")
    record = json.loads((out / 'program.json').read_text())
    return (out / 'program.S').read_text(), dict(record, command=text)


class Projection:
    """Synthetic or supplied FP32 physical operands, with shared quantization and result plumbing."""
    def __init__(self, p, rng, operands=None):
        import numpy as np
        import kernel_bench as bench
        self.m, self.k, self.n = p['m'], p['k'], p['n']
        if operands is None:
            self.a = np.zeros((self.m, self.k), dtype=np.float32)
            self.w = np.zeros((self.k, self.n), dtype=np.float32)
            self.a[:p['rows'], :p['logical_k']] = rng.standard_normal((p['rows'], p['logical_k']), dtype=np.float32)
            self.w[:p['logical_k'], :p['logical_n']] = (0.02 * rng.standard_normal((p['logical_k'], p['logical_n']))).astype(np.float32)
        else:
            if not isinstance(operands, (tuple, list)) or len(operands) != 2:
                raise ValueError('projection_operands requires an (A, W) pair of physical float32 arrays')
            self.a, self.w = (np.asarray(value) for value in operands)
            for name, value, shape in (('A', self.a, (self.m, self.k)), ('W', self.w, (self.k, self.n))):
                if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f'projection operand {name} must be finite float32 with physical shape {shape}')
            self.a, self.w = self.a.copy(), self.w.copy()
        self.a_codes, self.a_scale = bench.quantize(self.a)
        self.w_codes, self.w_scale = bench.quantize(self.w)
        self.scale, self.shape = self.a_scale + self.w_scale, f'{self.m}x{self.k}x{self.n}'

    def arena(self, extent, extras=b''):
        import kernel_bench as bench
        return bench.pack_fp8_tiles(self.a_codes) + bench.pack_fp8_weights(self.w_codes) + b'\xa5' * 256 + b'\xcd' * extent + b'\x5a' * 256 + extras

    def finish(self, source, record, expected, variant, *, golden=None, extras=b'', offset=None,
               extent=None, decode=None, arena=None, **meta):
        import kernel_bench as bench
        extent = 2 * self.m * self.n if extent is None else extent
        return Built(source, self.arena(extent, extras) if arena is None else arena,
                     self.m * self.k + self.k * self.n + 256 if offset is None else offset, extent, expected,
                     decode or (lambda raw, m=self.m, n=self.n: bench.bf16_values(bench.unpack_bf16_tiles(raw, m, n))),
                     golden, variant, record, dict(a_scale=self.a_scale, w_scale=self.w_scale, **meta))


def recipe(m, k, n):
    import atlas_program
    try:
        return atlas_program._reuse_module(m, k, n).removeprefix('atlas_program_reuse_')
    except ValueError:
        return None


def build_projection_gemm(p, rng, gen, *, operands=None):
    import kernel_bench as bench
    from atlas_vector_arithmetic import ordered_matmul, rescale_bf16_array
    reference = bench.load_reference('projection_gemm')
    projection = Projection(p, rng, operands=operands)
    if recipe(p['m'], p['k'], p['n']) is not None:
        source, record = gen('projection_gemm', ['--shape', projection.shape, '--rescale-exponent', str(projection.scale), '--traffic-reuse'])
        variant = f"{record['recipe'].removeprefix('atlas_program_reuse_')} recipe"
    else:
        source, record = gen('projection_gemm', ['--shape', projection.shape, '--rescale-exponent', str(projection.scale)])
        variant = 'default schedule (no traffic-reuse recipe for this shape)'
    return projection.finish(bench.relocate(source), record, reference.reference(projection.a, projection.w), variant,
                             golden=lambda a=projection.a_codes, w=projection.w_codes, scale=projection.scale:
                                 bench.pack_bf16_tiles(rescale_bf16_array(ordered_matmul(a, w), scale)))


def build_epilogue(p, rng, gen):
    import numpy as np
    import torch
    import kernel_bench as bench
    reference = bench.load_reference('epilogue')
    m, k, n = p['m'], p['k'], p['n']
    projection = Projection(p, rng)
    steps = p['steps']
    if steps == ['silu'] and p['output'] == 'bf16':
        source, record = gen('epilogue', ['--shape', projection.shape, '--steps', 'silu', '--rescale-exponent', str(projection.scale)])
        return projection.finish(bench.relocate(source), record, reference.reference(projection.a, projection.w),
                                 f"{record['recipe'].removeprefix('atlas_program_reuse_')} recipe + SiLU epilogue",
                                 offset=record['output_offset'], extent=record['output_bytes'])
    if steps == ['mul'] and p['output'] == 'fp8':
        gate_words = np.zeros((m, n), dtype=np.uint16)
        gate_words[:p['rows'], :p['logical_n']] = bench.bf16_words(torch.nn.functional.silu(
            torch.from_numpy(rng.standard_normal((p['rows'], p['logical_n']), dtype=np.float32))).numpy())
        gate = bench.bf16_values(gate_words)
        expected = reference.reference(projection.a, projection.w, gate)
        output_scale = bench.fp8_scale(expected)
        stride = 32 * p.get('consumer_width', n)
        source, record = gen('epilogue', ['--shape', projection.shape, '--steps', 'mul', '--fp8', '--rescale-exponent', str(projection.scale),
                                          '--output-scale-exponent', str(output_scale), '--output-row-stride', str(stride)])
        if record['operand_offset'] != record['output_offset'] + record['output_bytes'] + 256:
            raise ValueError('epilogue operand offset differs from the staged layout')

        def decode(raw):
            blocks = [bench.unpack_fp8_tiles(raw[r * stride:r * stride + n * 32], 32, n) for r in range(m // 32)]
            return bench.fp8_values(np.concatenate(blocks), output_scale)
        return projection.finish(bench.relocate(source), record, expected,
                                 f"{record['recipe'].removeprefix('atlas_program_reuse_')} recipe + multiply epilogue, FP8 out",
                                 offset=record['output_offset'], extent=record['output_bytes'], decode=decode,
                                 extras=bench.pack_bf16_tiles(gate_words), output_scale=output_scale)
    raise NoVariant(f'no bench staging for epilogue steps {steps} with {p["output"]} output')


def build_rne_add(p, rng, gen):
    import numpy as np
    import kernel_bench as bench
    from atlas_vector_arithmetic import decode_bf16, ordered_matmul, rescale_bf16_array, round_bf16
    reference = bench.load_reference('rne_add')
    m, k, n = p['m'], p['k'], p['n']
    projection = Projection(p, rng)
    residual_words = np.zeros((m, n), dtype=np.uint16)
    residual_words[:p['rows'], :p['logical_n']] = bench.bf16_words(rng.standard_normal((p['rows'], p['logical_n']), dtype=np.float32))
    source, record = gen('epilogue', ['--shape', projection.shape, '--steps', 'add_rne', '--rescale-exponent', str(projection.scale)])
    if record['operand_offset'] != record['output_offset'] + 2 * m * n + 256:
        raise ValueError('residual offset differs from the staged layout')

    def golden(a=projection.a_codes, w=projection.w_codes, scale=projection.scale):
        projected = rescale_bf16_array(ordered_matmul(a, w), scale)
        return bench.pack_bf16_tiles(round_bf16(decode_bf16(projected) + decode_bf16(residual_words)))
    return projection.finish(bench.relocate(source), record,
                             reference.reference(projection.a, projection.w, bench.bf16_values(residual_words)),
                             f"{record['recipe'].removeprefix('atlas_program_reuse_')} recipe + add_rne epilogue "
                             '(kernels/epilogue/gen.py --steps add_rne; kernels/rne_add/gen.py emits only the add sequence)',
                             offset=record['output_offset'], extras=bench.pack_bf16_tiles(residual_words), golden=golden)


def build_rope(p, rng, gen):
    import numpy as np
    import kernel_bench as bench
    from atlas_vector_arithmetic import decode_bf16, ordered_matmul, rescale_bf16_array, round_bf16
    reference = bench.load_reference('rope')
    m, k, n = p['m'], p['k'], p['n']
    projection = Projection(p, rng)
    source, record = gen('rope', ['--shape', projection.shape, '--rescale-exponent', str(projection.scale)])
    cos_words, sin_words = reference.tables(m)
    blob = b''.join(bench.pack_bf16_tiles(value[:m, :32]) for value in (cos_words, sin_words, sin_words ^ 0x8000))
    if record['table_offset'] != record['output_offset'] + 2 * m * n + 256:
        raise ValueError('RoPE table offset differs from the staged layout')
    heads = n // 64

    def golden(a=projection.a_codes, w=projection.w_codes, scale=projection.scale):
        x = decode_bf16(rescale_bf16_array(ordered_matmul(a, w), scale))
        cosine, sine = (np.tile(decode_bf16(words), heads) for words in (cos_words, sin_words))
        return bench.pack_bf16_tiles(round_bf16(decode_bf16(round_bf16(x * cosine)) + decode_bf16(round_bf16(reference.rotate_half(x) * sine))))
    return projection.finish(bench.relocate(source), record,
                             reference.reference(projection.a, projection.w, bench.bf16_values(cos_words), bench.bf16_values(sin_words)),
                             f"{record['recipe'].removeprefix('atlas_program_reuse_')} recipe + RoPE post-pass (positions 0..{m - 1})",
                             offset=record['output_offset'], extras=blob, golden=golden)


def build_rmsnorm(p, rng, gen):
    import numpy as np
    import kernel_bench as bench
    from atlas_rmsnorm_postpass import MASK_BYTES
    reference = bench.load_reference('rmsnorm')
    m, width = p['m'], p['width']
    h_words = np.zeros((m, width), dtype=np.uint16)
    h_words[:p['rows']] = bench.bf16_words(rng.standard_normal((p['rows'], width), dtype=np.float32))
    gamma_words = bench.bf16_words(1 + 0.1 * rng.standard_normal(width, dtype=np.float32))
    eps = p.get('eps', 1e-5)
    expected = reference.reference(bench.bf16_values(h_words), bench.bf16_values(gamma_words), eps=eps)
    scale = bench.fp8_scale(expected)
    source, record = gen('rmsnorm', ['--rows', str(m), '--width', str(width), '--norm-scale-exponent', str(scale),
                                   '--eps', str(eps)])
    masks = np.zeros((32, 32, 32), dtype='<u2')
    for i in range(32):
        masks[i, :, i] = 0x3f80
    h = bench.pack_bf16_tiles(h_words)
    output_offset, output_bytes = 2 * m * width, m * width
    if (record['output_offset'], record['mask_offset'], record['weight_offset']) != (
            output_offset, output_offset + output_bytes + 256, output_offset + output_bytes + 256 + MASK_BYTES):
        raise ValueError('RMSNorm offsets differ from the staged layout')
    arena = (h + b'\xcd' * output_bytes + b'\x5a' * 256 + b''.join(bench.pack_bf16_tiles(tile) for tile in masks)
             + bench.pack_bf16_tiles(np.broadcast_to(gamma_words, (32, width)).copy()) + b'\xa5' * 256)
    return Built(bench.relocate(source), arena, output_offset, output_bytes, expected,
                 lambda raw: bench.fp8_values(bench.unpack_fp8_tiles(raw, m, width), scale), None,
                 ('streamed tile-row RMSNorm, FP8 out' if record.get('variant') == 'streamed-tile-rows'
                  else 'standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out)'), record,
                 dict(output_scale=scale, eps=eps))


def attention_mask(p):
    import numpy as np
    rows, keys = p['rows'], p['keys']
    if p['mask'] == 'causal':
        return np.tril(np.ones((rows, keys), dtype=bool), p.get('query_start', keys - rows))
    mask = np.ones((rows, keys), dtype=bool)
    if p.get('mask_pattern') == 'prefix_state_block':
        mask[:keys - 1, keys - 1] = False  # the SmolVLA prefix state token starts a new attention block
    elif p.get('mask_pattern') not in (None, 'all_visible'):
        raise NoVariant(f"no synthetic mask pattern {p['mask_pattern']}")
    return mask


def build_attention(p, rng, gen, work, *, operands=None, mask_values=None):
    import numpy as np
    import kernel_bench as bench
    from atlas_attention_v2_program import AttentionV2Config, stage_v2, unpack_output_v2
    reference = bench.load_reference('attention')
    dim = p['head_dim']
    if dim == 64:
        config_type, stage, unpack = AttentionV2Config, stage_v2, unpack_output_v2
    elif dim in (128, 256):
        from atlas_attention_wide_program import AttentionWideConfig, stage_wide, unpack_output_wide
        config_type, stage, unpack = AttentionWideConfig, stage_wide, unpack_output_wide
    else:
        raise NoVariant(f"attention requires head dimension 64, 128, or 256; this launch has {dim}")
    rows, keys, heads, kv_heads = p['rows'], p['keys'], p['heads'], p['kv_heads']
    shapes = ((heads, rows, dim), (kv_heads, keys, dim), (kv_heads, keys, dim))
    if operands is None:
        values = [rng.standard_normal(shape, dtype=np.float32) for shape in shapes]
    else:
        if not isinstance(operands, (tuple, list)) or len(operands) != 3:
            raise ValueError('attention_operands requires a (Q, K, V) triple of physical float32 arrays')
        values = [np.asarray(value) for value in operands]
        for name, value, shape in zip(('Q', 'K', 'V'), values, shapes):
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f'attention operand {name} must be finite float32 with physical shape {shape}')
    words = [bench.bf16_words(value) for value in values]
    q, k, v = (bench.bf16_values(x) for x in words)
    if mask_values is None:
        mask = attention_mask(p)
    else:
        mask = np.asarray(mask_values)
        if p['mask'] != 'arbitrary' or mask.dtype != np.bool_ or mask.shape != (rows, keys) or not mask.any(axis=1).all():
            raise ValueError('attention mask must be boolean [rows,keys], with visible keys in each row, for arbitrary masking')
    expected = reference.reference(q, k, v, mask, scale=dim ** -0.5)
    exponents = dict(q_exponent=bench.fp8_scale(q), k_exponent=bench.fp8_scale(k), v_exponent=bench.fp8_scale(v))
    o_exponent = bench.fp8_scale(expected) if p['output_fp8'] else 0
    arguments = ['--rows', str(rows), '--keys', str(keys), '--heads', f'{heads}/{kv_heads}', '--mask', p['mask'],
                 '--q-exponent', str(exponents['q_exponent']), '--k-exponent', str(exponents['k_exponent']),
                 '--v-exponent', str(exponents['v_exponent']), '--o-exponent', str(o_exponent)]
    if p.get('variant'):
        arguments += ['--variant', p['variant']]
    if dim != 64:
        arguments += ['--head-dim', str(dim)]
    if p['mask'] == 'causal':
        arguments += ['--query-start', str(p['query_start'])]
    if p['mask'] == 'arbitrary':
        path = Path(tempfile.mkdtemp(prefix='mask-', dir=work)) / 'mask.npy'
        np.save(path, mask)
        arguments += ['--mask-npy', str(path)]
    if p['output_fp8']:
        arguments.append('--fp8')
    source, record = gen('attention', arguments)
    config = config_type(rows, keys, heads, kv_heads, head_dim=dim, mask=p['mask'], query_start=p.get('query_start'),
                         output_fp8=p['output_fp8'], o_exponent=o_exponent, **exponents)
    arena, offsets = stage(config, *words, mask_tiles=reference.mask_tiles(mask) if p['mask'] == 'arbitrary' else None)
    if offsets != record['offsets']:
        raise ValueError('attention offsets differ between gen.py and the staged synthetic arena')

    def decode(raw):
        value = unpack(config, raw)
        return bench.fp8_values(value, o_exponent) if p['output_fp8'] else bench.bf16_values(value)
    variant = ('attention v2' if dim == 64 else f'attention wide d{dim}') + (' packed decode' if config.decode_packed else '') + f", {offsets['unique_masks']} unique mask tiles"
    if p.get('variant'):
        variant += '; ' + p['variant']
    return Built(source, arena, offsets['output'], offsets['output_bytes'], expected, decode, None, variant, record,
                 dict(**exponents, o_exponent=o_exponent))


def build_smol_fusion(p, rng, gen):
    import numpy as np
    import kernel_bench as bench
    from atlas_program import dma_load_bytes
    from smol_fusion_fp8 import stage_fc1_fp8
    reference = bench.load_reference('smol_fusion')
    m, k, n = p['m'], p['k'], p['n']
    if p['operations'] != ['bias', 'gelu_tanh'] or p['output'] != 'fp8':
        raise NoVariant(f"no bench staging for smol_fusion {p['operations']} with {p['output']} output")
    projection = Projection(p, rng)
    bias_words = np.zeros(n, dtype=np.uint16)
    bias_words[:p['logical_n']] = bench.bf16_words(0.1 * rng.standard_normal(p['logical_n'], dtype=np.float32))
    expected = reference.reference(projection.a, projection.w, bench.bf16_values(bias_words))
    output_scale = bench.fp8_scale(expected)
    source, record = gen('smol_fusion', ['--shape', projection.shape, '--fp8', '--module', p['module'],
                                         '--rescale-exponent', str(projection.scale), '--output-scale-exponent', str(output_scale)])
    offset = m * k + k * n + 256
    row = dict(index=0, call=0, module=p['module'], physical_mkn=[m, k, n], logical_mkn=[m, k, n], logical_row_start=0,
               output_offset=offset, dram_bytes=offset + 2 * m * n + 256, recipe='traffic-reuse-v1',
               program_policy='traffic-reuse-v1', output_rescale_exponent=projection.a_scale + projection.w_scale, dram_base=0x90000000,
               requested_dma=dict(load_bytes=dma_load_bytes(m, k, n, traffic_reuse=True), store_bytes=2 * m * n))
    program = stage_fc1_fp8(row, projection.arena(2 * m * n), bias_bf16=bias_words.tobytes(),
                            output_scale_exponent=output_scale)
    if program.source != source:
        raise ValueError('smol_fusion program differs between gen.py and the staged synthetic arena')
    return projection.finish(source, record, expected,
                             f"{recipe(m, k, n)} recipe + fc1 bias and GELU (tanh), FP8 out (smol_fusion_fp8.stage_fc1_fp8)",
                             arena=program.arena, offset=offset, extent=m * n, output_scale=output_scale,
                             decode=lambda raw: bench.fp8_values(bench.unpack_fp8_tiles(raw, m, n), output_scale))


BUILDERS = {'projection_gemm': build_projection_gemm, 'epilogue': build_epilogue, 'rne_add': build_rne_add,
            'rope': build_rope, 'rmsnorm': build_rmsnorm, 'attention': build_attention, 'smol_fusion': build_smol_fusion}


def build(entry, work, model_root, *, projection_operands=None, attention_operands=None, attention_mask_values=None):
    """Stage a launch; supplied projection operands use physical float32 [M,K] and [K,N] arrays."""
    import numpy as np
    family = entry['family']
    if family not in BUILDERS:
        raise NoVariant(f'no model-shape bench builder for family {family}')
    if projection_operands is not None and family != 'projection_gemm':
        raise ValueError('projection_operands is supported only for projection_gemm launches')
    if (attention_operands is not None or attention_mask_values is not None) and family != 'attention':
        raise ValueError('attention_operands and attention_mask_values are supported only for attention launches')
    p = entry['params']
    # Deliberate capability exclusions happen before generation. Never infer unsupported shapes from
    # a generator's exit code or exception text, which can also describe broken implementations.
    if family in ('epilogue', 'rne_add', 'rope', 'smol_fusion') and recipe(p['m'], p['k'], p['n']) is None:
        raise NoVariant('traffic reuse is supported only for the qualified shapes')
    if family == 'rmsnorm':
        from atlas_rmsnorm_tiled import MAX_WIDTH
        if p['m'] not in (32, 128):
            raise NoVariant('RMSNorm benchmark physical rows must be 32 or 128')
        if p['width'] <= 0 or p['width'] % 32 or p['width'] > MAX_WIDTH:
            raise NoVariant(f'RMSNorm width must be a positive multiple of 32 <= {MAX_WIDTH} (current VMEM layout)')
    rng = np.random.default_rng(seed(entry['key']))
    gen = lambda fam, arguments: generator(fam, arguments, work, model_root)  # noqa: E731
    if family == 'attention':
        return build_attention(entry['params'], rng, gen, work, operands=attention_operands, mask_values=attention_mask_values)
    built = (build_projection_gemm(entry['params'], rng, gen, operands=projection_operands)
             if family == 'projection_gemm' else BUILDERS[family](entry['params'], rng, gen))
    if family in GOLDEN_FAMILIES:
        import kernel_bench as bench
        p = entry['params']
        built.logical = lambda raw: bench.unpack_bf16_tiles(raw, p['m'], p['n'])[:p['rows'], :p['logical_n']]
    return built


def compare(built, output, golden):
    """Bit-exactness over the logical extent, and the padding words (rows and columns the model drops) that differ."""
    import numpy as np
    result = dict(full_extent_exact=output == golden)
    if result['full_extent_exact'] or built.logical is None:
        return dict(result, bit_exact=result['full_extent_exact'], padding_differences=0)
    actual, expected = (np.frombuffer(value, dtype='<u2') for value in (output, golden))
    logical = bool(np.array_equal(built.logical(output), built.logical(golden)))
    differ = actual != expected
    pairs = {frozenset((int(a), int(b))) for a, b in zip(actual[differ], expected[differ])}
    padding = int(differ.sum()) - int((built.logical(output) != built.logical(golden)).sum())
    return dict(result, bit_exact=logical, padding_differences=padding,
                padding_signed_zero_only=logical and pairs <= {frozenset((0, 0x8000))})


def program_identity(built, atlas_mlir_bin=None, min_tier='stream-verified'):
    """Identify the exact staged inputs and cross-check the address-specific executable words."""
    import io
    from npu_model.util.converter import input_to_program
    words = tuple(int(word) for word in input_to_program(io.StringIO(built.source)).assemble())
    result = dict(source_sha256=hashlib.sha256(built.source.encode()).hexdigest(),
                  arena_sha256=hashlib.sha256(built.arena).hexdigest(), arena_bytes=len(built.arena),
                  program_words=len(words),
                  words_sha256=hashlib.sha256(b''.join(w.to_bytes(4, 'little') for w in words)).hexdigest())
    if atlas_mlir_bin is not None:
        import atlas_backend
        compiled = atlas_backend.cross_checked(built.source, words, bin_dir=atlas_mlir_bin, min_tier=min_tier)
        result['backend'] = dict(compiled.record(), words_equal_assembler=True, min_tier=min_tier,
                                 tools=atlas_backend.tools(atlas_mlir_bin).identity())
    else:
        result['backend'] = None
    return result


def checked_source(path):
    path = Path(path).absolute()
    if any(any(word in part.lower() for word in ('hammer', 'vlsi')) for part in path.parts):
        raise ValueError('excluded source path')
    if any(candidate.is_symlink() for candidate in (*reversed(path.parents), path)):
        raise ValueError('source identity requires a path without symbolic links')
    return path


def source_identity(root, python_only=False):
    """Hash source bytes, pruning excluded names and links before walking their descendants."""
    root = checked_source(root)
    excluded = lambda name: any(word in name.lower() for word in ('hammer', 'vlsi'))  # noqa: E731
    if not root.is_dir():
        raise ValueError('missing source directory: ' + str(root))
    sources = {}
    for directory, names, files in os.walk(root, followlinks=False):
        retained = []
        for name in sorted(names):
            if excluded(name) or name in ('.git', '__pycache__', '.pytest_cache'):
                continue
            if (Path(directory) / name).is_symlink():
                raise ValueError('source identity requires directories without symbolic links')
            retained.append(name)
        names[:] = retained
        for name in sorted(files):
            if excluded(name) or name.endswith(('.pyc', '.pyo')) or (python_only and not name.endswith('.py')):
                continue
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError('source identity requires files without symbolic links')
            if not path.is_file():
                continue
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def cache_context(model_root, atlas_mlir_bin=None, min_tier='stream-verified', tools=None):
    """Bind staging, golden/fidelity, generators, model code/data, runtime, and compiler policy."""
    packages = {}
    for name in ('numpy', 'torch'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    model_root = checked_source(model_root)
    revision = subprocess.run(['git', '-C', str(model_root), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    return dict(schema='atlas.model_bench.cache.v1',
                implementation={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__).absolute(), ROOT / 'atlas_kernels_path.py')},
                library=source_identity(ROOT / 'lib', python_only=True),
                generators=source_identity(ROOT / 'kernels', python_only=True),
                model=dict(root=str(model_root), revision=revision, sources=source_identity(model_root / 'npu_model')),
                runtime=dict(python=sys.version, executable=sys.executable, packages=packages),
                compiler=dict(enabled=atlas_mlir_bin is not None, min_tier=min_tier, tools=tools))


def cache_binding(entry, mode, context, max_cycles, max_instructions, timeout=None):
    return dict(context=context, key=entry['key'], family=entry['family'], params=entry.get('params'),
                mode=mode, max_cycles=max_cycles, max_instructions=max_instructions, timeout=timeout)


def staged_identity(built, identity):
    """Keep the deterministic executable/input/output layout separate from diagnostic text."""
    fields = ('source_sha256', 'words_sha256', 'program_words', 'arena_sha256', 'arena_bytes')
    helpers = {}
    for name in ('run_model_tile', 'chipyard', 'chipyard.arena'):
        filename = getattr(sys.modules.get(name), '__file__', None)
        if filename:
            path = checked_source(filename)
            helpers[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict({key: identity[key] for key in fields}, output_offset=built.output_offset,
                output_bytes=built.output_bytes, staging_helpers=helpers)


def write_job_result(path, result):
    """Publish a complete checkpoint or result even when the worker is about to time out."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=1) + '\n')
    temporary.replace(path)


def run_job(entry, mode, model_root, work, max_cycles, max_instructions, atlas_mlir_bin=None,
            min_tier='stream-verified', checkpoint=None, identity_only=False, timeout=None):
    """One functional or timed execution of one launch; returns a result dict (never raises for NoVariant)."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    sys.path.insert(0, str(model_root))
    sys.path.insert(0, str(ROOT))
    import atlas_kernels_path  # noqa: F401
    import torch
    torch.set_num_threads(1)
    import kernel_bench as bench
    tools = None
    if atlas_mlir_bin is not None:
        import atlas_backend
        tools = atlas_backend.tools(atlas_mlir_bin).identity()
    result = dict(key=entry['key'], mode=mode, cache_binding=cache_binding(
        entry, mode, cache_context(model_root, atlas_mlir_bin, min_tier, tools), max_cycles, max_instructions, timeout))
    started = time.perf_counter()
    try:
        built = build(entry, work, model_root)
    except NoVariant as error:
        return dict(result, status='no-variant', error=str(error))
    except Exception as error:  # noqa: BLE001 - staging and generator bugs are launch failures
        return dict(result, status='error', error=f'{type(error).__name__}: {error}',
                    seconds=round(time.perf_counter() - started, 1))
    result.update(variant=built.variant, generator_words=built.generator.get('words'), generator_command=built.generator.get('command'),
                  meta=built.meta)
    try:
        result.update(program_identity(built, atlas_mlir_bin, min_tier))
        result['cache_identity'] = staged_identity(built, result)
        if identity_only:
            return dict(result, status='identity')
        if checkpoint is not None:
            # Preserve the verified program identity if the parent terminates a slow timed execution.
            write_job_result(checkpoint, dict(result, status='running'))
        if mode == 'functional':
            from atlas_execution import Executor
            execution = Executor(model_root).execute(built.source, built.arena, built.output_offset, built.output_bytes,
                                                     max_instructions=max_instructions)
            output = execution.output
            golden = built.golden() if built.golden else None
            result.update(program_words=execution.program_words, instructions=execution.instructions,
                          model_revision=execution.model['revision'], numerics=execution.model['numerics'],
                          output_sha256=hashlib.sha256(output).hexdigest(),
                          **(dict(bit_exact=None) if golden is None else compare(built, output, golden)),
                          **bench.fidelity(built.decode(output), built.reference))
        else:
            cycles, arena = bench.timed_cycles(model_root, built.source, built.arena, max_cycles)
            output = arena[built.output_offset:built.output_offset + built.output_bytes]
            result.update(model_cycles=cycles, output_sha256=hashlib.sha256(output).hexdigest())
        result['status'] = 'ok'
    except Exception as error:  # noqa: BLE001 - every execution failure is reported per launch
        result.update(status='error', error=f'{type(error).__name__}: {error}')
    result['seconds'] = round(time.perf_counter() - started, 1)
    return result


def job_main(args):
    entry = json.loads(Path(args.job).read_text())
    result = run_job(entry, args.mode, args.model_root, Path(args.job).parent, args.max_cycles, args.max_instructions,
                     args.atlas_mlir_bin, args.min_tier, args.result, args.identity_only, args.job_timeout)
    write_job_result(args.result, result)
    return 0


def subset(entry, rows):
    """A row subset of a row-uniform attention launch, and the factor that scales its cycles to the full launch."""
    p = dict(entry['params'])
    if entry['family'] != 'attention' or p['mask'] not in ('none',) and p.get('mask_pattern') not in ('all_visible',):
        return None, None
    if p['rows'] <= rows or p['rows'] % rows:
        return None, None
    factor = p['rows'] // rows
    p['rows'] = rows
    return dict(entry, key=entry['key'] + f'#subset{rows}', params=p), factor


def execute(launches, args):
    work = args.out / 'jobs'
    work.mkdir(parents=True, exist_ok=True)
    results = defaultdict(dict)
    context = cache_context(args.model_root, args.atlas_mlir_bin, args.min_tier, args.backend_tools)

    def one(entry, mode, timeout):
        name = hashlib.sha256((entry['key'] + mode).encode()).hexdigest()[:16]
        job, out = work / f'{name}.job.json', work / f'{name}.{mode}.json'
        job.write_text(json.dumps(entry) + '\n')
        command = [sys.executable, '-B', str(Path(__file__).absolute()), '--job', str(job), '--mode', mode,
                   '--result', str(out), '--model-root', str(args.model_root), '--max-cycles', str(args.max_cycles),
                   '--max-instructions', str(args.max_instructions), '--job-timeout', str(timeout)]
        if args.atlas_mlir_bin:
            command += ['--atlas-mlir-bin', str(args.atlas_mlir_bin), '--min-tier', args.min_tier]
        environment = dict(os.environ, OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        if args.resume and out.exists():
            cached = json.loads(out.read_text())
            backend = cached.get('backend')
            # A compiler-enabled run must never silently reuse an assembler-only job or a weaker policy.
            allowed_tiers = ('verified', 'stream-verified', 'encode-only')
            policy_matches = (not args.atlas_mlir_bin and not backend) or (args.atlas_mlir_bin and backend and
                              backend.get('tier') in allowed_tiers[:allowed_tiers.index(args.min_tier) + 1] and
                              backend.get('min_tier') == args.min_tier and backend.get('tools') == args.backend_tools)
            binding = cache_binding(entry, mode, context, args.max_cycles, args.max_instructions, timeout)
            if cached.get('cache_binding') == binding and (cached.get('status') == 'no-variant' or
                    (cached.get('status') in ('ok', 'timeout') and policy_matches and cached.get('cache_identity'))):
                # Regeneration checks assembled words and input bytes even when implementation hashes match.
                identity_out = work / f'{name}.{mode}.identity.json'
                identity_out.unlink(missing_ok=True)
                identity_command = list(command)
                identity_command[identity_command.index('--result') + 1] = str(identity_out)
                try:
                    process = subprocess.run([*identity_command, '--identity-only'], capture_output=True,
                                             text=True, timeout=timeout, env=environment)
                    fresh = json.loads(identity_out.read_text()) if not process.returncode and identity_out.exists() else {}
                except (subprocess.TimeoutExpired, ValueError):
                    fresh = {}
                if fresh.get('cache_binding') == binding and (
                        cached['status'] == 'no-variant' and fresh.get('status') == 'no-variant'
                        and fresh.get('error') == cached.get('error') or
                        cached['status'] in ('ok', 'timeout') and fresh.get('status') == 'identity'
                        and cached.get('cache_identity') == fresh.get('cache_identity')):
                    return entry, mode, cached
        out.unlink(missing_ok=True)
        started = time.perf_counter()
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=environment)
        except subprocess.TimeoutExpired:
            result = json.loads(out.read_text()) if out.exists() else dict(key=entry['key'], mode=mode)
            result.update(status='timeout', seconds=round(time.perf_counter() - started, 1))
            if args.atlas_mlir_bin and not result.get('backend'):
                result.update(status='error', error='job timed out before compiler verification completed')
            write_job_result(out, result)
            return entry, mode, result
        if process.returncode or not out.exists():
            return entry, mode, dict(key=entry['key'], mode=mode, status='error', error=process.stderr.strip()[-2000:])
        return entry, mode, json.loads(out.read_text())

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one, entry, mode, args.subset_after if entry['family'] == 'attention' and mode == 'timed' else args.job_timeout)
                   for entry in launches for mode in ('functional', 'timed')]
        pending = list(futures)
        while pending:
            for future in as_completed(pending):
                pending.remove(future)
                entry, mode, result = future.result()
                results[entry['key']][mode] = result
                print(json.dumps({key: result.get(key) for key in ('key', 'mode', 'status', 'seconds', 'model_cycles', 'bit_exact', 'cosine')}), flush=True)
                if mode == 'timed' and result['status'] == 'timeout':
                    part, factor = subset(entry, args.subset_rows)
                    if part is not None:
                        results[entry['key']]['subset'] = dict(rows=args.subset_rows, factor=factor)
                        pending.append(pool.submit(one, part, 'timed-subset', args.job_timeout))
                if mode == 'timed-subset':
                    full = entry['key'].split('#subset')[0]
                    results[full]['subset'].update(result=result)
                break
    return results


def merge(launches, results):
    rows = []
    for entry in launches:
        functional = results.get(entry['key'], {}).get('functional', {})
        timed = results.get(entry['key'], {}).get('timed', {})
        part = results.get(entry['key'], {}).get('subset')
        row = dict(entry, shape=describe(entry))
        statuses = {functional.get('status'), timed.get('status')}
        if 'no-variant' in statuses and 'error' not in statuses:
            row.update(status='no-variant', error=(functional if functional.get('status') == 'no-variant' else timed).get('error'))
            rows.append(row)
            continue
        row.update(variant=functional.get('variant') or timed.get('variant'), generator_words=functional.get('generator_words'),
                   generator_command=functional.get('generator_command') or timed.get('generator_command'),
                   program_words=functional.get('program_words'), instructions=functional.get('instructions'),
                   functional_seconds=functional.get('seconds'), timed_seconds=timed.get('seconds'),
                   model_revision=functional.get('model_revision'), numerics=functional.get('numerics'),
                   golden=GOLDEN_FAMILIES.get(entry['family']), bit_exact=functional.get('bit_exact'),
                   full_extent_exact=functional.get('full_extent_exact'), padding_differences=functional.get('padding_differences'),
                   padding_signed_zero_only=functional.get('padding_signed_zero_only'),
                   cosine=functional.get('cosine'), relative_error=functional.get('relative_error'),
                   max_abs_error=functional.get('max_abs_error'), meta=functional.get('meta'))
        for key in ('backend', 'source_sha256', 'words_sha256', 'arena_sha256', 'arena_bytes', 'reuse_proof'):
            row[key] = functional.get(key)
        errors = [f"{mode}: {r.get('error') or r.get('status')}" for mode, r in (('functional', functional), ('timed', timed))
                  if r.get('status') not in ('ok', 'timeout')]
        if timed.get('status') == 'ok':
            row.update(model_cycles=timed['model_cycles'], cycles_source='full launch',
                       timed_output_equal=timed.get('output_sha256') == functional.get('output_sha256') if functional.get('status') == 'ok' else None)
        elif timed.get('status') == 'timeout':
            row['timed_note'] = f"timed run exceeded {timed['seconds']:.0f} s"
            if timed.get('note'):
                row['timed_provenance_note'] = timed['note']
            done = (part or {}).get('result', {})
            if done.get('status') == 'ok':
                row.update(model_cycles=done['model_cycles'] * part['factor'], timed_output_equal=None,
                           cycles_source=f"estimate: {part['rows']}-row subset ({done['model_cycles']:,} cycles) x {part['factor']}",
                           subset_seconds=done.get('seconds'))
            elif done.get('status') == 'error':
                errors.append(f"timed-subset: {done.get('error')}")
        if functional.get('status') == 'timeout':
            errors.append(f"functional: exceeded {functional['seconds']:.0f} s")
        row['status'] = 'error' if errors else 'ok'
        if errors:
            row['error'] = '; '.join(errors)
        if 'model_cycles' in row:
            row['total_cycles'] = row['model_cycles'] * entry['count']
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------- report

def shape_text(record):
    output = record['outputs'][0] if record.get('outputs') else [[], '']
    inputs = ', '.join('x'.join(map(str, shape)) or 'scalar' for shape, _ in record.get('inputs', [])[:2])
    return f"{inputs or '-'} -> {'x'.join(map(str, output[0])) or 'scalar'} {output[1]}"


def checks(model, launches):
    """Derived launch counts against the qualified kit's declared counts, where the coverage declares them."""
    kit = model.get('kit_launches', {})
    derived = Counter()
    for entry in launches:
        if entry['family'] == 'attention':
            derived['attention'] += entry['count']
        elif entry['family'] == 'rmsnorm':
            derived['rmsnorm'] += entry['count']
        else:
            derived['projection_panels'] += entry['count']
    return [(key, derived.get(key, 0), kit.get(key)) for key in ('projection_panels', 'attention', 'rmsnorm')
            if key in kit or derived.get(key)]


def render(document):
    model, rows = document['model'], document['launches']
    seconds = [row.get(key) or 0 for row in rows for key in ('functional_seconds', 'timed_seconds', 'subset_seconds')]
    job_seconds, longest = sum(seconds), max(seconds, default=0)
    shape_kind = 'synthetic shapes' if model == 'shape_sweep' else 'model shapes'
    lines = [f"# {model} kernel bench at {shape_kind}", '',
             f"Regenerate: `{document['command']}`", '',
             f"Input: `{document['coverage']}` (SHA-256 `{document['coverage_sha256']}`), coverage status {document['coverage_status']}. "
             f"The coverage derives {len(rows):,} distinct launches; {sum(row['status'] == 'ok' for row in rows):,} executed "
             f"on npu_model `{document['npu_model']}` with numerics=\"rtl\" "
             f"on seeded synthetic inputs at {shape_kind} ({document['workers']} worker processes; {job_seconds:,.0f} job-seconds in all, "
             f"longest job {longest:,.0f} s).", '',
             'Model cycles are npu_model Core cycle-model counts, not RTL cycles. Cycles x count multiplies one launch by the number of '
             'launches in the model; the sums below are device-only estimates that ignore host time, launch overhead, and data '
             'movement between launches. Bit-exact compares output bytes with the independent lib/golden reference where the family '
             'has one; cosine and rel-err compare the decoded output with an FP32 PyTorch reference on the unquantized inputs, so they '
             'include FP8 input quantization. Inputs are synthetic, so fidelity and data-dependent cycles are representative, not the model\'s.', '']
    lines += ['A distinct launch groups the same shape and variant; a distinct program is the address-specific program text, '
              'as in the separate 1,589-program kit check.', '',
              'Shares below are by launch kind: `epilogue`, `rne_add`, `rope`, and `smol_fusion` include their GEMM projection '
              'and fused stage, while `projection_gemm` covers unfused panels. They are not shares of non-GEMM time; '
              'the generator metadata does not provide separately measured cycles for those stages.', '']
    if document.get('backend'):
        lines += [f"Compiler backend: machine IR, atlas-opt stream and timing verifiers, and atlas-emit, with minimum tier "
                  f"`{document['backend']['min_tier']}`. Successful jobs require emitted words to equal npu_model's assembler; "
                  'per-job JSON records binary SHA-256 digests, diagnostics, and word counts.', '']
    else:
        lines += ['Programs were assembled by npu_model only, with no compiler tier.', '']
    reconciliation = document['launch_checks']
    if reconciliation:
        lines += ['| device launches | derived here | qualified kit |', '| --- | ---: | ---: |']
        lines += [f"| {key} | {value:,} | {'not declared' if kit is None else f'{kit:,}'} |" for key, value, kit in reconciliation]
        lines.append('')
    ok = [row for row in rows if 'total_cycles' in row]
    by_family = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        item = by_family[row['family']]
        item[0] += row['count']
        if 'total_cycles' in row:
            item[1] += row['total_cycles']
            item[2] += row['count']
    total = sum(value[1] for value in by_family.values())
    estimated = [row for row in ok if row.get('cycles_source', 'full launch') != 'full launch']
    estimated_count = sum(r['count'] for r in estimated)
    total_label = f'{total:,}' + (f' (includes {estimated_count:,} estimated launches)' if estimated else '')
    caveat = (f" {sum(r['count'] for r in estimated):,} of those launches use a row-subset estimate ("
              + ', '.join(f"`{r['family']}` {r['shape']}" for r in estimated) + ').') if estimated else ''
    lines += ['## Summary', '', f"Device cycle estimate (sum of model cycles x count over {sum(r['count'] for r in ok):,} of "
              f"{sum(r['count'] for r in rows):,} launches): {total_label} npu_model model cycles. It ignores host time and launch overhead.{caveat}", '']
    if document.get('reuse_proof'):
        proof = document['reuse_proof']['path']
        retained = document['reuse_proof'].get('sha256') in (
            'a4fee0b4f03633063d758880ef44e4f50156de9d4a8bffee14ade4a70c7f515f',
            '7745da6b8410d7fb6a2456e4699af8c048f340be4d51be47521208d5bec1437f',
            '5a4a685e3467b80d2b1d9ce20806dcf554c9f2aa233d90d400253c86b1b7573d')
        freshness = '; zero functional or timed bench jobs were rerun for this retained report' if retained else ''
        lines += [f"These totals include historically reused functional and timed results{freshness}. "
                  f"Reconstructed staging, program-word checks, and per-job result hashes are recorded in "
                  f"`{proof}` beside the generated JSON.", '']
    tiers = Counter(row['backend']['tier'] for row in rows if row.get('backend'))
    lines += ['Compiler tier counts by distinct launch: ' + (', '.join(f'{count:,} `{tier}`' for tier, count in sorted(tiers.items()))
              if tiers else 'none') + '.', '',
              '| launch kind | launches | launches with cycles | model cycles x count | share |', '| --- | ---: | ---: | ---: | ---: |']
    for family, (count, cycles, timed) in sorted(by_family.items(), key=lambda item: -item[1][1]):
        lines.append(f"| `{family}` | {count:,} | {timed:,} | {cycles:,} | {100 * cycles / total if total else 0:.2f}% |")
    unsupported = [row for row in rows if row['status'] != 'ok']
    lines += ['', f"Unsupported or failed launches: {len(unsupported)}."]
    for row in unsupported:
        lines.append(f"- `{row['family']}` {row['shape']} x{row['count']:,}: {row['status']}: {row.get('error', '')}")
    estimates = [row for row in rows if row.get('cycles_source', 'full launch') != 'full launch' or row.get('timed_note')]
    slow = [row for row in rows if max(row.get('functional_seconds') or 0, row.get('timed_seconds') or 0) > SLOW_SECONDS]
    if estimates or slow:
        lines += ['', 'Shapes that took more than 20 minutes on npu_model:']
        for row in sorted({id(r): r for r in estimates + slow}.values(), key=lambda r: r['key']):
            provenance = row.get('timed_provenance_note')
            if (not provenance and document.get('reuse_proof', {}).get('sha256') ==
                    '7745da6b8410d7fb6a2456e4699af8c048f340be4d51be47521208d5bec1437f'
                    and model == 'smolvla' and row['family'] == 'attention' and row.get('timed_note')
                    and row.get('params', {}).get('rows') == 1024 and row['params'].get('keys') == 1024):
                provenance = ('timeout marker reconstructed from bench-smolvla.log after the merge crash '
                              '(retained log: 1,200.1 s)')
            suffix = f"; timeout provenance: {provenance.rstrip('.')}" if provenance else ''
            lines.append(f"- `{row['family']}` {row['shape']}: functional {row.get('functional_seconds')} s, timed "
                         f"{row.get('timed_seconds')} s; {row.get('timed_note', 'completed')}; "
                         f"cycles: {row.get('cycles_source', 'none')}{suffix}")
    lines.append('')
    lines += ['## Launches', '', '| family | shape | export dtype | variant | tier | words | count | model cycles | cycles x count | golden | '
              'bit-exact | timed equals functional | cosine | rel-err | modules |',
              '| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | :-: | :-: | ---: | ---: | --- |']
    for row in sorted(rows, key=lambda row: (row['status'] != 'ok' or 'total_cycles' not in row,
                                             -row.get('total_cycles', 0))):
        if row['status'] == 'no-variant':
            lines.append(f"| `{row['family']}` | {row['shape']} | {', '.join(row['dtypes'])} | no-variant | n/a | | {row['count']:,} | | | | | | | | "
                         f"{', '.join(f'`{m}`' for m in row['modules'][:3])}{' ...' if len(row['modules']) > 3 else ''} |")
            continue
        exact = 'n/a' if row.get('bit_exact') is None else 'exact' if row['bit_exact'] else 'DIFFERS'
        if row.get('padding_differences'):
            exact += f" (padding: {row['padding_differences']:,} words {'+0/-0' if row.get('padding_signed_zero_only') else 'differ'})"
        timed = row.get('timed_output_equal')
        timed = 'not compared' if timed is None else 'exact' if timed else 'DIFFERS'
        cycles = f"{row['model_cycles']:,}" if 'model_cycles' in row else 'not run'
        if row.get('cycles_source', 'full launch') != 'full launch':
            cycles += ' (estimate)'
        total = format(row['total_cycles'], ',') if 'total_cycles' in row else ''
        cosine = '' if row.get('cosine') is None else format(row['cosine'], '.5f')
        relative = '' if row.get('relative_error') is None else format(100 * row['relative_error'], '.3f') + '%'
        lines.append(f"| `{row['family']}` | {row['shape']} | {', '.join(row['dtypes'])} | {row.get('variant') or row['status']} | "
                     f"{(row.get('backend') or {}).get('tier', 'none')} | {row.get('program_words') or ''} | {row['count']:,} | {cycles} | "
                     f"{total} | {row.get('golden') or 'none'} | {exact} | {timed} | {cosine} | {relative} | "
                     f"{', '.join(f'`{m}`' for m in row['modules'][:3])}{' ...' if len(row['modules']) > 3 else ''} |")
    lines += ['', '`words` is the executed program size in 32-bit instruction words (after the arena relocation the bench applies). '
              'Bit-exact covers the logical rows and columns the model consumes; a padding note counts output words in the '
              'zero-padded rows or columns that differ from the golden (`+0/-0`: only the sign of zero differs).']
    unmapped = document['unmapped']
    lines += ['', f"## Unmapped ops ({len(unmapped)} rows, {sum(r['count'] for r in unmapped):,} instances, "
              f"{sum(r['count'] * r['flops_each'] for r in unmapped):,} FLOPs)", '']
    if unmapped:
        lines += ['| phase | module | op | shape | count | FLOPs |', '| --- | --- | --- | --- | ---: | ---: |']
        lines += [f"| {r['phase']} | `{r['module'] or '(top)'}` | `{r['op']}`{(' ' + r['component']) if 'component' in r else ''} | "
                  f"{shape_text(r)} | {r['count']:,} | {r['count'] * r['flops_each']:,} |" for r in unmapped]
    else:
        lines.append('None: every exported op has a placement.')
    if document['uncovered']:
        lines += ['', f"## Atlas rows without a launch ({len(document['uncovered'])})", '',
                  'These rows are placed on Atlas but neither define a launch nor belong to one; add a `launch` hint to their mapping rule.', '',
                  '| phase | module | op | shape | count | where |', '| --- | --- | --- | --- | ---: | --- |']
        lines += [f"| {r['phase']} | `{r['module'] or '(top)'}` | `{r['op']}` | {shape_text(r)} | {r['count']:,} | `{r['where']}` |"
                  for r in document['uncovered']]
    notes = document.get('notes')
    if notes:
        lines += ['', '## Notes']
        for note in notes:
            lines += ['', note]
    return '\n'.join(lines) + '\n'


def failures(rows):
    result = []
    for row in rows:
        if row.get('bit_exact') is False:
            result.append(f"{row['key']}: independent golden mismatch")
        if row.get('timed_output_equal') is False:
            result.append(f"{row['key']}: timed/functional mismatch")
        if row.get('padding_differences') and not row.get('padding_signed_zero_only'):
            result.append(f"{row['key']}: {row['padding_differences']} padding words differ from the golden beyond signed zero")
        if row['status'] == 'error':
            result.append(f"{row['key']}: {row.get('error')}")
    return result


def load_model(coverage, name):
    summary = json.loads(Path(coverage).read_text())
    models = summary['models']
    if name is None:
        if len(models) != 1:
            raise SystemExit(f"--model is required; coverage has {', '.join(models)}")
        name = next(iter(models))
    if name not in models:
        raise SystemExit(f'model {name} is not in {coverage}')
    return name, models[name]


def write(document, args):
    args.out.mkdir(parents=True, exist_ok=True)
    document['compiler_summary'] = dict(tiers=dict(sorted(Counter(
        row['backend']['tier'] for row in document['launches'] if row.get('backend')).items())))
    (args.out / 'MODEL-BENCH.json').write_text(json.dumps(document, indent=1) + '\n')
    text = render(document)
    (args.out / 'MODEL-BENCH.md').write_text(text)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--coverage', type=Path, help='coverage.json from coverage/build_table.py')
    parser.add_argument('--model', help='model name in the coverage (default: its only model)')
    parser.add_argument('--model-root', type=Path, help='npu_model checkout at 47b8599')
    parser.add_argument('--atlas-mlir-bin', type=Path, help='explicit atlas-opt and atlas-emit directory; no environment fallback')
    parser.add_argument('--min-tier', choices=('encode-only', 'stream-verified', 'verified'), default='stream-verified',
                        help='required compiler tier (default: stream-verified when --atlas-mlir-bin is given)')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--markdown', type=Path, help='also write MODEL-BENCH.md here')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--family', action='append', help='bench only these families')
    parser.add_argument('--plan-only', action='store_true', help='derive and report launches without generating or running')
    parser.add_argument('--resume', action='store_true', help='reuse job results already in OUT/jobs')
    parser.add_argument('--job-timeout', type=float, default=7200, help='seconds per functional or timed job')
    parser.add_argument('--subset-after', type=float, default=SLOW_SECONDS,
                        help='seconds after which a row-uniform attention timed run is replaced by a row-subset estimate')
    parser.add_argument('--subset-rows', type=int, default=128)
    parser.add_argument('--max-cycles', type=int, default=2_000_000_000)
    parser.add_argument('--max-instructions', type=int, default=500_000_000)
    parser.add_argument('--command', help='regeneration command recorded in the report (default: this invocation)')
    parser.add_argument('--note', action='append', default=[], help='a line for the Notes section; repeatable')
    parser.add_argument('--report', type=Path, help='re-render OUT/MODEL-BENCH.md from OUT/MODEL-BENCH.json')
    parser.add_argument('--job', help=argparse.SUPPRESS)
    parser.add_argument('--mode', help=argparse.SUPPRESS)
    parser.add_argument('--result', help=argparse.SUPPRESS)
    parser.add_argument('--identity-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.job:
        return job_main(args)
    if args.report:
        document = json.loads((args.report / 'MODEL-BENCH.json').read_text())
        args.out = args.report
        if args.note:
            document['notes'] = args.note
        write(document, args)
        return 1 if failures(document['launches']) else 0
    if args.coverage is None or args.out is None or (args.model_root is None and not args.plan_only):
        parser.error('--coverage and --out are required, and --model-root unless --plan-only')
    args.backend_tools = None
    if args.atlas_mlir_bin and not args.plan_only:
        sys.path.insert(0, str(ROOT / 'lib'))
        import atlas_backend
        args.atlas_mlir_bin = args.atlas_mlir_bin.absolute()
        args.backend_tools = atlas_backend.tools(args.atlas_mlir_bin).identity()
    name, model = load_model(args.coverage, args.model)
    launches, uncovered = plan(model)
    if args.family:
        launches = [entry for entry in launches if entry['family'] in args.family]
    started = time.perf_counter()
    coverage_path = args.coverage.absolute()
    shown = str(coverage_path.relative_to(ROOT)) if coverage_path.is_relative_to(ROOT) else f'{coverage_path.parent.name}/{coverage_path.name}'
    document = dict(schema='atlas.model_bench.v1', model=name, coverage=shown,
                    coverage_sha256=hashlib.sha256(args.coverage.read_bytes()).hexdigest(),
                    coverage_status=model.get('status', 'COMPLETE'), workers=args.workers,
                    command=args.command or 'python ' + ' '.join(shlex.quote(a) for a in [sys.argv[0], *(argv or sys.argv[1:])]),
                    launch_checks=checks(model, launches), unmapped=model.get('unmapped', []), uncovered=uncovered,
                    notes=args.note, backend=(dict(min_tier=args.min_tier, tools=args.backend_tools)
                                              if args.atlas_mlir_bin else None))
    if args.plan_only:
        rows = [dict(entry, shape=describe(entry), status='planned') for entry in launches]
        document.update(npu_model='not run', wall_seconds=0.0, launches=rows)
        write(document, args)
        return 0
    from_root = subprocess.run(['git', '-C', str(args.model_root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    results = execute(launches, args)
    rows = merge(launches, results)
    document.update(npu_model=from_root, wall_seconds=round(time.perf_counter() - started, 1), launches=rows)
    write(document, args)
    problems = failures(rows)
    for problem in problems:
        print(f'FAIL: {problem}', file=sys.stderr)
    print(f"{name}: {len(rows)} launches, {sum(r['status'] == 'ok' for r in rows)} ok, "
          f"{sum(r['status'] == 'no-variant' for r in rows)} no-variant; report {args.out / 'MODEL-BENCH.md'}")
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
