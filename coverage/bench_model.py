"""Bench every device launch a model's coverage places on Atlas, at the model's real shapes, on npu_model.

Input is the coverage.json that coverage/build_table.py writes (with or without --allow-unmapped). Its atlas_rows
and the mapping rules' launch hints become distinct device launches: projection panels (plain projection_gemm or a
fused family's program), attention launches, and RMSNorm launches, each with its count in the model. For each
distinct launch the script

- generates the program with the family's kernels/<family>/gen.py (a subprocess; a shape the generator rejects is
  reported as no-variant with the generator's error),
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
import json
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
    """The family generator rejected this shape."""


# ----------------------------------------------------------------------------------------------- launch plan

def pad32(value):
    return -(-value // 32) * 32


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


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
    if row['op'] == 'aten.scaled_dot_product_attention.default':
        (_, heads, rows, head_dim), key_shape = row['inputs'][0][0], row['inputs'][1][0]
        keys, kv_heads = key_shape[2], hint.get('kv_heads', key_shape[1])
    else:
        _, heads, rows, keys = out
        kv_heads, head_dim = hint.get('kv_heads', heads), hint.get('head_dim', 64)
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
                    params = dict(rows=product(out[:-1]), m=pad32(product(out[:-1])), width=hint.get('width', out[-1]))
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
        return len(out) >= 2 and product(out[:-1]) == p['rows']
    return True


def describe(entry):
    p, family = entry['params'], entry['family']
    if family == 'attention':
        text = f"{p['rows']}x{p['keys']} {p['heads']}/{p['kv_heads']} d{p['head_dim']} {p['mask']}"
        if p.get('mask_pattern'):
            text += f" ({p['mask_pattern']})"
        return text + (' FP8 out' if p['output_fp8'] else '')
    if family == 'rmsnorm':
        return f"{p['m']}x{p['width']}" + (f" ({p['rows']} rows)" if p['rows'] != p['m'] else '')
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
    """Run kernels/<family>/gen.py; return its source and record, or raise NoVariant with its error."""
    out = Path(tempfile.mkdtemp(prefix=f'{family}-', dir=work))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(model_root), ATLAS_NPU_MODEL_ROOT=str(model_root))
    command = [sys.executable, '-B', str(ROOT / 'kernels' / family / 'gen.py'), '--out', str(out), '--name', 'program', *arguments]
    process = subprocess.run(command, capture_output=True, text=True, env=environment)
    text = f"kernels/{family}/gen.py {' '.join(shlex.quote(a) for a in arguments)}"
    if process.returncode:
        lines = [line for line in process.stderr.strip().splitlines() if line.strip()]
        raise NoVariant(f"{text}: {lines[-1] if lines else f'exit {process.returncode}'}")
    record = json.loads((out / 'program.json').read_text())
    return (out / 'program.S').read_text(), dict(record, command=text)


def synthetic_projection(p, rng):
    """FP32 A [M, K] ~ N(0, 1) and W [K, N] ~ N(0, 0.02^2) with zero padding outside the logical extent."""
    import numpy as np
    import kernel_bench as bench
    a = np.zeros((p['m'], p['k']), dtype=np.float32)
    w = np.zeros((p['k'], p['n']), dtype=np.float32)
    a[:p['rows'], :p['logical_k']] = rng.standard_normal((p['rows'], p['logical_k']), dtype=np.float32)
    w[:p['logical_k'], :p['logical_n']] = (0.02 * rng.standard_normal((p['logical_k'], p['logical_n']))).astype(np.float32)
    return a, w, bench.quantize(a), bench.quantize(w)


def projection_arena(p, a_codes, w_codes, extent, extras=b''):
    import kernel_bench as bench
    return bench.pack_fp8_tiles(a_codes) + bench.pack_fp8_weights(w_codes) + b'\xa5' * 256 + b'\xcd' * extent + b'\x5a' * 256 + extras


def recipe(m, k, n):
    import atlas_program
    try:
        return atlas_program._reuse_module(m, k, n).removeprefix('atlas_program_reuse_')
    except ValueError:
        return None


def build_projection_gemm(p, rng, gen):
    import kernel_bench as bench
    from atlas_vector_arithmetic import ordered_matmul, rescale_bf16_array
    reference = bench.load_reference('projection_gemm')
    m, k, n = p['m'], p['k'], p['n']
    a, w, (a_codes, a_scale), (w_codes, w_scale) = synthetic_projection(p, rng)
    shape, scale = f'{m}x{k}x{n}', str(a_scale + w_scale)
    try:
        source, record = gen('projection_gemm', ['--shape', shape, '--rescale-exponent', scale, '--traffic-reuse'])
        variant = f"{record['recipe'].removeprefix('atlas_program_reuse_')} recipe"
    except NoVariant as reuse:
        try:
            source, record = gen('projection_gemm', ['--shape', shape, '--rescale-exponent', scale])
        except NoVariant as default:
            raise NoVariant(f'{reuse}; {default}') from None
        variant = 'default schedule (no traffic-reuse recipe for this shape)'
    return Built(bench.relocate(source), projection_arena(p, a_codes, w_codes, 2 * m * n), m * k + k * n + 256, 2 * m * n,
                 reference.reference(a, w), lambda raw: bench.bf16_values(bench.unpack_bf16_tiles(raw, m, n)),
                 lambda: bench.pack_bf16_tiles(rescale_bf16_array(ordered_matmul(a_codes, w_codes), a_scale + w_scale)),
                 variant, record, dict(a_scale=a_scale, w_scale=w_scale))


def build_epilogue(p, rng, gen):
    import numpy as np
    import torch
    import kernel_bench as bench
    reference = bench.load_reference('epilogue')
    m, k, n = p['m'], p['k'], p['n']
    a, w, (a_codes, a_scale), (w_codes, w_scale) = synthetic_projection(p, rng)
    scale = a_scale + w_scale
    steps = p['steps']
    if steps == ['silu'] and p['output'] == 'bf16':
        source, record = gen('epilogue', ['--shape', f'{m}x{k}x{n}', '--steps', 'silu', '--rescale-exponent', str(scale)])
        return Built(bench.relocate(source), projection_arena(p, a_codes, w_codes, record['output_bytes']), record['output_offset'],
                     record['output_bytes'], reference.reference(a, w),
                     lambda raw: bench.bf16_values(bench.unpack_bf16_tiles(raw, m, n)), None,
                     f"{record['recipe'].removeprefix('atlas_program_reuse_')} recipe + SiLU epilogue", record,
                     dict(a_scale=a_scale, w_scale=w_scale))
    if steps == ['mul'] and p['output'] == 'fp8':
        gate_words = np.zeros((m, n), dtype=np.uint16)
        gate_words[:p['rows'], :p['logical_n']] = bench.bf16_words(torch.nn.functional.silu(
            torch.from_numpy(rng.standard_normal((p['rows'], p['logical_n']), dtype=np.float32))).numpy())
        gate = bench.bf16_values(gate_words)
        expected = reference.reference(a, w, gate)
        output_scale = bench.fp8_scale(expected)
        stride = 32 * p.get('consumer_width', n)
        source, record = gen('epilogue', ['--shape', f'{m}x{k}x{n}', '--steps', 'mul', '--fp8', '--rescale-exponent', str(scale),
                                          '--output-scale-exponent', str(output_scale), '--output-row-stride', str(stride)])
        if record['operand_offset'] != record['output_offset'] + record['output_bytes'] + 256:
            raise ValueError('epilogue operand offset differs from the staged layout')

        def decode(raw):
            blocks = [bench.unpack_fp8_tiles(raw[r * stride:r * stride + n * 32], 32, n) for r in range(m // 32)]
            return bench.fp8_values(np.concatenate(blocks), output_scale)
        return Built(bench.relocate(source), projection_arena(p, a_codes, w_codes, record['output_bytes'], bench.pack_bf16_tiles(gate_words)),
                     record['output_offset'], record['output_bytes'], expected, decode, None,
                     f"{record['recipe'].removeprefix('atlas_program_reuse_')} recipe + multiply epilogue, FP8 out", record,
                     dict(a_scale=a_scale, w_scale=w_scale, output_scale=output_scale))
    raise NoVariant(f'no bench staging for epilogue steps {steps} with {p["output"]} output')


def build_rne_add(p, rng, gen):
    import numpy as np
    import kernel_bench as bench
    from atlas_vector_arithmetic import decode_bf16, ordered_matmul, rescale_bf16_array, round_bf16
    reference = bench.load_reference('rne_add')
    m, k, n = p['m'], p['k'], p['n']
    a, w, (a_codes, a_scale), (w_codes, w_scale) = synthetic_projection(p, rng)
    residual_words = np.zeros((m, n), dtype=np.uint16)
    residual_words[:p['rows'], :p['logical_n']] = bench.bf16_words(rng.standard_normal((p['rows'], p['logical_n']), dtype=np.float32))
    source, record = gen('epilogue', ['--shape', f'{m}x{k}x{n}', '--steps', 'add_rne', '--rescale-exponent', str(a_scale + w_scale)])
    if record['operand_offset'] != record['output_offset'] + 2 * m * n + 256:
        raise ValueError('residual offset differs from the staged layout')

    def golden():
        projected = rescale_bf16_array(ordered_matmul(a_codes, w_codes), a_scale + w_scale)
        return bench.pack_bf16_tiles(round_bf16(decode_bf16(projected) + decode_bf16(residual_words)))
    return Built(bench.relocate(source), projection_arena(p, a_codes, w_codes, 2 * m * n, bench.pack_bf16_tiles(residual_words)),
                 record['output_offset'], 2 * m * n, reference.reference(a, w, bench.bf16_values(residual_words)),
                 lambda raw: bench.bf16_values(bench.unpack_bf16_tiles(raw, m, n)), golden,
                 f"{record['recipe'].removeprefix('atlas_program_reuse_')} recipe + add_rne epilogue "
                 '(kernels/epilogue/gen.py --steps add_rne; kernels/rne_add/gen.py emits only the add sequence)', record,
                 dict(a_scale=a_scale, w_scale=w_scale))


def build_rope(p, rng, gen):
    import numpy as np
    import kernel_bench as bench
    from atlas_vector_arithmetic import decode_bf16, ordered_matmul, rescale_bf16_array, round_bf16
    reference = bench.load_reference('rope')
    m, k, n = p['m'], p['k'], p['n']
    a, w, (a_codes, a_scale), (w_codes, w_scale) = synthetic_projection(p, rng)
    source, record = gen('rope', ['--shape', f'{m}x{k}x{n}', '--rescale-exponent', str(a_scale + w_scale)])
    cos_words, sin_words = reference.tables(m)
    blob = b''.join(bench.pack_bf16_tiles(value[:m, :32]) for value in (cos_words, sin_words, sin_words ^ 0x8000))
    if record['table_offset'] != record['output_offset'] + 2 * m * n + 256:
        raise ValueError('RoPE table offset differs from the staged layout')
    heads = n // 64

    def golden():
        x = decode_bf16(rescale_bf16_array(ordered_matmul(a_codes, w_codes), a_scale + w_scale))
        cosine, sine = (np.tile(decode_bf16(words), heads) for words in (cos_words, sin_words))
        return bench.pack_bf16_tiles(round_bf16(decode_bf16(round_bf16(x * cosine)) + decode_bf16(round_bf16(reference.rotate_half(x) * sine))))
    return Built(bench.relocate(source), projection_arena(p, a_codes, w_codes, 2 * m * n, blob), record['output_offset'], 2 * m * n,
                 reference.reference(a, w, bench.bf16_values(cos_words), bench.bf16_values(sin_words)),
                 lambda raw: bench.bf16_values(bench.unpack_bf16_tiles(raw, m, n)), golden,
                 f"{record['recipe'].removeprefix('atlas_program_reuse_')} recipe + RoPE post-pass (positions 0..{m - 1})", record,
                 dict(a_scale=a_scale, w_scale=w_scale))


def build_rmsnorm(p, rng, gen):
    import numpy as np
    import kernel_bench as bench
    from atlas_rmsnorm_postpass import MASK_BYTES
    reference = bench.load_reference('rmsnorm')
    m, width = p['m'], p['width']
    h_words = np.zeros((m, width), dtype=np.uint16)
    h_words[:p['rows']] = bench.bf16_words(rng.standard_normal((p['rows'], width), dtype=np.float32))
    gamma_words = bench.bf16_words(1 + 0.1 * rng.standard_normal(width, dtype=np.float32))
    expected = reference.reference(bench.bf16_values(h_words), bench.bf16_values(gamma_words))
    scale = bench.fp8_scale(expected)
    source, record = gen('rmsnorm', ['--rows', str(m), '--width', str(width), '--norm-scale-exponent', str(scale)])
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
                 'standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out)', record, dict(output_scale=scale))


def attention_mask(p):
    import numpy as np
    rows, keys = p['rows'], p['keys']
    if p['mask'] == 'causal':
        return np.tril(np.ones((rows, keys), dtype=bool), keys - rows)
    mask = np.ones((rows, keys), dtype=bool)
    if p.get('mask_pattern') == 'prefix_state_block':
        mask[:keys - 1, keys - 1] = False  # the SmolVLA prefix state token starts a new attention block
    elif p.get('mask_pattern') not in (None, 'all_visible'):
        raise NoVariant(f"no synthetic mask pattern {p['mask_pattern']}")
    return mask


def build_attention(p, rng, gen, work):
    import numpy as np
    import kernel_bench as bench
    from atlas_attention_v2_program import AttentionV2Config, stage_v2, unpack_output_v2
    reference = bench.load_reference('attention')
    if p['head_dim'] != 64:
        raise NoVariant(f"attention v2 requires head dimension 64; this launch has {p['head_dim']}")
    rows, keys, heads, kv_heads = p['rows'], p['keys'], p['heads'], p['kv_heads']
    words = [bench.bf16_words(rng.standard_normal(shape, dtype=np.float32))
             for shape in ((heads, rows, 64), (kv_heads, keys, 64), (kv_heads, keys, 64))]
    q, k, v = (bench.bf16_values(x) for x in words)
    mask = attention_mask(p)
    expected = reference.reference(q, k, v, mask)
    exponents = dict(q_exponent=bench.fp8_scale(q), k_exponent=bench.fp8_scale(k), v_exponent=bench.fp8_scale(v))
    o_exponent = bench.fp8_scale(expected) if p['output_fp8'] else 0
    arguments = ['--rows', str(rows), '--keys', str(keys), '--heads', f'{heads}/{kv_heads}', '--mask', p['mask'],
                 '--q-exponent', str(exponents['q_exponent']), '--k-exponent', str(exponents['k_exponent']),
                 '--v-exponent', str(exponents['v_exponent']), '--o-exponent', str(o_exponent)]
    if p['mask'] == 'causal':
        arguments += ['--query-start', str(p['query_start'])]
    if p['mask'] == 'arbitrary':
        path = Path(tempfile.mkdtemp(prefix='mask-', dir=work)) / 'mask.npy'
        np.save(path, mask)
        arguments += ['--mask-npy', str(path)]
    if p['output_fp8']:
        arguments.append('--fp8')
    source, record = gen('attention', arguments)
    config = AttentionV2Config(rows, keys, heads, kv_heads, mask=p['mask'], query_start=p.get('query_start'),
                               output_fp8=p['output_fp8'], o_exponent=o_exponent, **exponents)
    arena, offsets = stage_v2(config, *words, mask_tiles=reference.mask_tiles(mask) if p['mask'] == 'arbitrary' else None)
    if offsets != record['offsets']:
        raise ValueError('attention offsets differ between gen.py and the staged synthetic arena')

    def decode(raw):
        value = unpack_output_v2(config, raw)
        return bench.fp8_values(value, o_exponent) if p['output_fp8'] else bench.bf16_values(value)
    variant = 'attention v2' + (' packed decode' if config.decode_packed else '') + f", {offsets['unique_masks']} unique mask tiles"
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
    a, w, (a_codes, a_scale), (w_codes, w_scale) = synthetic_projection(p, rng)
    bias_words = np.zeros(n, dtype=np.uint16)
    bias_words[:p['logical_n']] = bench.bf16_words(0.1 * rng.standard_normal(p['logical_n'], dtype=np.float32))
    expected = reference.reference(a, w, bench.bf16_values(bias_words))
    output_scale = bench.fp8_scale(expected)
    source, record = gen('smol_fusion', ['--shape', f'{m}x{k}x{n}', '--fp8', '--module', p['module'],
                                         '--rescale-exponent', str(a_scale + w_scale), '--output-scale-exponent', str(output_scale)])
    offset = m * k + k * n + 256
    row = dict(index=0, call=0, module=p['module'], physical_mkn=[m, k, n], logical_mkn=[m, k, n], logical_row_start=0,
               output_offset=offset, dram_bytes=offset + 2 * m * n + 256, recipe='traffic-reuse-v1',
               program_policy='traffic-reuse-v1', output_rescale_exponent=a_scale + w_scale, dram_base=0x90000000,
               requested_dma=dict(load_bytes=dma_load_bytes(m, k, n, traffic_reuse=True), store_bytes=2 * m * n))
    program = stage_fc1_fp8(row, projection_arena(p, a_codes, w_codes, 2 * m * n), bias_bf16=bias_words.tobytes(),
                            output_scale_exponent=output_scale)
    if program.source != source:
        raise ValueError('smol_fusion program differs between gen.py and the staged synthetic arena')
    return Built(source, program.arena, offset, m * n, expected,
                 lambda raw: bench.fp8_values(bench.unpack_fp8_tiles(raw, m, n), output_scale), None,
                 f"{recipe(m, k, n)} recipe + fc1 bias and GELU (tanh), FP8 out (smol_fusion_fp8.stage_fc1_fp8)", record,
                 dict(a_scale=a_scale, w_scale=w_scale, output_scale=output_scale))


BUILDERS = {'projection_gemm': build_projection_gemm, 'epilogue': build_epilogue, 'rne_add': build_rne_add,
            'rope': build_rope, 'rmsnorm': build_rmsnorm, 'attention': build_attention, 'smol_fusion': build_smol_fusion}


def build(entry, work, model_root):
    import numpy as np
    family = entry['family']
    if family not in BUILDERS:
        raise NoVariant(f'no model-shape bench builder for family {family}')
    rng = np.random.default_rng(seed(entry['key']))
    gen = lambda fam, arguments: generator(fam, arguments, work, model_root)  # noqa: E731
    if family == 'attention':
        return build_attention(entry['params'], rng, gen, work)
    built = BUILDERS[family](entry['params'], rng, gen)
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


def run_job(entry, mode, model_root, work, max_cycles, max_instructions):
    """One functional or timed execution of one launch; returns a result dict (never raises for NoVariant)."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    sys.path.insert(0, str(model_root))
    sys.path.insert(0, str(ROOT))
    import atlas_kernels_path  # noqa: F401
    import torch
    torch.set_num_threads(1)
    import kernel_bench as bench
    result = dict(key=entry['key'], mode=mode)
    started = time.perf_counter()
    try:
        built = build(entry, work, model_root)
    except NoVariant as error:
        return dict(result, status='no-variant', error=str(error))
    result.update(variant=built.variant, generator_words=built.generator.get('words'), generator_command=built.generator.get('command'),
                  source_sha256=hashlib.sha256(built.source.encode()).hexdigest(), meta=built.meta)
    try:
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
    result = run_job(entry, args.mode, args.model_root, Path(args.job).parent, args.max_cycles, args.max_instructions)
    Path(args.result).write_text(json.dumps(result, indent=1) + '\n')
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

    def one(entry, mode, timeout):
        name = hashlib.sha256((entry['key'] + mode).encode()).hexdigest()[:16]
        job, out = work / f'{name}.job.json', work / f'{name}.{mode}.json'
        if args.resume and out.exists():
            return entry, mode, json.loads(out.read_text())
        job.write_text(json.dumps(entry) + '\n')
        command = [sys.executable, '-B', str(Path(__file__).absolute()), '--job', str(job), '--mode', mode,
                   '--result', str(out), '--model-root', str(args.model_root), '--max-cycles', str(args.max_cycles),
                   '--max-instructions', str(args.max_instructions)]
        started = time.perf_counter()
        environment = dict(os.environ, OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=environment)
        except subprocess.TimeoutExpired:
            result = dict(key=entry['key'], mode=mode, status='timeout', seconds=round(time.perf_counter() - started, 1))
            out.write_text(json.dumps(result, indent=1) + '\n')
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
        if 'no-variant' in statuses:
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
        errors = [f"{mode}: {r.get('error') or r.get('status')}" for mode, r in (('functional', functional), ('timed', timed))
                  if r.get('status') not in ('ok', 'timeout')]
        if timed.get('status') == 'ok':
            row.update(model_cycles=timed['model_cycles'], cycles_source='full launch',
                       timed_output_equal=timed.get('output_sha256') == functional.get('output_sha256') if functional.get('status') == 'ok' else None)
        elif timed.get('status') == 'timeout':
            row['timed_note'] = f"timed run exceeded {timed['seconds']:.0f} s"
            done = (part or {}).get('result', {})
            if done.get('status') == 'ok':
                row.update(model_cycles=done['model_cycles'] * part['factor'], timed_output_equal=None,
                           cycles_source=f"estimate: {part['rows']}-row subset ({done['model_cycles']:,} cycles) x {part['factor']}",
                           subset_seconds=done.get('seconds'))
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

def gflop(value):
    return f'{value / 1e9:,.3f}'


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
    lines = [f"# {model} kernel bench at model shapes", '',
             f"Regenerate: `{document['command']}`", '',
             f"Input: `{document['coverage']}` (SHA-256 `{document['coverage_sha256']}`), coverage status {document['coverage_status']}. "
             f"Every distinct device launch the coverage places on Atlas ran on npu_model `{document['npu_model']}` with numerics=\"rtl\" "
             f"on seeded synthetic inputs at the model's shape ({document['workers']} worker processes; {job_seconds:,.0f} job-seconds in all, "
             f"longest job {longest:,.0f} s).", '',
             'Model cycles are npu_model Core cycle-model counts, not RTL cycles. Cycles x count multiplies one launch by the number of '
             'launches in the model; the sums below are device-only estimates that ignore host time, launch overhead, and data '
             'movement between launches. Bit-exact compares output bytes with the independent lib/golden reference where the family '
             'has one; cosine and rel-err compare the decoded output with an FP32 PyTorch reference on the unquantized inputs, so they '
             'include FP8 input quantization. Inputs are synthetic, so fidelity and data-dependent cycles are representative, not the model\'s.', '']
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
    caveat = (f" {sum(r['count'] for r in estimated):,} of those launches use a row-subset estimate ("
              + ', '.join(f"`{r['family']}` {r['shape']}" for r in estimated) + ').') if estimated else ''
    lines += ['## Summary', '', f"Device cycle estimate (sum of model cycles x count over {sum(r['count'] for r in ok):,} of "
              f"{sum(r['count'] for r in rows):,} launches): {total:,} npu_model model cycles. It ignores host time and launch overhead.{caveat}", '',
              '| family | launches | launches with cycles | model cycles x count | share |', '| --- | ---: | ---: | ---: | ---: |']
    for family, (count, cycles, timed) in sorted(by_family.items(), key=lambda item: -item[1][1]):
        lines.append(f"| `{family}` | {count:,} | {timed:,} | {cycles:,} | {100 * cycles / total if total else 0:.2f}% |")
    lines += ['', 'Top shapes by total cycles:', '', '| family | shape | count | model cycles | cycles x count |', '| --- | --- | ---: | ---: | ---: |']
    for row in sorted(ok, key=lambda r: -r['total_cycles'])[:10]:
        mark = ' (estimate)' if row.get('cycles_source', 'full launch') != 'full launch' else ''
        lines.append(f"| `{row['family']}` | {row['shape']} | {row['count']:,} | {row['model_cycles']:,}{mark} | {row['total_cycles']:,}{mark} |")
    unsupported = [row for row in rows if row['status'] != 'ok']
    lines += ['', f"Unsupported or failed launches: {len(unsupported)}."]
    for row in unsupported:
        lines.append(f"- `{row['family']}` {row['shape']} x{row['count']:,}: {row['status']}: {row.get('error', '')}")
    estimates = [row for row in rows if row.get('cycles_source', 'full launch') != 'full launch' or row.get('timed_note')]
    slow = [row for row in rows if max(row.get('functional_seconds') or 0, row.get('timed_seconds') or 0) > SLOW_SECONDS]
    if estimates or slow:
        lines += ['', 'Shapes that took more than 20 minutes on npu_model:']
        for row in sorted({id(r): r for r in estimates + slow}.values(), key=lambda r: r['key']):
            lines.append(f"- `{row['family']}` {row['shape']}: functional {row.get('functional_seconds')} s, timed "
                         f"{row.get('timed_seconds')} s; {row.get('timed_note', 'completed')}; cycles: {row.get('cycles_source', 'none')}")
    lines.append('')
    lines += ['## Launches', '', '| family | shape | export dtype | variant | words | count | model cycles | cycles x count | golden | '
              'bit-exact | timed equals functional | cosine | rel-err | modules |',
              '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | :-: | :-: | ---: | ---: | --- |']
    for row in rows:
        if row['status'] == 'no-variant':
            lines.append(f"| `{row['family']}` | {row['shape']} | {', '.join(row['dtypes'])} | no-variant | | {row['count']:,} | | | | | | | | "
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
                     f"{row.get('program_words') or ''} | {row['count']:,} | {cycles} | "
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
        lines += ['', '## Notes', '', *notes]
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
                    notes=args.note)
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
