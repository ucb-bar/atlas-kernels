#!/usr/bin/env python3
"""Qualify isolated synthetic Qwen projection fusions, with optional Core timing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from atlas_execution import Executor  # noqa: E402
import atlas_backend  # noqa: E402
import atlas_program_qwen_fused as fusion  # noqa: E402
from atlas_path_guard import checked  # noqa: E402
from atlas_rne_add_stream import assembly as stream_add  # noqa: E402
from atlas_vector_arithmetic import ordered_matmul, rescale_bf16_array  # noqa: E402
from kernel_bench import (bf16_words, pack_bf16_tiles, quantize, timed_cycles)  # noqa: E402


def exact_sum(left, right):
    """Independent host FP32 addition and BF16 RNE, including canonical NaN.

    BF16 operands are embedded exactly in FP32. This oracle imports no device
    arithmetic or generated exact-add instruction implementation.
    """
    a, b = np.asarray(left, dtype=np.uint16), np.asarray(right, dtype=np.uint16)
    with np.errstate(over='ignore', invalid='ignore'):
        summed = ((a.astype(np.uint32) << 16).view(np.float32)
                  + (b.astype(np.uint32) << 16).view(np.float32))
    raw = summed.view(np.uint32)
    rounded = ((raw + np.uint32(0x7fff) + ((raw >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(summed), np.uint16(0x7fc0), rounded).astype(np.uint16)


def expected_arena(arena, offset, output):
    return arena[:offset] + output + arena[offset + len(output):]


def qualify_case(role, logical_rows, model_root, atlas_bin, out, *, timed=False):
    """One complete real physical panel, full arena guards, and split baseline."""
    name = f'{"qkv-bias" if role == "q" else role + "-residual"}-rows{logical_rows}'
    target = checked(out / name)
    target.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(130 + logical_rows + (role == 'down'))
    m, k, n = 32, 8960 if role == 'down' else 1536, 128
    a = np.zeros((m, k), dtype=np.float32)
    a[:logical_rows] = rng.standard_normal((logical_rows, k), dtype=np.float32)
    w = (rng.standard_normal((k, n), dtype=np.float32) * np.float32(0.02))
    a_codes, a_scale = quantize(a)
    w_codes, w_scale = quantize(w)
    scale = a_scale + w_scale
    # Nonzero captured-style scale is exercised independently of add rounding.
    assert scale != 0
    shape = (n,) if role == 'q' else (m, n)
    addend = bf16_words(rng.standard_normal(shape, dtype=np.float32))
    # A broad finite exponent range, signed zero, and subnormal input words.
    addend.flat[:12] = [0, 0x8000, 1, 0x8001, 0x3b80, 0xbb80,
                       0x3f80, 0xbf80, 0x3f81, 0xbf81, 0x5f00, 0xdf00]
    source, arena, plan = fusion.stage(role, a_codes, w_codes, addend, scale)
    plain = fusion.assembly(role, m, k, n, scale, fused=False)
    projected = rescale_bf16_array(ordered_matmul(a_codes, w_codes), scale)
    addend_panel = np.broadcast_to(addend, (m, n)) if role == 'q' else addend
    summed_words = exact_sum(projected, addend_panel)
    golden = pack_bf16_tiles(summed_words)
    host_sum = ((projected.astype(np.uint32) << 16).view(np.float32)
                + (addend_panel.astype(np.uint32) << 16).view(np.float32))
    truncating = (host_sum.view(np.uint32) >> 16).astype(np.uint16)
    rne_distinguishing = int(np.count_nonzero(truncating != summed_words))
    if not rne_distinguishing:
        raise AssertionError('fixture does not distinguish exact RNE from truncation')
    projected_bytes = pack_bf16_tiles(projected)
    operand_bytes = pack_bf16_tiles(addend_panel)
    size = plan['output_bytes']
    split_arena = projected_bytes + operand_bytes + b'\xcd' * size + b'\xa7' * 256
    split_source = stream_add(m, n)
    wanted = {
        'plain': (plain, arena, plan['output_offset'], projected_bytes, plan['plain_dma']),
        'split_add': (split_source, split_arena, 2 * size, golden,
                      dict(load_bytes=2 * size, store_bytes=size)),
        'fused': (source, arena, plan['output_offset'], golden, plan['fused_dma']),
    }
    executor = Executor(model_root)
    receipts = {}
    for label, (text, image, offset, output, dma) in wanted.items():
        program, _ = executor.assemble(text)
        compiled = atlas_backend.cross_checked(text, program.assemble(), bin_dir=atlas_bin,
                                                min_tier='verified', name=f'{name}-{label}.S')
        execution = executor.execute(text, image, offset, size,
                                     expected_arena=expected_arena(image, offset, output), requested_dma=dma)
        record = dict(compiler=compiled.record(), words_equal_assembler=True,
                      functional=execution.metadata(), full_arena_exact=True,
                      input_sha256=hashlib.sha256(image).hexdigest(),
                      output_sha256=hashlib.sha256(output).hexdigest())
        (target / f'{label}.S').write_text(text)
        (target / f'{label}.bin').write_bytes(compiled.binary())
        if timed:
            executor._bookend()
            cycles, actual = timed_cycles(model_root, text, image, 20_000_000)
            if actual != execution.arena:
                raise AssertionError(f'{name}/{label}: timed full arena differs')
            executor._bookend()
            record.update(model_cycles=cycles, timed_full_arena_exact=True)
        receipts[label] = record
        (target / f'{label}.json').write_text(json.dumps(record, indent=2) + '\n')
        print(f'{name}/{label}: exact, compiler verified'
              + (f', {record["model_cycles"]} cycles' if timed else ''), flush=True)
    row = dict(name=name, roles=['q', 'k', 'v'] if role == 'q' else [role],
               logical_rows=logical_rows, physical_shape=[m, k, n],
               a_scale=a_scale, weight_scale=w_scale, rescale_exponent=scale,
               seed=130 + logical_rows + (role == 'down'),
               rne_distinguishing_physical_words=rne_distinguishing,
               plan=plan, independent_golden='ordered FP8 BF16 MACs + rescale + host FP32/BF16 RNE sum',
               full_arena_exact=True, compiler_tier='verified', words_equal_assembler=True,
               receipts={label: str(target / f'{label}.json') for label in receipts})
    if timed:
        cycles = {label: receipt['model_cycles'] for label, receipt in receipts.items()}
        row.update(cycles=cycles, split_total_cycles=cycles['plain'] + cycles['split_add'],
                   fused_increment_cycles=cycles['fused'] - cycles['plain'],
                   fused_saved_vs_split_cycles=cycles['plain'] + cycles['split_add'] - cycles['fused'])
    (target / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--atlas-bin', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timed', action='store_true')
    parser.add_argument('--case', choices=['q', 'o', 'down'], action='append')
    parser.add_argument('--logical-rows', choices=[1, 32], type=int, action='append')
    args = parser.parse_args()
    model, atlas_bin, out = map(checked, (args.model_root, args.atlas_bin, args.out))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = dict(schema='atlas-kernels.qwen-projection-fusions.v1',
                   scope='isolated seeded synthetic panels; no connected graph, host timing, or model accuracy claim',
                   decode='32 physical rows are processed, including padding; split_add also processes the full panel',
                   compact_decode='a 128-element compact add is a separate alternative; extraction and transport are unmeasured',
                   timing='npu_model Core cycles, numerics=rtl' if args.timed else 'PENDING',
                   tools=atlas_backend.tools(atlas_bin).identity(), cases=rows)
    for role in args.case or ('q', 'o', 'down'):
        for logical_rows in args.logical_rows or (1, 32):
            rows.append(qualify_case(role, logical_rows, model, atlas_bin, out, timed=args.timed))
            (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(cases=len(rows), full_arena_exact=True, words_equal_assembler=True,
                          summary=str(out / 'summary.json'))), flush=True)


if __name__ == '__main__':
    main()
