"""Opt-in Tiny fused projections with streamed B DMA and ordered MXU0 rows.

Only exact original family templates are admitted. The producer's BF16 pop,
rescale, ordered hook, and optional RoPE post-pass retain every arithmetic
instruction. B's successor arrives while the first row pair transposes and
consumes the current tile. Later row pairs consume the transposed cache.
Hook scratch, including the full RNE-add register footprint, is dead before
the next core tile restores its operands.
"""
from __future__ import annotations

import re

from atlas_epilogue import Epilogue, Step, TileInput
from atlas_program_checks import check_matrix_registers
from atlas_program_reuse_common import store_output
from atlas_rope_postpass import ResidentRopeEpilogue, append_postpass
from smol_weight_relocate import DEFAULT_ARENA, relocate_assembly

BANK_BYTES = 262144
SCALAR_SCRATCH = (10, 11)
SHAPES = frozenset({(128, 2048, 2048), (128, 5632, 1024)})
LOW_ARENA = 0x90000000


def kinds(m, k, n):
    """Return the explicitly supported fused families at a physical shape."""
    if any(type(v) is not int for v in (m, k, n)) or (m, k, n) not in SHAPES:
        raise ValueError('Tiny fused streaming requires a qualified physical shape')
    if k == 5632:
        return ('add_rne',)
    return ('mul', 'silu', 'add_rne', 'rope')


def layout(m, k, n):
    kinds(m, k, n)
    a_bytes, b_bytes = m * k, k * 32
    # Preserve the established down-projection output scratch. The smaller
    # producer uses the end of its complete resident B panel for scratch.
    out = 188416 if k == 5632 else b_bytes
    a_regions = []
    for offset in range(0, a_bytes, BANK_BYTES):
        a_regions.append([BANK_BYTES + offset,
                          BANK_BYTES + min(a_bytes, offset + BANK_BYTES)])
    return dict(a_bank_ranges=a_regions, b_column_range=[0, b_bytes],
                output_scratch_range=[out, out + 2048],
                dma_load_bytes=a_bytes + k * n, dma_store_bytes=2 * m * n)


def _epilogue(m, k, n, kind, *, output_scale_exponent=0, output_column_tile=0):
    if kind not in kinds(m, k, n):
        raise ValueError('unsupported Tiny fused family at this shape')
    offset = m * k + k * n + 256
    common = dict(m=m, n=n, output_offset=offset, scratch_registers=SCALAR_SCRATCH)
    if kind == 'rope':
        extra = offset + 2 * m * n + 256
        return ResidentRopeEpilogue(**common, cos_offset=extra,
                                    sin_offset=extra + 64 * m,
                                    neg_sin_offset=extra + 128 * m)
    if kind == 'mul':
        # The two captured 2048-wide SwiGLU shards feed the 5632-wide
        # consumer. Their FP8 stores retain its holes and column placement.
        if output_column_tile not in (0, 64):
            raise ValueError('Tiny fused mul requires captured shard column 0 or 64')
        common.update(output_dtype='fp8', output_scale_exponent=output_scale_exponent,
                      output_row_stride=5632 * 32,
                      output_column_tile=output_column_tile)
    extra = offset + Epilogue(**common).output_bytes + 256
    operand = TileInput(extra) if kind in ('mul', 'add_rne') else None
    return Epilogue(**common, steps=(Step(kind, operand),))


def dma_load_bytes(m, k, n, *, kind):
    """Compulsory A/B bytes plus the original family's hook/table bytes."""
    return layout(m, k, n)['dma_load_bytes'] + _epilogue(m, k, n, kind).dma_load_bytes


def _assembly(m, k, n, exponent, epilogue):
    plan = layout(m, k, n)
    out = plan['output_scratch_range'][0] // 4
    offset = epilogue.output_offset
    lines = ['li x5, 4096', 'li x8, 2048', f'li x3, {out}',
             f'li x9, {out + 256}', 'li x30, 1',
             'dma.config x30, ch0', 'dma.config x30, ch1',
             f'vli.all m4, {(exponent + 127) << 7}', 'delay 66']
    # Split A preload at physical banks so no DMA crosses a bank boundary.
    for part, (start, end) in enumerate(plan['a_bank_ranges']):
        label = f'tiny_fused_a_{part}'
        lines += [f'li x19, {start // 4}', f'li x20, {start - BANK_BYTES}',
                  f'li x21, {(end - start) // 4096}', label + ':',
                  'dma.load x19, x20, x5, ch0', 'dma.wait ch0',
                  'addi x19, x19, 1024', 'add x20, x20, x5',
                  'addi x21, x21, -1', f'beq x21, x0, {label}_done',
                  'nop', f'jal x0, {label}', 'nop', label + '_done:']
    lines += ['li x5, 1024', f'li x15, {m * k}', f'li x12, {offset}',
              f'li x17, {n // 32}', f'li x16, {n * 64}',
              'tiny_fused_columns:', 'li x19, 0', 'add x20, x15, x0',
              'dma.load x19, x20, x5, ch0', 'dma.wait ch0',
              'add x22, x12, x0']
    for row in range(0, m // 32, 2):
        rows = min(2, m // 32 - row)
        for tile in range(k // 32):
            if row == 0:
                if tile:
                    lines += ['dma.wait ch1']
                if tile + 1 < k // 32:
                    lines += [f'li x24, {(tile + 1) * 256}',
                              'addi x20, x20, 1024',
                              'dma.load x24, x20, x5, ch1']
                lines += [f'li x6, {tile * 256}', 'vload m1, 0(x6)', 'delay 34',
                          'vtrpose.xlu m1, m1', 'delay 66']
                if m > 64:
                    lines += ['vstore m1, 0(x6)', 'delay 34']
            else:
                lines += [f'li x6, {tile * 256}', 'vload m1, 0(x6)', 'delay 34']
            op = 'vmatmul.mxu0' if tile == 0 else 'vmatmul.acc.mxu0'
            lines += ['vmatpush.weight.mxu0 w0, m1',
                      f'li x4, {(BANK_BYTES + row * k * 32 + tile * 1024) // 4}',
                      'vload m0, 0(x4)', 'delay 34', f'{op} acc0, m0, w0']
            if rows == 2:
                # m6 has no core value live after this compute. Fused hooks
                # may freely overwrite it and any other nonpersistent pair.
                lines += [f'li x7, {(BANK_BYTES + (row + 1) * k * 32 + tile * 1024) // 4}',
                          'vload m6, 0(x7)', 'delay 34', f'{op} acc1, m6, w0']
            lines += ['delay 96' if tile == k // 32 - 1 else 'delay 34']
        for acc in range(rows):
            lines += store_output(acc, 22, exponent, epilogue)
            lines += ['add x22, x22, x16']
    lines += ['addi x15, x20, 1024', 'add x12, x12, x8', 'addi x17, x17, -1',
              'beq x17, x0, tiny_fused_finished', 'nop',
              'jal x0, tiny_fused_columns', 'nop', 'tiny_fused_finished:', 'nop', 'ecall']
    source = '\n'.join(lines) + '\n'
    if isinstance(epilogue, ResidentRopeEpilogue):
        source = append_postpass(source, epilogue)
    check_matrix_registers(source)
    return source


def _admit(source, m, k, n):
    accepted = kinds(m, k, n)
    if not isinstance(source, str):
        raise ValueError('Tiny fused streaming requires assembly text')
    scales = re.findall(r'^vli\.all m4, (\d+)$', source, re.MULTILINE)
    if len(scales) != 1 or int(scales[0]) % 128:
        raise ValueError('Tiny fused streaming requires one power-of-two rescale')
    exponent = (int(scales[0]) >> 7) - 127
    if not -126 <= exponent <= 127:
        raise ValueError('Tiny fused rescale must be a normal finite BF16 power of two')
    relocated = source.startswith(f'li x31, {LOW_ARENA}\n')
    from atlas_program import assembly as original
    candidates = []
    for kind in accepted:
        if kind == 'mul':
            pops = set(re.findall(r'^seli e0, (\d+)$', source, re.MULTILINE))
            if len(pops) != 1:
                continue
            scale = 127 - int(next(iter(pops)))
            if not -126 <= scale <= 119:
                continue
            candidates += [(kind, _epilogue(m, k, n, kind,
                            output_scale_exponent=scale, output_column_tile=column))
                           for column in (0, 64)]
        else:
            candidates.append((kind, _epilogue(m, k, n, kind)))
    for kind, epilogue in candidates:
        expected = original(m, k, n, epilogue.output_offset, exponent,
                            traffic_reuse=True, epilogue=epilogue)
        if kind == 'rope':
            expected = append_postpass(expected, epilogue)
        if relocated:
            expected = relocate_assembly(expected, DEFAULT_ARENA, LOW_ARENA)
        if source == expected:
            return kind, exponent, epilogue, relocated
    raise ValueError('Tiny fused streaming requires an exact original fused family template')


def describe(source, m, k, n):
    """Admit a source and report its canonical family and unchanged DRAM arena."""
    kind, exponent, epilogue, relocated = _admit(source, m, k, n)
    return dict(kind=kind, rescale_exponent=exponent,
                dram_base=LOW_ARENA if relocated else DEFAULT_ARENA,
                output_offset=epilogue.output_offset, output_bytes=epilogue.output_bytes,
                output_dtype=epilogue.output_dtype,
                output_scale_exponent=epilogue.output_scale_exponent,
                output_row_stride=epilogue.output_row_stride,
                output_column_tile=epilogue.output_column_tile,
                dma_load_bytes=dma_load_bytes(m, k, n, kind=kind),
                dma_store_bytes=epilogue.dma_store_bytes)


def optimize(source, m, k, n):
    """Replace an exact original fused program, preserving arithmetic and arena.

    Source comparison includes every immediate, delay, hook, and post-pass.
    The sole accepted relocation is the established 0x90000000 arena. Custom
    hooks, modified programs, plain projections, and repeated transforms fail.
    """
    _, exponent, epilogue, relocated = _admit(source, m, k, n)
    result = _assembly(m, k, n, exponent, epilogue)
    if relocated:
        result = relocate_assembly(result, DEFAULT_ARENA, LOW_ARENA)
    return result
