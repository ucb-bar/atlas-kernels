"""Opt-in SmolVLA projection with B DMA overlapped by the first row pair.

A remains resident for the whole contraction. The first pair transposes and
caches each B tile while its successor arrives through the other DMA channel.
The remaining pair consumes that cache. Arithmetic and increasing-K order are
unchanged, and every control-flow edge drains fixed-latency work.
"""
from __future__ import annotations

import re

from atlas_program_checks import checked_assembly, check_matrix_registers
from atlas_program_reuse_common import store_output

BANK_BYTES = 262144
SCALAR_SCRATCH = (10, 11)
SHAPES = frozenset({(128, 768, 768), (128, 3072, 768),
                    (64, 736, 2048), (64, 2048, 736),
                    (64, 736, 960), (64, 960, 736),
                    (128, 960, 2560), (64, 960, 2560)})


def layout(m, k, n):
    if any(type(v) is not int for v in (m, k, n)) or (m, k, n) not in SHAPES:
        raise ValueError('streamed Smol projection requires a qualified physical shape')
    b_bytes = k * 32
    scratch_end = b_bytes + 2048
    spill = max(0, m * k - BANK_BYTES)
    if scratch_end + spill > BANK_BYTES:
        raise ValueError('resident A and B column exceed two VMEM banks')
    return dict(b_column_range=[0, b_bytes],
                output_scratch_range=[b_bytes, scratch_end],
                a_bank1_range=[BANK_BYTES, BANK_BYTES + min(m * k, BANK_BYTES)],
                a_bank0_range=[scratch_end, scratch_end + spill],
                dma_load_bytes=m * k + k * n, dma_store_bytes=2 * m * n)


def dma_load_bytes(m, k, n):
    return layout(m, k, n)['dma_load_bytes']


@checked_assembly(scratch_registers=SCALAR_SCRATCH)
def assembly(m, k, n, output_offset, rescale_exponent=0):
    """Emit the plain projection in its original packed, guarded arena."""
    plan = layout(m, k, n)
    if type(output_offset) is not int or output_offset != m * k + k * n + 256:
        raise ValueError('projection output must retain its original guarded offset')
    if type(rescale_exponent) is not int or not -126 <= rescale_exponent <= 127:
        raise ValueError('projection requires one normal finite BF16 rescale')
    scratch_end = plan['output_scratch_range'][1]

    def a_word(offset):
        return (BANK_BYTES + offset if offset < BANK_BYTES
                else scratch_end + offset - BANK_BYTES) // 4

    out_word = k * 8
    lines = ['li x5, 1024', 'li x8, 2048', f'li x3, {out_word}',
             f'li x9, {out_word + 256}', 'li x30, 1',
             'dma.config x30, ch0', 'dma.config x30, ch1',
             f'vli.all m4, {(rescale_exponent + 127) << 7}', 'delay 66']
    for part, (offset, size) in enumerate(((0, min(m * k, BANK_BYTES)),
                                         (BANK_BYTES, max(0, m * k - BANK_BYTES)))):
        if not size:
            continue
        tag = f'smol_a_{part}'
        lines += [f'li x19, {a_word(offset)}', f'li x20, {offset}',
                  f'li x21, {size // 1024}', tag + ':',
                  'dma.load x19, x20, x5, ch0', 'dma.wait ch0',
                  'addi x19, x19, 256', 'addi x20, x20, 1024',
                  'addi x21, x21, -1', f'beq x21, x0, {tag}_done',
                  'nop', f'jal x0, {tag}', 'nop', tag + '_done:']
    lines += [f'li x15, {m * k}', f'li x12, {output_offset}',
              f'li x17, {n // 32}', f'li x16, {n * 64}',
              'smol_columns:', 'li x19, 0', 'add x20, x15, x0',
              'dma.load x19, x20, x5, ch0', 'dma.wait ch0',
              'add x22, x12, x0']
    for row in range(0, m // 32, 2):
        for tile in range(k // 32):
            if row == 0:
                if tile:
                    lines += ['dma.wait ch1']
                # Address setup is unrolled so the next raw B tile always
                # targets a distinct cache tile. No final speculative load.
                if tile + 1 < k // 32:
                    lines += [f'li x24, {(tile + 1) * 256}',
                              'addi x20, x20, 1024',
                              'dma.load x24, x20, x5, ch1']
                lines += [f'li x6, {tile * 256}', 'vload m1, 0(x6)', 'delay 34',
                          'vtrpose.xlu m1, m1', 'delay 66',
                          'vstore m1, 0(x6)', 'delay 34']
            else:
                lines += [f'li x6, {tile * 256}', 'vload m1, 0(x6)', 'delay 34']
            op = 'vmatmul.mxu0' if tile == 0 else 'vmatmul.acc.mxu0'
            lines += ['vmatpush.weight.mxu0 w0, m1',
                      f'li x4, {a_word(row * k * 32 + tile * 1024)}',
                      'vload m0, 0(x4)', 'delay 34', f'{op} acc0, m0, w0',
                      f'li x7, {a_word((row + 1) * k * 32 + tile * 1024)}',
                      'vload m6, 0(x7)', 'delay 34', f'{op} acc1, m6, w0',
                      'delay 96' if tile == k // 32 - 1 else 'delay 34']
        for acc in (0, 1):
            lines += store_output(acc, 22, rescale_exponent, None)
            lines += ['add x22, x22, x16']
    lines += ['addi x15, x20, 1024', 'add x12, x12, x8', 'addi x17, x17, -1',
              'beq x17, x0, smol_finished', 'nop', 'jal x0, smol_columns', 'nop',
              'smol_finished:', 'nop', 'ecall']
    source = '\n'.join(lines) + '\n'
    check_matrix_registers(source)
    return source


def optimize(source, m, k, n):
    """Replace only an exact original plain program, preserving its arena.

    Accept the original traffic-reuse source, or its established low-word
    relocation. Full-template comparison rejects hooks, changed addresses,
    post-passes, and already transformed programs rather than guessing liveness.
    """
    layout(m, k, n)
    scales = re.findall(r'^vli\.all m4, (\d+)$', source, re.MULTILINE)
    if len(scales) != 1 or int(scales[0]) % 128:
        raise ValueError('projection requires exactly one power-of-two rescale')
    exponent = (int(scales[0]) >> 7) - 127
    from atlas_program import assembly as original
    offset = m * k + k * n + 256
    expected = original(m, k, n, offset, exponent, traffic_reuse=True)
    relocated = source.startswith('li x31, 2415919104\n')
    if relocated:
        from smol_weight_relocate import relocate_assembly, DEFAULT_ARENA
        expected = relocate_assembly(expected, DEFAULT_ARENA, 2415919104)
    if source != expected:
        raise ValueError('streamed projection requires an exact original plain program')
    result = assembly(m, k, n, offset, exponent)
    if relocated:
        result = relocate_assembly(result, DEFAULT_ARENA, 2415919104)
    return result
