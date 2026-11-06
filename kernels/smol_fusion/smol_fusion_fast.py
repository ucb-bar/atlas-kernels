"""Opt-in streamed vision fc1 producer with the original bias/GELU/FP8 hook.

Only the complete original 128x768x3072 template is admitted. A remains in
bank 1; the first row pair transposes B while its successor arrives, and the
second pair reads that cache. Ordered MXU0 products and every hook instruction
are preserved, including the captured FP8 consumer scale and compact arena.
"""
from __future__ import annotations

import re

from atlas_epilogue import Epilogue, Step, TileInput
from atlas_program_checks import check_matrix_registers
from atlas_program_reuse_common import store_output
from smol_weight_relocate import DEFAULT_ARENA, relocate_assembly

BANK_BYTES = 262144
LOW_ARENA = 0x90000000
SHAPES = frozenset({(128, 768, 3072)})
SCALAR_SCRATCH = (10, 11)


def layout(m, k, n):
    if any(type(v) is not int for v in (m, k, n)) or (m, k, n) not in SHAPES:
        raise ValueError('streamed Smol fusion requires the qualified physical shape')
    return dict(b_column_range=[0, k * 32],
                output_scratch_range=[k * 32, k * 32 + 2048],
                a_bank_range=[BANK_BYTES, BANK_BYTES + m * k],
                dma_load_bytes=m * k + k * n + 2 * m * n,
                dma_store_bytes=m * n)


def _epilogue(m, k, n, output_scale_exponent):
    layout(m, k, n)
    offset = m * k + k * n + 256
    return Epilogue(m, n, offset,
                    (Step('add', TileInput(offset + m * n + 256, 'channel')),
                     Step('gelu_tanh')),
                    output_dtype='fp8', output_scale_exponent=output_scale_exponent,
                    scratch_registers=SCALAR_SCRATCH)


def _assembly(m, k, n, exponent, epilogue):
    plan = layout(m, k, n)
    out = plan['output_scratch_range'][0] // 4
    lines = ['li x5, 4096', 'li x8, 2048', f'li x3, {out}',
             f'li x9, {out + 256}', 'li x30, 1',
             'dma.config x30, ch0', 'dma.config x30, ch1',
             f'vli.all m4, {(exponent + 127) << 7}', 'delay 66',
             f'li x19, {BANK_BYTES // 4}', 'li x20, 0',
             f'li x21, {m * k // 4096}', 'smol_fused_a:',
             'dma.load x19, x20, x5, ch0', 'dma.wait ch0',
             'addi x19, x19, 1024', 'add x20, x20, x5',
             'addi x21, x21, -1', 'beq x21, x0, smol_fused_a_done',
             'nop', 'jal x0, smol_fused_a', 'nop', 'smol_fused_a_done:',
             'li x5, 1024', f'li x15, {m * k}',
             f'li x23, {epilogue.output_offset}', f'li x17, {n // 32}',
             f'li x16, {n * 64}', 'smol_fused_columns:',
             'li x19, 0', 'add x20, x15, x0',
             'dma.load x19, x20, x5, ch0', 'dma.wait ch0',
             'add x22, x23, x0']
    for row in range(0, m // 32, 2):
        for tile in range(k // 32):
            if row == 0:
                if tile:
                    lines += ['dma.wait ch1']
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
                      f'li x4, {(BANK_BYTES + row * k * 32 + tile * 1024) // 4}',
                      'vload m0, 0(x4)', 'delay 34', f'{op} acc0, m0, w0',
                      f'li x7, {(BANK_BYTES + (row + 1) * k * 32 + tile * 1024) // 4}',
                      'vload m6, 0(x7)', 'delay 34', f'{op} acc1, m6, w0',
                      'delay 96' if tile == k // 32 - 1 else 'delay 34']
        for acc in (0, 1):
            lines += store_output(acc, 22, exponent, epilogue)
            lines += ['add x22, x22, x16']
    lines += ['addi x15, x20, 1024', 'add x23, x23, x8',
              'addi x17, x17, -1', 'beq x17, x0, smol_fused_finished', 'nop',
              'jal x0, smol_fused_columns', 'nop', 'smol_fused_finished:', 'nop', 'ecall']
    result = '\n'.join(lines) + '\n'
    check_matrix_registers(result)
    return result


def _admit(source, m, k, n):
    layout(m, k, n)
    if not isinstance(source, str):
        raise ValueError('streamed Smol fusion requires assembly text')
    scales = re.findall(r'^vli\.all m4, (\d+)$', source, re.MULTILINE)
    pops = set(re.findall(r'^seli e0, (\d+)$', source, re.MULTILINE))
    if len(scales) != 1 or int(scales[0]) % 128 or len(pops) != 1:
        raise ValueError('streamed Smol fusion requires exact rescale and FP8 scale literals')
    exponent = (int(scales[0]) >> 7) - 127
    if not -126 <= exponent <= 127:
        raise ValueError('Smol fusion rescale must be a normal finite BF16 power of two')
    epilogue = _epilogue(m, k, n, 127 - int(next(iter(pops))))
    from atlas_program import assembly as original
    expected = original(m, k, n, epilogue.output_offset, exponent,
                        traffic_reuse=True, epilogue=epilogue)
    relocated = source.startswith(f'li x31, {LOW_ARENA}\n')
    if relocated:
        expected = relocate_assembly(expected, DEFAULT_ARENA, LOW_ARENA)
    if source != expected:
        raise ValueError('streamed Smol fusion requires the exact original FP8 fc1 template')
    return exponent, epilogue, relocated


def describe(source, m, k, n):
    """Admit the complete original program and describe its preserved arena."""
    exponent, epilogue, relocated = _admit(source, m, k, n)
    return dict(kind='bias-gelu-fp8', rescale_exponent=exponent,
                output_scale_exponent=epilogue.output_scale_exponent,
                output_offset=epilogue.output_offset, output_bytes=epilogue.output_bytes,
                output_dtype='fp8', output_layout='tiles',
                dram_base=LOW_ARENA if relocated else DEFAULT_ARENA,
                extra_offset=epilogue.input_regions[0]['offset'],
                extra_bytes=epilogue.input_regions[0]['size_bytes'],
                dma_load_bytes=layout(m, k, n)['dma_load_bytes'],
                dma_store_bytes=epilogue.dma_store_bytes)


def optimize(source, m, k, n):
    """Transform an exact original FP8 fc1 source; reject edits and repeats."""
    exponent, epilogue, relocated = _admit(source, m, k, n)
    result = _assembly(m, k, n, exponent, epilogue)
    return relocate_assembly(result, DEFAULT_ARENA, LOW_ARENA) if relocated else result


def stage_fc1_fp8_fast(row, initial, *, bias_bf16, output_scale_exponent):
    """Stage the original immutable arena and replace only its admitted source."""
    from smol_fusion_fp8 import stage_fc1_fp8
    from smol_fusion_program import FusionProgram, _sha
    program = stage_fc1_fp8(row, initial, bias_bf16=bias_bf16,
                             output_scale_exponent=output_scale_exponent)
    source = optimize(program.source, *row['physical_mkn'])
    metadata = dict(program.metadata, variant='streamed-fc1-fp8-v1',
                    source_sha256=_sha(source.encode()))
    return FusionProgram(source, program.arena, metadata)
