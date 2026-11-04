"""Opt-in standalone rotate-half RoPE over finite BF16 tiled tensors.

The coefficient contract is BF16 cos, sin, and the exact sign-bit inverse of
sin, with one 32x32 tile per physical row tile shared by all head64 groups.
TinyLlama's BF16-table arithmetic is exact under this contract. Rounding an
FP32 coefficient table first changes SmolVLA's host arithmetic, so that use
requires its own whole-action accuracy decision and does not change defaults.
"""
from dataclasses import asdict
import struct

from atlas_rope_postpass import ResidentRopeEpilogue, postpass
from atlas_resident_output import RESIDENT_H_WORD
from atlas_program_checks import check_matrix_registers
from smol_weight_relocate import DEFAULT_ARENA, relocate_assembly

ARENA_BASE = 0x90000000


def _spec(m, n):
    if any(type(v) is not int or v <= 0 for v in (m, n)) or m % 32 or n % 64:
        raise ValueError('physical M must be a positive multiple of 32 and N a positive multiple of 64')
    size = 2 * m * n
    output = size + 256
    return ResidentRopeEpilogue(m, n, output, cos_offset=output + size + 256,
        sin_offset=output + size + 256 + 64*m, neg_sin_offset=output + size + 256 + 128*m)


def layout(m, n, *, dram_base=ARENA_BASE):
    """Describe H, output, tables, guards, VMEM, and exact DMA extents.

    M and N are physical dimensions. Logical tail rows are validated by
    prepare(); callers must not treat M as an unpadded logical row count.
    """
    e = _spec(m, n)
    extent = e.neg_sin_offset + 64*m + 256
    if (type(dram_base) is not int or dram_base < 0 or dram_base % 64
            or dram_base + extent > 2**32):
        raise ValueError('aligned physical arena must fit the low32 DMA window')
    return dict(m=m, n=n, input_offset=0, input_bytes=2*m*n,
        output_offset=e.output_offset, output_bytes=e.output_bytes,
        cos_offset=e.cos_offset, sin_offset=e.sin_offset, neg_sin_offset=e.neg_sin_offset,
        coefficient_bytes=192*m, arena_bytes=extent, dram_base=dram_base,
        dma_load_bytes=2*m*n+192*m, dma_store_bytes=2*m*n,
        input_layout='bf16_tiles', output_layout='bf16_tiles', head_dimension=64,
        coefficient_layout='one_bf16_32x32_tile_per_row_tile_shared_by_heads',
        h_vmem_word=RESIDENT_H_WORD, coefficient_vmem_byte=1048576,
        output_stage_vmem_byte=1310720, epilogue=asdict(e))


def assembly(m, n, *, dram_base=ARENA_BASE):
    """Emit an independent launch with H DMA followed by the stock post-pass."""
    s = layout(m, n, dram_base=dram_base)
    e = _spec(m, n)
    lines = ['li x30, 1', 'dma.config x30, ch0', f'li x15, {RESIDENT_H_WORD}',
        'li x16, 0', 'li x17, 4096', f"li x18, {s['input_bytes']//4096}", 'rope_h_load:',
        'dma.load x15, x16, x17, ch0', 'dma.wait ch0', 'addi x15, x15, 1024',
        'li x19, 4096', 'add x16, x16, x19', 'addi x18, x18, -1',
        'bne x18, x0, rope_h_load', 'nop']
    # The stock pass waits for every output DMA. No extra terminal delay.
    source = relocate_assembly('\n'.join(lines + postpass(e) + ['ecall']) + '\n',
                               DEFAULT_ARENA, dram_base)
    check_matrix_registers(source)
    return source


def _words(raw, extent, name):
    if not isinstance(raw, (bytes, bytearray, memoryview)) or len(bytes(raw)) != extent:
        raise ValueError(name + ' has the wrong packed BF16 byte extent')
    words = tuple(x[0] for x in struct.iter_unpack('<H', bytes(raw)))
    if any((value & 0x7f80) == 0x7f80 for value in words):
        raise ValueError(name + ' must be finite BF16')
    return words


def _at(row, col, width):
    return ((row//32)*(width//32)+col//32)*1024 + (col%32//16)*512 + row%32*16 + col%16


def prepare(m, n, h, cosine, sine, negative_sine, *, logical_rows=None, dram_base=ARENA_BASE):
    """Stage exact little-endian BF16 tiles, guards, and output poison.

    H is MxN; each coefficient is Mx32 in the same BF16 tile-pair layout.
    Tail H rows must be signed zero, with cosine 1 and signed-zero sine.
    negative_sine must equal sine XOR 0x8000 on every word, including zeros.
    This function does not silently cast FP32 coefficients or select positions.
    """
    s = layout(m, n, dram_base=dram_base)
    rows = m if logical_rows is None else logical_rows
    if type(rows) is not int or not 1 <= rows <= m:
        raise ValueError('logical_rows must be an integer in 1..M')
    hw = _words(h, 2*m*n, 'H')
    cw, sw, nw = (_words(value, 64*m, name) for value, name in
                  ((cosine, 'cosine'), (sine, 'sine'), (negative_sine, 'negative sine')))
    if any((value & 0x7fff) > 0x3f80 for value in (*cw, *sw)):
        raise ValueError('cosine and sine must lie in [-1,1]')
    if any(neg != (value ^ 0x8000) for value, neg in zip(sw, nw)):
        raise ValueError('negative sine must be the exact BF16 sign-bit inverse of sine')
    for row in range(rows, m):
        if any(hw[_at(row, col, n)] & 0x7fff for col in range(n)):
            raise ValueError('padded H rows must be signed zero')
        if any(cw[_at(row, col, 32)] != 0x3f80 or sw[_at(row, col, 32)] & 0x7fff for col in range(32)):
            raise ValueError('padded coefficient rows require cosine 1 and signed-zero sine')
    initial = bytes(h) + b'\xa5'*256 + b'\xcd'*s['output_bytes'] + b'\x5a'*256
    initial += bytes(cosine) + bytes(sine) + bytes(negative_sine) + b'\xa5'*256
    if len(initial) != s['arena_bytes']:
        raise AssertionError('standalone rope staging extent differs')
    return dict(s, logical_rows=rows, source=assembly(m, n, dram_base=dram_base), initial=initial,
                written_spans=[dict(offset=s['output_offset'], bytes=s['output_bytes'])], output_dtype='bf16')
