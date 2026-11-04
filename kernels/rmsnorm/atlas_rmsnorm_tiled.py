"""Standalone RMSNorm with one 32-row BF16 tile row resident at a time.

The pinned resident post-passes remain unchanged. This generator uses the same
DRAM arena layout, but streams H tile rows and stages FP8 at the end of VMEM.
"""
from atlas_epilogue import Epilogue
from atlas_norm_postpass import _bits, _load_pair
from atlas_program_checks import check_matrix_registers
from atlas_rmsnorm_postpass import H_WORD, EXTRA_WORD, MASK_BYTES

VMEM_BYTES = 1536 * 1024
STAGE_WORD = (VMEM_BYTES - 2048) // 4
MAX_WIDTH = (4 * STAGE_WORD - 4 * EXTRA_WORD - MASK_BYTES) // 64


def _dma_load(target, offset, size, label):
    """Contiguous DMA in legal <=4096-byte, 32-byte-aligned transfers."""
    count, tail = divmod(size, 4096)
    lines = [f'li x15, {target}', f'li x16, {offset}']
    if count:
        lines += ['li x17, 4096', f'li x18, {count}', label + ':',
                  'dma.load x15, x16, x17, ch0', 'dma.wait ch0',
                  'addi x15, x15, 1024', 'li x19, 4096', 'add x16, x16, x19',
                  'addi x18, x18, -1', f'bne x18, x0, {label}', 'nop']
    if tail:
        lines += [f'li x17, {tail}', 'dma.load x15, x16, x17, ch0', 'dma.wait ch0']
    return lines


def standalone_source(rows, width, mask_offset, weight_offset, scale=-8):
    """Return unrelocated assembly; H starts at 0 and FP8 at 2*rows*width.

    Physical dimensions are positive multiples of 32. Width is at most 7136
    for the current gamma/masks/staging VMEM layout, not an ISA width limit.
    All output rows are physical; logical padding is the caller's contract.
    """
    if type(rows) is not int or rows <= 0 or rows % 32:
        raise ValueError('RMSNorm rows must be a positive multiple of 32')
    if type(width) is not int or width <= 0 or width % 32 or width > MAX_WIDTH:
        raise ValueError(f'RMSNorm width must be a positive multiple of 32 <= {MAX_WIDTH} (current VMEM layout)')
    output_offset, output_bytes = 2 * rows * width, rows * width
    bridge = Epilogue(rows, width, output_offset, output_dtype='fp8',
                      output_scale_exponent=scale, scratch_registers=(10, 11))
    # Also reject overlapping extras and low32 overflow before emitting text.
    regions = [(0, output_offset + output_bytes), (mask_offset, MASK_BYTES), (weight_offset, 64 * width)]
    for i, (offset, size) in enumerate(regions):
        if type(offset) is not int or offset < 0 or offset % 32 or offset + size >= 2**32:
            raise ValueError('RMSNorm arena regions must be aligned and fit low32')
        if any(offset < p + n and p < offset + size for p, n in regions[:i]):
            raise ValueError('RMSNorm arena regions overlap')
    lines = ['li x30, 1', 'dma.config x30, ch0']
    lines += _dma_load(EXTRA_WORD, mask_offset, MASK_BYTES, 'norm_masks_load')
    lines += _dma_load(EXTRA_WORD + MASK_BYTES // 4, weight_offset, 64 * width, 'norm_gamma_load')
    lines += [f'li x13, {rows // 32}', 'li x22, 0', f'li x20, {output_offset}', 'norm_row:']
    # Reuse the bulk/tail transfer sequence with an externally maintained DRAM pointer.
    load = _dma_load(H_WORD, 0, 64 * width, 'norm_h_load')
    load[1] = 'add x16, x22, x0'
    lines += load
    lines += [f'li x19, {64 * width}', 'add x22, x22, x19',
              'vli.all m10, 0', 'delay 66', f'li x16, {H_WORD}']
    # Each mask gives a distinct column to a tile sum. Reduce at most 32 tiles,
    # then accumulate each group's replicated row sum in m10/m11.
    remaining = width // 32
    group = 0
    while remaining:
        count = min(32, remaining)
        lines += ['vli.all m12, 0', 'delay 66', f'li x17, {EXTRA_WORD}', f'li x18, {count}',
                  f'norm_stats_{group}:', *_load_pair(2, 16), 'vmul.bf16 m6, m2, m2', 'delay 66',
                  'vredsum.row.bf16 m6, m6', 'delay 39', *_load_pair(8, 17),
                  'vmul.bf16 m6, m6, m8', 'delay 66', 'vadd.bf16 m12, m12, m6', 'delay 66',
                  'addi x16, x16, 512', 'addi x17, x17, 512', 'addi x18, x18, -1',
                  f'bne x18, x0, norm_stats_{group}', 'nop',
                  'vredsum.row.bf16 m12, m12', 'delay 39', 'vadd.bf16 m10, m10, m12', 'delay 66']
        remaining -= count
        group += 1
    lines += [f'vli.all m6, {_bits(1 / width)}', 'delay 66', 'vmul.bf16 m10, m10, m6', 'delay 66',
              f'vli.all m6, {_bits(1e-5)}', 'delay 66', 'vadd.bf16 m10, m10, m6', 'delay 66',
              'vsqrt.bf16 m10, m10', 'delay 66', 'vrecip.bf16 m10, m10', 'delay 66',
              f'li x16, {H_WORD}', f'li x17, {EXTRA_WORD + MASK_BYTES // 4}', f'li x18, {width // 32}',
              f'li x3, {STAGE_WORD}', f'li x9, {STAGE_WORD + 256}', 'norm_output:', *_load_pair(2, 16),
              'vmul.bf16 m2, m2, m10', 'delay 66', *_load_pair(6, 17),
              'vmul.bf16 m2, m2, m6', 'delay 66']
    lines += bridge.store(20)
    lines += ['addi x16, x16, 512', 'addi x17, x17, 512', 'li x19, 2048', 'add x20, x20, x19',
              'addi x18, x18, -1', 'bne x18, x0, norm_output', 'nop',
              'addi x13, x13, -1', 'bne x13, x0, norm_row', 'nop', 'ecall']
    source = '\n'.join(lines) + '\n'
    check_matrix_registers(source)
    return source
