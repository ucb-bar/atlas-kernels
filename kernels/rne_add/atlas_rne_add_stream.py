"""Opt-in standalone exact BF16 addition of two tiled tensors.

Inputs and output use the library's BF16 tile-pair layout. The launch streams
one pair of tiles at a time, so its VMEM footprint is independent of shape.
This adds a kernel option; no model placement or default generator changes.
"""

from atlas_rne_add import CONSTANTS, rne_add


def flat_geometry(elements):
    """Return a compact tile shape for a contiguous BF16 elementwise stream.

    No layout conversion is needed: both inputs and the output use the same
    lane permutation. Only the final chunk is padded to 1024 BF16 words.
    """
    if type(elements) is not int or elements <= 0:
        raise ValueError('element count must be a positive integer')
    padded = ((elements + 1023) // 1024) * 1024
    return 32, padded // 32


def assembly(rows, width, *, dram_base=0x90000000):
    """Add physical rows x width tensors at base and base + 2*rows*width.

    Output starts at base + 4*rows*width. Dimensions must be positive multiples
    of 32. The caller pads logical rows and columns before tiling.
    """
    if any(type(d) is not int or d <= 0 or d % 32 for d in (rows, width)):
        raise ValueError('physical dimensions must be positive multiples of 32')
    size = 2 * rows * width
    if (type(dram_base) is not int or dram_base < 0 or dram_base % 32
            or dram_base + 3 * size >= 2**32):
        raise ValueError('aligned arena must fit the DMA low32 address window')
    lines = ['li x30, 0', 'dma.config x30, ch0',
             'li x1, 0', 'li x2, 512', 'li x3, 256', 'li x4, 768',
             f'li x5, {dram_base}', f'li x6, {dram_base + size}',
             f'li x7, {dram_base + 2 * size}', 'li x8, 2048',
             f'li x10, {rows * width // 1024}']
    lines += [line for register, bits in CONSTANTS
              for line in (f'vli.all m{register}, {bits}', 'delay 66')]
    lines += ['rne_stream_tile:',
              'dma.load x1, x5, x8, ch0', 'dma.wait ch0',
              'dma.load x2, x6, x8, ch0', 'dma.wait ch0',
              'vload m0, 0(x1)', 'delay 34', 'vload m1, 0(x3)', 'delay 34',
              'vload m2, 0(x2)', 'delay 34', 'vload m3, 0(x4)', 'delay 34']
    lines += rne_add(0, 2, 52, constants=False)
    lines += ['vstore m52, 0(x1)', 'delay 34', 'vstore m53, 0(x3)', 'delay 34',
              'dma.store x7, x1, x8, ch0', 'dma.wait ch0',
              'add x5, x5, x8', 'add x6, x6, x8', 'add x7, x7, x8',
              'addi x10, x10, -1', 'bne x10, x0, rne_stream_tile', 'nop',
              'ecall']
    return '\n'.join(lines) + '\n'
