"""Opt-in Qwen projection panels with exact BF16 bias or residual addition.

The existing ordered producer is retained, including resident A for K1536.
Only q/k/v, o, and down roles are admitted. Logits and other shapes require a
separate contract even when their physical dimensions happen to coincide.
"""
from dataclasses import dataclass, replace

import numpy as np

from atlas_epilogue import Epilogue, Step, TileInput
from atlas_program import assembly as projection
from atlas_program import dma_load_bytes
from smol_weight_relocate import DEFAULT_ARENA, relocate_assembly

DRAM_BASE = 0x90000000
ROLES = ('q', 'k', 'v', 'o', 'down')


@dataclass(frozen=True)
class ChannelRNE(Epilogue):
    """Channel broadcast using the existing exact-add instructions and guard.

    Validate the complete base contract through a tiled surrogate, then retain
    the channel operand for inherited address generation. Keeping add_rne in
    steps ensures checked_assembly checks the actual hook's MRF footprint.
    """

    def __post_init__(self):
        if (len(self.steps) != 1 or self.steps[0].op != 'add_rne'
                or not isinstance(self.steps[0].operand, TileInput)
                or self.steps[0].operand.layout != 'channel'):
            raise ValueError('channel RNE requires one exact BF16 channel add')
        # Validate channel extent as well as the stricter tile surrogate.
        Epilogue(self.m, self.n, self.output_offset,
                 (Step('add', self.steps[0].operand),),
                 self.output_dtype, self.output_scale_exponent, self.output_layout,
                 self.scratch_registers, self.output_row_stride, self.output_column_tile)
        Epilogue(self.m, self.n, self.output_offset,
                 (Step('add_rne', replace(self.steps[0].operand, layout='tiles')),),
                 self.output_dtype, self.output_scale_exponent, self.output_layout,
                 self.scratch_registers, self.output_row_stride, self.output_column_tile)


def layout(role, m, k, n, *, dram_base=DRAM_BASE):
    """Return the guarded arena and selected producer's exact DMA accounting."""
    if role not in ROLES:
        raise ValueError('Qwen fusion requires q/k/v/o/down role')
    expected = (32, 8960 if role == 'down' else 1536, 128)
    if any(type(d) is not int for d in (m, k, n)) or (m, k, n) != expected:
        raise ValueError('Qwen fusion requires its complete physical panel shape')
    output = m * k + k * n + 256
    size = 2 * m * n
    operand = output + size + 256
    end = operand + size + 256
    if (type(dram_base) is not int or dram_base < 0 or dram_base % 32
            or dram_base + end >= 2**32):
        raise ValueError('Qwen fusion arena must fit the aligned DMA low32 window')
    resident = k == 1536
    loads = dma_load_bytes(m, k, n, resident_a=resident)
    return dict(role=role, shape=[m, k, n], output_offset=output,
                output_bytes=size, operand_offset=operand, operand_bytes=size,
                arena_bytes=end, operand_layout='channel' if role in ('q', 'k', 'v') else 'tiles',
                resident_a=resident, dram_base=dram_base,
                plain_dma=dict(load_bytes=loads, store_bytes=size),
                fused_dma=dict(load_bytes=loads + size, store_bytes=size))


def assembly(role, m, k, n, rescale_exponent, *, dram_base=DRAM_BASE, fused=True):
    """Emit the selected plain producer or its exact-add fused counterpart."""
    plan = layout(role, m, k, n, dram_base=dram_base)
    if type(rescale_exponent) is not int or not -126 <= rescale_exponent <= 127:
        raise ValueError('Qwen projection requires one normal finite integer rescale')
    if type(fused) is not bool:
        raise ValueError('fused must be a boolean')
    hook = None
    if fused:
        cls = ChannelRNE if plan['operand_layout'] == 'channel' else Epilogue
        hook = cls(m, n, plan['output_offset'],
                   (Step('add_rne', TileInput(plan['operand_offset'], plan['operand_layout'])),),
                   scratch_registers=(27, 28))
    source = projection(m, k, n, plan['output_offset'], rescale_exponent,
                        resident_a=plan['resident_a'], epilogue=hook)
    return relocate_assembly(source, DEFAULT_ARENA, dram_base)


def stage(role, a_codes, weight_codes, addend_words, rescale_exponent, *, dram_base=DRAM_BASE):
    """Stage physical FP8 matrices and a BF16 channel vector or residual panel.

    BF16 channel vectors are expanded to one broadcast tile per output column.
    Callers zero activation padding explicitly; the exact hook processes every
    physical row, including padding. Residual input therefore has all 32 rows.
    """
    from kernel_bench import pack_bf16_tiles, pack_fp8_tiles, pack_fp8_weights
    a, w, addend = map(np.asarray, (a_codes, weight_codes, addend_words))
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[0]:
        raise ValueError('Qwen operands require compatible physical matrices')
    m, k, n = a.shape[0], a.shape[1], w.shape[1]
    plan = layout(role, m, k, n, dram_base=dram_base)
    if a.dtype != np.uint8 or w.dtype != np.uint8 or addend.dtype != np.uint16:
        raise ValueError('Qwen staging requires FP8 uint8 and BF16 uint16 words')
    if np.any((a & 127) == 127) or np.any((w & 127) == 127):
        raise ValueError('NaN FP8 is outside the ordered projection contract')
    expected = (n,) if plan['operand_layout'] == 'channel' else (m, n)
    if addend.shape != expected:
        raise ValueError('Qwen addend shape differs from its explicit role layout')
    if plan['operand_layout'] == 'channel':
        addend = np.broadcast_to(addend, (32, n))
    arena = (pack_fp8_tiles(a) + pack_fp8_weights(w) + b'\xa5' * 256
             + b'\xcd' * plan['output_bytes'] + b'\x5a' * 256
             + pack_bf16_tiles(addend) + b'\xa7' * 256)
    assert len(arena) == plan['arena_bytes']
    return assembly(role, m, k, n, rescale_exponent, dram_base=dram_base), arena, plan
