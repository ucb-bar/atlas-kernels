"""Explicit attention v2 variant without the dead terminal DELAY.

The original generator remains byte-identical for existing SHA-pinned receipts.
Only the terminal pad is removed; both DMA channel waits remain in place.
"""

from atlas_attention_v2_program import assembly_v2 as _original_assembly_v2
from atlas_attention_v2_program import check_program_v2


def assembly_v2(config, offsets, *, dram_base=0x90000000):
    """Generate v2 with a drained halt, removing exactly one dead instruction."""
    source = _original_assembly_v2(config, offsets, dram_base=dram_base)
    tail = "dma.wait ch0\ndma.wait ch1\ndelay 128\necall\n"
    if not source.endswith(tail):
        raise ValueError("attention v2 halt tail changed; re-qualify the no-delay variant")
    return check_program_v2(source[:-len(tail)] + "dma.wait ch0\ndma.wait ch1\necall\n")
