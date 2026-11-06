"""Opt-in attention schedule with completed DMA transfers and a guarded halt.

The default attention generators and kits are unchanged. This conservative
variant sacrifices DMA overlap so the compiler timing verifier can prove all
VMEM accesses independent of pending transfers, including dynamically addressed
head and token loops. A scalar NOP makes the terminal DELAY retire before ECALL.
"""
import re


def synchronous_dma(source):
    tail = 'dma.wait ch0\ndma.wait ch1\ndelay 128\necall\n'
    if not source.endswith(tail) or source.splitlines().count('ecall') != 1:
        raise ValueError('attention v2 halt tail changed; re-qualify synchronous DMA')
    lines = []
    for line in source.splitlines():
        if line.strip() == 'ecall':
            lines.append('nop')
        lines.append(line)
        match = re.fullmatch(r'\s*dma\.(?:load|store)\s+.*?,\s*(ch[01])\s*', line)
        if line.strip().startswith(('dma.load ', 'dma.store ')) and match is None:
            raise ValueError('attention DMA transfer syntax changed; re-qualify synchronous DMA')
        if match:
            lines.append('dma.wait ' + match.group(1))
    return '\n'.join(lines) + '\n'
