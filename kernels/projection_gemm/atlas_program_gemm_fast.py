"""Opt-in paired-row GEMM schedule with overlapping operand streams.

Original generators supply the data format, output hooks, and K order. Six
shapes retain their DMA layout; the down projection uses full-A residency.
The paired MXU0 schedule uses m6 for the second row so its load overlaps the
first compute. Select assembly explicitly, or transform a family program.
"""
from __future__ import annotations

import re

from atlas_program import assembly as original_assembly, dma_load_bytes as original_dma_load_bytes
from atlas_program_checks import check_matrix_registers

SHAPES = frozenset({(128, 2048, 2048), (128, 5632, 1024), (128, 768, 768),
                    (128, 3072, 768), (64, 736, 2048), (64, 2048, 736),
                    (128, 768, 3072)})
_COMPUTE = re.compile(r'vmatmul(?:\.acc)?\.mxu0 acc([01]), m0, w0')


def optimize(source, m, k, n):
    """Select residency and pair scheduling for admitted original programs.

    LSU loads retain delay 34. Weight push streams concurrently with the
    following A load, giving at least 36 cycles before its compute. Row 1
    uses m6, leaving m0 readable by row 0. Its load plus delay 34 separates
    the two computes by at least 36 cycles. After row 1, delay 34 plus the
    next B load and delay 34 keeps the next same-slot push beyond 62 cycles.
    The first pop reads acc0 after both row streams and the final delay;
    acc1 is popped only after acc0 has been stored. K order is unchanged.
    """
    if any(type(v) is not int for v in (m, k, n)) or (m, k, n) not in SHAPES:
        raise ValueError('fast GEMM requires an admitted 64- or 128-row shape')
    if (m, k, n) == (128, 5632, 1024) and '\npairs:\n' in source:
        from atlas_program_gemm_fast_breadth import optimize as resident_breadth
        return resident_breadth(source, m, k, n)
    lines = source.splitlines()
    computes = [(i, _COMPUTE.fullmatch(line)) for i, line in enumerate(lines)]
    computes = [(i, match) for i, match in computes if match]
    if not computes:
        raise ValueError('fast GEMM requires the original paired-row core')
    # Hooks may reuse m6 after compute. Require a full restoring write on
    # every admitted straight-line path before any old m6 value is consumed.
    # Reset at CFG edges, so a skipped restore cannot establish liveness.
    restored = False
    for line in lines:
        op = line.split(' ', 1)[0]
        if line.endswith(':') or op in {'beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'jal', 'jalr'} or _COMPUTE.fullmatch(line):
            restored = False
        registers = re.findall(r'\bm\d+\b', line)
        if 'm6' in registers:
            full_write = (registers[0] == 'm6' and 'm6' not in registers[1:]
                          and (op in {'vload', 'vli.all'} or op.startswith('vmatpop.')
                               or op.endswith('.bf16')))
            if not restored and not full_write:
                raise ValueError('original m6 value is live across the fast core')
            restored |= full_write
    def crosses_control_flow(start):
        for line in lines[start:]:
            if line.startswith(('v', 'dma.')):
                return False
            if line.endswith(':') or line.startswith(('beq ', 'jal ')):
                return True
        return True

    edits, counts = {}, [0, 0]
    for i, match in computes:
        row = int(match[1])
        if lines[i + 1] != 'delay 96':
            raise ValueError('unexpected original compute delay')
        counts[row] += 1
        if row == 0:
            edits[i + 1] = None
        else:
            # Only scalar address setup may intervene between the row load
            # and its compute. The pinned recipes have an immediate delay.
            if lines[i - 1] != 'delay 34' or not lines[i - 2].startswith('vload m0, '):
                raise ValueError('unexpected second-row operand load')
            edits[i - 2] = lines[i - 2].replace('vload m0, ', 'vload m6, ')
            edits[i] = lines[i].replace(', m0, ', ', m6, ')
            # The verifier requires fixed-latency work drained at CFG edges.
            edits[i + 1] = 'delay 96' if crosses_control_flow(i + 2) else 'delay 34'
    if counts[0] != counts[1]:
        raise ValueError('fast GEMM requires complete row pairs')
    pushes = 0
    for i, line in enumerate(lines):
        if line == 'vmatpush.weight.mxu0 w0, m1':
            if lines[i + 1] != 'delay 32':
                raise ValueError('unexpected original weight-push delay')
            if not crosses_control_flow(i + 2):
                edits[i + 1] = None
            pushes += 1
    if not pushes:
        raise ValueError('missing original weight pushes')
    result = '\n'.join(edits.get(i, line) for i, line in enumerate(lines)
                       if edits.get(i, line) is not None) + '\n'
    check_matrix_registers(result)
    return result


def assembly(m, k, n, output_offset, rescale_exponent=None, *, epilogue=None, weight_source=None):
    """Generate the opt-in core with the original output-hook interface."""
    if (m, k, n) == (128, 5632, 1024):
        from atlas_program_gemm_fast_breadth import assembly as resident_breadth
        return resident_breadth(m, k, n, output_offset, rescale_exponent,
                               epilogue=epilogue, weight_source=weight_source)
    source = original_assembly(m, k, n, output_offset, rescale_exponent,
                               traffic_reuse=True, epilogue=epilogue, weight_source=weight_source)
    return optimize(source, m, k, n)


def dma_load_bytes(m, k, n, *, epilogue=None):
    """Account for the selected core and its unchanged output-hook loads."""
    if any(type(v) is not int for v in (m, k, n)) or (m, k, n) not in SHAPES:
        raise ValueError('fast GEMM requires an admitted 64- or 128-row shape')
    if (m, k, n) == (128, 5632, 1024):
        from atlas_program_gemm_fast_breadth import dma_load_bytes as resident_bytes
        if epilogue is not None and (epilogue.m, epilogue.n) != (m, n):
            raise ValueError('epilogue geometry differs from contraction')
        return resident_bytes(m, k, n) + (epilogue.dma_load_bytes if epilogue is not None else 0)
    return original_dma_load_bytes(m, k, n, traffic_reuse=True, epilogue=epilogue)
