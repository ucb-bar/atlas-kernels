"""Opt-in full-A residency for the captured 128/5632/1024 contraction.

A occupies physical VMEM banks 1 through 3. Bank 0 holds one transposed B
column and the unchanged breadth output scratch. Each A and B byte is loaded
once; every accumulator follows increasing K and is popped exactly once.
"""
from __future__ import annotations

import re

from atlas_program_checks import checked_assembly, check_matrix_registers
from atlas_program_reuse_common import load_b_column, store_output

SHAPE = (128, 5632, 1024)
SCALAR_SCRATCH = (10, 11)
BANK_BYTES = 262144
A_BYTES = 720896
B_COLUMN_BYTES = 180224
OUTPUT_BYTE = 188416


def layout(m, k, n):
    if any(type(value) is not int for value in (m, k, n)) or (m, k, n) != SHAPE:
        raise ValueError('full-A breadth requires physical shape 128/5632/1024')
    return dict(a_range=[BANK_BYTES, BANK_BYTES + A_BYTES],
                a_bank_ranges=[[262144, 524288], [524288, 786432], [786432, 983040]],
                b_column_range=[0, B_COLUMN_BYTES],
                output_scratch_range=[OUTPUT_BYTE, OUTPUT_BYTE + 2048],
                dma_load_bytes=A_BYTES + k * n, dma_store_bytes=2 * m * n)


def dma_load_bytes(m, k, n):
    return layout(m, k, n)['dma_load_bytes']


def _core(prefix, output_pair, b_low, *, relocated=False, channels=(0, 1), fast=True):
    """Emit conservative arithmetic, then apply the shared proven pair timing."""
    m, k, n = SHAPE
    lines = list(prefix)
    for bank, count in ((1, 64), (2, 64), (3, 48)):
        tag = f'full_a_bank_{bank}'
        lines += [f'li x19, {bank * BANK_BYTES // 4}', f'li x20, {(bank - 1) * BANK_BYTES}',
                  f'li x21, {count}', tag + ':', 'dma.load x19, x20, x5, ch0', 'dma.wait ch0',
                  'addi x19, x19, 1024', 'add x20, x20, x5', 'addi x21, x21, -1',
                  f'beq x21, x0, {tag}_done', 'nop', f'jal x0, {tag}', 'nop', tag + '_done:']
    lines += ['li x5, 1024', f'li x15, {b_low}', 'full_columns:', 'li x19, 0',
              'add x20, x15, x0', 'li x21, 176', *load_b_column('full_b', channels),
              'add x15, x20, x0', 'add x23, x12, x0']
    for row in (0, 2):
        lines += [f'li x4, {(BANK_BYTES + row * k * 32) // 4}',
                  f'li x7, {(BANK_BYTES + (row + 1) * k * 32) // 4}', 'li x6, 0']
        for inner in range(k // 32):
            op = 'vmatmul.mxu0' if inner == 0 else 'vmatmul.acc.mxu0'
            lines += ['vload m1, 0(x6)', 'delay 34', 'vmatpush.weight.mxu0 w0, m1', 'delay 32',
                      'vload m0, 0(x4)', 'delay 34', f'{op} acc0, m0, w0', 'delay 96',
                      'vload m0, 0(x7)', 'delay 34', f'{op} acc1, m0, w0', 'delay 96',
                      'addi x4, x4, 256', 'addi x7, x7, 256', 'addi x6, x6, 256']
        lines += output_pair
    lines += ['add x12, x12, x8', 'addi x14, x14, -1', 'beq x14, x0, full_finished', 'nop',
              'jal x0, full_columns', 'nop', 'full_finished:', 'nop', 'ecall']
    if relocated:
        # Only the new core's loads are unrelocated. The captured output hook
        # already contains its original add/sub relocation around each DMA.
        hook_ids = set()
        # Find exact hook occurrences so we never wrap a captured hook load twice.
        for first in range(len(lines) - len(output_pair) + 1):
            if lines[first:first + len(output_pair)] == output_pair:
                hook_ids.update(range(first, first + len(output_pair)))
        result = []
        for index, line in enumerate(lines):
            if index not in hook_ids and line.startswith('dma.load '):
                result += ['add x20, x20, x31', line, 'sub x20, x20, x31']
            else:
                result.append(line)
        lines = result
    source = '\n'.join(lines) + '\n'
    if not fast:
        check_matrix_registers(source)
        return source
    # This helper performs the m6 restored-value check across the supplied hook.
    # Import lazily so the parent dispatcher may select this module safely.
    from atlas_program_gemm_fast import optimize as optimize_pairs
    result = optimize_pairs(source, *SHAPE)
    check_matrix_registers(result)
    return result


@checked_assembly(scratch_registers=SCALAR_SCRATCH)
def _original_assembly(m, k, n, output_offset, rescale_exponent=None, *, epilogue=None, weight_source=None):
    """Generate the full-A core with the existing output-hook interface."""
    layout(m, k, n)
    if type(output_offset) is not int or output_offset != m * k + k * n + 256:
        raise ValueError('full-A breadth requires the captured guarded output offset')
    if epilogue is not None:
        from atlas_epilogue import Epilogue
        if type(epilogue) is not Epilogue:
            raise ValueError('full-A breadth requires an exact nonresident Epilogue')
        if (epilogue.m, epilogue.n, epilogue.output_offset) != (m, n, output_offset):
            raise ValueError('epilogue geometry differs from contraction')
    if rescale_exponent is not None and (type(rescale_exponent) is not int or not -126 <= rescale_exponent <= 127):
        raise ValueError('rescale factor must be one normal finite BF16 power of two')
    if weight_source is not None:
        weight_source.validate(k, n)
    prefix = ['li x30, 1', 'dma.config x30, ch0', 'dma.config x30, ch1',
              'li x5, 4096', 'li x8, 2048', f'li x3, {OUTPUT_BYTE // 4}',
              f'li x9, {OUTPUT_BYTE // 4 + 256}', f'li x12, {output_offset}',
              f'li x22, {n * 64}', f'li x14, {n // 32}']
    if weight_source is not None:
        prefix += weight_source.setup(20)
    if rescale_exponent is not None:
        prefix += [f'vli.all m4, {(rescale_exponent + 127) << 7}', 'delay 66']
    pair = []
    for acc in (0, 1):
        pair += store_output(acc, 23, rescale_exponent, epilogue) + ['add x23, x23, x22']
    return _core(prefix, pair, weight_source.low if weight_source else A_BYTES,
                 channels=(2, 3) if weight_source else (0, 1), fast=False)


def assembly(m, k, n, output_offset, rescale_exponent=None, *, epilogue=None, weight_source=None):
    """Generate with checked original hook liveness, then pipeline the core."""
    source = _original_assembly(m, k, n, output_offset, rescale_exponent,
                                epilogue=epilogue, weight_source=weight_source)
    from atlas_program_gemm_fast import optimize as optimize_pairs
    return optimize_pairs(source, m, k, n)


def optimize(source, m, k, n):
    """Replace a standard or captured breadth core, preserving its exact hooks.

    The accepted input is the original Tiny breadth program, before the shared
    pair optimizer. Its two-row output sequence is reused twice per column.
    """
    layout(m, k, n)
    lines = source.splitlines()
    required = ('pairs:', 'columns:', 'column_done:', 'pair_done:', 'li x28, 2',
                'li x24, 44', 'li x3, 47104', 'li x9, 47360', 'li x5, 4096',
                'li x8, 2048', 'li x12, 6488320', 'li x22, 65536', 'li x20, 720896')
    if any(lines.count(line) != 1 for line in required):
        raise ValueError('full-A breadth requires the original captured breadth structure')
    start = lines.index('column_done:') + 1
    end = next((i for i in range(start, len(lines)) if lines[i] == 'add x12, x12, x8'), None)
    if end is None or lines[start] != 'add x23, x12, x0':
        raise ValueError('missing captured output-pair boundary')
    pair = lines[start + 1:end]
    if ([line for line in pair if line.startswith('vmatpop.')] !=
            ['vmatpop.bf16.acc.mxu0 m2, acc0', 'vmatpop.bf16.acc.mxu0 m2, acc1']):
        raise ValueError('captured output pair has unexpected accumulator ownership')
    if any(line.endswith(':') or line.split(' ', 1)[0] in ('beq', 'bne', 'jal', 'jalr') for line in pair):
        raise ValueError('captured output pair must be straight-line')
    # The enlarged A region excludes the resident-output hooks used by
    # other families. Source-level selection must enforce that restriction
    # as well as the typed assembly API: all hook VMEM traffic stays in the
    # original bank-0 scratch, and scalar setup cannot move those pointers.
    for line in pair:
        if line.startswith(('vload ', 'vstore ')) and not re.fullmatch(
                r'v(?:load|store) m\d+, 0\(x(?:3|9)\)', line):
            raise ValueError('breadth hook VMEM traffic must stay in output scratch')
        scalar = re.match(r'([a-z.]+) (x\d+),', line)
        if scalar and not scalar[1].startswith('dma.') and scalar[2] not in {'x10', 'x11', 'x23'}:
            raise ValueError('breadth hook overwrites a resident-core register')
        if line.startswith(('dma.load ', 'dma.store ')):
            args = line.replace(',', '').split()
            vmem = args[1] if args[0] == 'dma.load' else args[2]
            if vmem not in {'x3', 'x9'} or args[-1] != 'ch0':
                raise ValueError('breadth hook DMA must stay in output scratch')
        elif line.startswith('dma.') and line != 'dma.wait ch0':
            raise ValueError('unexpected breadth hook DMA configuration')
    expected_tail = ['add x12, x12, x22', 'add x20, x27, x0', 'addi x28, x28, -1',
                     'beq x28, x0, finished', 'nop', 'jal x0, pairs', 'nop',
                     'finished:', 'nop', 'ecall']
    if lines[lines.index('pair_done:') + 1:] != expected_tail:
        raise ValueError('breadth source has an unexpected tail or post-pass')
    prefix = lines[:lines.index('pairs:')]
    configs = [line for line in prefix if line.startswith('dma.config ')]
    if configs != ['dma.config x30, ch0', 'dma.config x30, ch1']:
        raise ValueError('breadth transform requires only the original two DMA channels')
    base_setup = [line for line in prefix if re.search(r'\bx30\b', line) and not line.startswith('dma.config ')]
    relocation_setup = [line for line in prefix if re.search(r'\bx31\b', line)]
    relocated = relocation_setup == ['li x31, 2415919104'] and base_setup == ['li x30, 0']
    standard = base_setup == ['li x30, 1'] and not relocation_setup
    if not (relocated or standard):
        raise ValueError('breadth transform requires standard or established low-word DMA addressing')
    # Columns now span all four rows, so its count is initialized only once.
    prefix += ['li x14, 32']
    return _core(prefix, pair, A_BYTES, relocated=relocated)


def transform(source):
    """Shape-specific convenience API for captured source regeneration."""
    return optimize(source, *SHAPE)
