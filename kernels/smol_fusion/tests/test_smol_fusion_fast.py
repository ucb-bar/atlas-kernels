"""Exact fc1 admission, hook preservation, and streamed operand ownership."""
from dataclasses import replace
from pathlib import Path
import re
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[3]))
import atlas_kernels_path  # noqa: E402,F401
import atlas_program  # noqa: E402
import smol_fusion_fast as fast  # noqa: E402
from atlas_epilogue import Step  # noqa: E402
from atlas_program_reuse_common import store_output  # noqa: E402
from smol_fusion_fp8 import stage_fc1_fp8  # noqa: E402
from smol_weight_relocate import DEFAULT_ARENA, relocate_assembly  # noqa: E402

SHAPE = (128, 768, 3072)


def original(exponent=-19, output_scale=-4):
    e = fast._epilogue(*SHAPE, output_scale)
    return atlas_program.assembly(*SHAPE, e.output_offset, exponent,
                                  traffic_reuse=True, epilogue=e), e


def prove_core(source):
    """Track every waited DMA identity and ordered product before each hook.

    The hook is checked verbatim separately. Its only persistent scalar writes
    are x10/x11, while its matrix scratch is overwritten before the next tile.
    """
    m, k, n = SHAPE
    lines = source.splitlines()
    labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(':')}
    assert len(labels) == sum(line.endswith(':') for line in lines)
    regs, memory, matrix, accum = {'x0': 0}, {}, {}, {}
    pending, configured, popped = {}, set(), set()
    load_bytes, weight, pc = 0, None, 0

    def accessible(base, size=1024):
        assert base % 1024 == 0 and 0 <= base < base + size <= 2 * fast.BANK_BYTES
        assert base // fast.BANK_BYTES == (base + size - 1) // fast.BANK_BYTES
        assert all(base + size <= start or end <= base for start, end, _ in pending.values())
        return base

    def address(arg):
        return accessible(4 * regs[re.fullmatch(r'0\((x\d+)\)', arg)[1]])

    for _ in range(400000):
        line = lines[pc]
        op, *args = line.replace(',', '').split()
        pc += 1
        if line == 'smol_fused_finished:':
            assert not pending
            assert len(popped) == m * n // 1024
            assert load_bytes == m * k + k * n
            return
        if op.endswith(':') or op in ('nop', 'delay', 'vli.all'):
            continue
        if op == 'li':
            regs[args[0]] = int(args[1], 0)
        elif op in ('add', 'addi'):
            regs[args[0]] = regs[args[1]] + (regs[args[2]] if op == 'add' else int(args[2]))
        elif op == 'beq':
            if regs[args[0]] == regs[args[1]]:
                pc = labels[args[2]]
        elif op == 'jal':
            assert args[0] == 'x0'
            pc = labels[args[1]]
        elif op == 'dma.config':
            assert regs[args[0]] == 1
            configured.add(args[1])
        elif op == 'dma.load':
            vmem, dram, size = (regs[arg] for arg in args[:3])
            base, end = 4 * vmem, 4 * vmem + size
            accessible(base, size)
            assert args[3] in configured and args[3] not in pending
            assert size in (1024, 4096)
            assert 0 <= dram < dram + size <= m * k + k * n
            pending[args[3]] = (base, end, {base + d: (dram + d, False)
                                         for d in range(0, size, 1024)})
            load_bytes += size
        elif op == 'dma.wait':
            _, _, writes = pending.pop(args[0])
            memory.update(writes)
        elif op == 'vload':
            matrix[args[0]] = memory[address(args[1])]
        elif op == 'vtrpose.xlu':
            src, transposed = matrix[args[1]]
            assert src >= m * k and not transposed
            matrix[args[0]] = (src, True)
        elif op == 'vstore':
            assert args[0] == 'm1'
            memory[address(args[1])] = matrix[args[0]]
        elif op == 'vmatpush.weight.mxu0':
            weight = matrix[args[1]]
            assert weight[1]
        elif op in ('vmatmul.mxu0', 'vmatmul.acc.mxu0'):
            src, transposed = matrix[args[1]]
            assert not transposed and src < m * k
            row, inner = divmod(src, k * 32)
            col, b_inner = divmod(weight[0] - m * k, k * 32)
            assert inner == b_inner
            tile = inner // 1024
            if op == 'vmatmul.mxu0':
                assert tile == 0
                accum[args[0]] = (row, col, [])
            a_row, a_col, products = accum[args[0]]
            assert (row, col) == (a_row, a_col)
            assert tile == len(products)
            products.append(tile)
        elif op == 'vmatpop.bf16.acc.mxu0':
            assert not pending
            row, col, products = accum.pop(args[1])
            assert len(products) == k // 32
            assert (row, col) not in popped
            assert regs['x22'] == m * k + k * n + 256 + row * n * 64 + col * 2048
            popped.add((row, col))
            while lines[pc] != 'add x22, x22, x16':
                pc += 1
        else:
            raise AssertionError('unexpected core instruction: ' + line)
    raise AssertionError('core failed to terminate')


class SmolFusionFastTests(unittest.TestCase):
    def test_streamed_operands_have_waited_disjoint_dma_and_ordered_products(self):
        source, _ = original()
        prove_core(fast.optimize(source, *SHAPE))

    def test_full_hook_arithmetic_is_verbatim_for_both_accumulators(self):
        for exponent, scale in ((-19, -4), (-126, -126), (127, 119)):
            source, e = original(exponent, scale)
            result = fast.optimize(source, *SHAPE)
            for acc in (0, 1):
                hook = '\n'.join(store_output(acc, 22, exponent, e)) + '\n'
                self.assertIn(hook, source)
                self.assertEqual(result.count(hook), 2)
            self.assertEqual(result.count('vli.all m4,'), 1)
            record = fast.describe(source, *SHAPE)
            self.assertEqual(record['output_scale_exponent'], scale)
            self.assertEqual(record['dma_load_bytes'], 3244032)
            self.assertEqual(record['dma_store_bytes'], 393216)
            self.assertEqual(record['extra_offset'], 2851328)

    def test_exact_low_arena_relocation_commutes(self):
        source, _ = original()
        low = relocate_assembly(source, DEFAULT_ARENA, fast.LOW_ARENA)
        self.assertEqual(fast.optimize(low, *SHAPE),
                         relocate_assembly(fast.optimize(source, *SHAPE), DEFAULT_ARENA, fast.LOW_ARENA))
        self.assertEqual(fast.describe(low, *SHAPE)['dram_base'], fast.LOW_ARENA)

    def test_modified_programs_other_hooks_scales_and_repeated_transforms_fail(self):
        source, e = original()
        custom = atlas_program.assembly(*SHAPE, e.output_offset, -19, traffic_reuse=True,
                                        epilogue=replace(e, steps=(Step('relu'),)))
        candidates = [source.replace('delay 96', 'delay 95', 1),
                      source.replace('vadd.bf16 m2, m2, m6', 'vmul.bf16 m2, m2, m6', 1),
                      source.replace('li x12, 2457856', 'li x12, 2458880', 1),
                      source.replace('seli e0, 131', 'seli e0, 132', 1),
                      source.replace('vli.all m4, 13824', 'vli.all m4, 13825'),
                      source.replace('vli.all m4, 13824', 'vli.all m4, 32640'),
                      source + 'nop\n', custom, fast.optimize(source, *SHAPE), None,
                      relocate_assembly(source, DEFAULT_ARENA, 0x80000000)]
        for candidate in candidates:
            with self.subTest(source=str(candidate)[:50]), self.assertRaises(ValueError):
                fast.optimize(candidate, *SHAPE)

    def test_layout_and_shape_admission(self):
        plan = fast.layout(*SHAPE)
        regions = [plan[key] for key in ('a_bank_range', 'b_column_range', 'output_scratch_range')]
        for i, (start, end) in enumerate(regions):
            self.assertEqual(start // fast.BANK_BYTES, (end - 1) // fast.BANK_BYTES)
            for other_start, other_end in regions[i + 1:]:
                self.assertTrue(end <= other_start or other_end <= start)
        for shape in ((True, 768, 3072), (64, 768, 3072), (128, 768, 768)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                fast.layout(*shape)

    def test_staging_retains_every_arena_byte_and_metadata_except_source(self):
        m, k, n = SHAPE
        offset = m * k + k * n + 256
        size = offset + 2 * m * n + 256
        row = dict(module='model.vision_model.encoder.layers.0.mlp.fc1', physical_mkn=list(SHAPE),
                   recipe='traffic-reuse-v1', output_offset=offset, output_rescale_exponent=-19,
                   requested_dma=dict(load_bytes=m*k+k*n, store_bytes=2*m*n))
        arena = b'A'*(m*k) + b'W'*(k*n) + b'\xa5'*256 + b'\xcd'*(2*m*n) + b'Z'*256
        self.assertEqual(len(arena), size)
        bias = struct.pack(f'<{n}H', *([0x3c00] * n))
        kwargs = dict(bias_bf16=bias, output_scale_exponent=-4)
        old = stage_fc1_fp8(row, arena, **kwargs)
        new = fast.stage_fc1_fp8_fast(row, arena, **kwargs)
        self.assertEqual(new.arena, old.arena)
        self.assertEqual(new.source, fast.optimize(old.source, *SHAPE))
        self.assertEqual({k: v for k, v in new.metadata.items() if k not in ('variant', 'source_sha256')},
                         {k: v for k, v in old.metadata.items() if k != 'source_sha256'})


if __name__ == '__main__':
    unittest.main()
