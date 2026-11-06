"""Exact family admission, hook preservation, and streamed operand ownership."""
from dataclasses import replace
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
import atlas_program  # noqa: E402
import atlas_program_tiny_fused_fast as fast  # noqa: E402
from atlas_epilogue import Epilogue, Step  # noqa: E402
from atlas_program_reuse_common import store_output  # noqa: E402
from atlas_rope_postpass import append_postpass, postpass  # noqa: E402
from smol_weight_relocate import DEFAULT_ARENA, relocate_assembly  # noqa: E402


def original(shape, kind, exponent=-14, **kwargs):
    e = fast._epilogue(*shape, kind, **kwargs)
    source = atlas_program.assembly(*shape, e.output_offset, exponent,
                                    traffic_reuse=True, epilogue=e)
    return (append_postpass(source, e) if kind == 'rope' else source), e


def prove_core(source, m, k, n):
    """Execute core scalar flow and track exact 1-KiB A/B tile identities.

    Hook blocks are checked verbatim separately. Skip those blocks here so
    this proof needs no model of their BF16 arithmetic or post-pass scratch.
    Every DMA tile remains inaccessible until its channel is waited.
    """
    lines = source.splitlines()
    labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(':')}
    assert len(labels) == sum(line.endswith(':') for line in lines)
    regs, memory, matrix, accum = {'x0': 0}, {}, {}, {}
    pending, configured, popped = {}, set(), set()
    loads, weight, pc = 0, None, 0

    def accessible(base, size=1024):
        assert base % 1024 == 0 and 0 <= base < base + size <= 4 * fast.BANK_BYTES
        assert base // fast.BANK_BYTES == (base + size - 1) // fast.BANK_BYTES
        assert all(base + size <= start or end <= base for start, end, _ in pending.values())
        return base

    def address(arg):
        return accessible(4 * regs[re.fullmatch(r'0\((x\d+)\)', arg)[1]])

    for _ in range(1000000):
        line = lines[pc]
        op, *args = line.replace(',', '').split()
        pc += 1
        if line == 'tiny_fused_finished:':
            assert not pending
            assert len(popped) == m * n // 1024
            assert loads == m * k + k * n
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
            writes = {base + d: (dram + d, False) for d in range(0, size, 1024)}
            pending[args[3]] = (base, end, writes)
            loads += size
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
            row, col, products = accum.pop(args[1])
            assert len(products) == k // 32
            assert (row, col) not in popped
            assert regs['x22'] == m * k + k * n + 256 + row * n * 64 + col * 2048
            popped.add((row, col))
            # Exact hook blocks are tested below. Core-live scalar pointers
            # and operands survive those blocks; only this final advance is
            # part of the core flow.
            while lines[pc] != 'add x22, x22, x16':
                pc += 1
        else:
            raise AssertionError('unexpected core instruction: ' + line)
    raise AssertionError('core failed to terminate')


class TinyFusedFastTests(unittest.TestCase):
    def test_every_family_has_ordered_products_and_compulsory_input_traffic(self):
        for shape in sorted(fast.SHAPES):
            for kind in fast.kinds(*shape):
                with self.subTest(shape=shape, kind=kind):
                    source, _ = original(shape, kind)
                    prove_core(fast.optimize(source, *shape), *shape)

    def test_all_hook_arithmetic_and_rope_postpass_are_verbatim(self):
        for shape in sorted(fast.SHAPES):
            for kind in fast.kinds(*shape):
                with self.subTest(shape=shape, kind=kind):
                    source, e = original(shape, kind)
                    result = fast.optimize(source, *shape)
                    for acc in range(min(2, shape[0] // 32)):
                        hook = '\n'.join(store_output(acc, 22, -14, e)) + '\n'
                        self.assertEqual(result.count(hook), max(1, shape[0] // 64))
                    if kind == 'rope':
                        self.assertTrue(result.endswith('\n'.join(postpass(e) + ['ecall']) + '\n'))
                    self.assertEqual(result.count('vli.all m4,'), 1)

    def test_fp8_shards_keep_scale_strides_holes_and_operand_offsets(self):
        for m in (128,):
            shape = (m, 2048, 2048)
            for column in (0, 64):
                for scale in (-13, -1, 0):
                    with self.subTest(m=m, column=column, scale=scale):
                        source, e = original(shape, 'mul', output_scale_exponent=scale,
                                             output_column_tile=column)
                        record = fast.describe(source, *shape)
                        result = fast.optimize(source, *shape)
                        self.assertEqual(record['output_dtype'], 'fp8')
                        self.assertEqual(record['output_scale_exponent'], scale)
                        self.assertEqual(record['output_column_tile'], column)
                        self.assertEqual(record['output_row_stride'], 180224)
                        self.assertEqual(record['output_bytes'], e.output_bytes)
                        self.assertEqual(record['dma_store_bytes'], m * 2048)
                        self.assertIn(f'seli e0, {127 - scale}\n', result)

    def test_exact_low_arena_relocation_commutes_with_optimization(self):
        for shape in sorted(fast.SHAPES):
            for kind in fast.kinds(*shape):
                with self.subTest(shape=shape, kind=kind):
                    source, _ = original(shape, kind)
                    relocated = relocate_assembly(source, DEFAULT_ARENA, fast.LOW_ARENA)
                    expected = relocate_assembly(fast.optimize(source, *shape), DEFAULT_ARENA, fast.LOW_ARENA)
                    self.assertEqual(fast.optimize(relocated, *shape), expected)
                    self.assertEqual(fast.describe(relocated, *shape)['dram_base'], fast.LOW_ARENA)

    def test_modified_templates_custom_hooks_and_second_transform_reject(self):
        shape = (128, 2048, 2048)
        source, e = original(shape, 'add_rne')
        custom = replace(e, steps=(Step('silu'), *e.steps))
        custom_source = atlas_program.assembly(*shape, e.output_offset, -14,
                                               traffic_reuse=True, epilogue=custom)
        plain = atlas_program.assembly(*shape, e.output_offset, -14, traffic_reuse=True)
        candidates = [source.replace('delay 96', 'delay 95', 1),
                      source.replace('li x12, 4456704', 'li x12, 4456736', 1),
                      source.replace('vmul.bf16 m2, m2, m4', 'vadd.bf16 m2, m2, m4', 1),
                      source.replace('vli.all m4, 14464', 'vli.all m4, 14465', 1),
                      source + 'nop\n', plain, custom_source, fast.optimize(source, *shape),
                      relocate_assembly(source, DEFAULT_ARENA, 0x80000000)]
        for candidate in candidates:
            with self.subTest(source=candidate[:80]), self.assertRaises(ValueError):
                fast.optimize(candidate, *shape)

    def test_layout_ranges_are_disjoint_and_bank_bounded(self):
        for shape in fast.SHAPES:
            plan = fast.layout(*shape)
            regions = [plan['b_column_range'], plan['output_scratch_range'], *plan['a_bank_ranges']]
            for i, (start, end) in enumerate(regions):
                self.assertEqual(start % 1024, 0)
                self.assertEqual(end % 1024, 0)
                self.assertEqual(start // fast.BANK_BYTES, (end - 1) // fast.BANK_BYTES)
                for other_start, other_end in regions[i + 1:]:
                    self.assertTrue(end <= other_start or other_end <= start)
            self.assertEqual(sum(end - start for start, end in plan['a_bank_ranges']), shape[0] * shape[1])

    def test_unqualified_dimensions_and_invalid_scales_reject(self):
        for shape in ((True, 2048, 2048), (32, 2048, 2048), (32, 5632, 1024), (128, 2048, 1024)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                fast.layout(*shape)
        shape = (128, 2048, 2048)
        source, _ = original(shape, 'silu')
        for immediate in (0, 32640, 14465):
            with self.subTest(immediate=immediate), self.assertRaises(ValueError):
                fast.optimize(source.replace('vli.all m4, 14464', f'vli.all m4, {immediate}'), *shape)

    def test_exact_original_32_row_program_is_rejected(self):
        shape = (32, 2048, 2048)
        m, k, n = shape
        e = Epilogue(m, n, m * k + k * n + 256, steps=(Step('silu'),),
                     scratch_registers=(10, 11))
        source = atlas_program.assembly(*shape, e.output_offset, -14,
                                        traffic_reuse=True, epilogue=e)
        for text in (source, relocate_assembly(source, DEFAULT_ARENA, fast.LOW_ARENA)):
            for admit in (fast.optimize, fast.describe):
                with self.subTest(admit=admit.__name__), self.assertRaisesRegex(ValueError, 'qualified physical shape'):
                    admit(text, *shape)


if __name__ == '__main__':
    unittest.main()
