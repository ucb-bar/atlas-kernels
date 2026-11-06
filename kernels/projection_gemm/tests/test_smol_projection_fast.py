"""Symbolic operand, residency, and strict admission checks for streamed B."""
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[3]))
import atlas_kernels_path  # noqa: E402,F401
import atlas_program
import atlas_program_smol_projection_fast as fast
from atlas_epilogue import Epilogue, Step
from smol_weight_relocate import DEFAULT_ARENA, relocate_assembly
from reuse_proof import prove


class SmolProjectionFastTests(unittest.TestCase):
    def test_all_shapes_preserve_ordered_products_and_compulsory_traffic(self):
        for m, k, n in sorted(fast.SHAPES):
            with self.subTest(shape=(m, k, n)):
                offset = m * k + k * n + 256
                old = atlas_program.assembly(m, k, n, offset, -14, traffic_reuse=True)
                new = fast.optimize(old, m, k, n)
                before, after = [prove(source, m, k, n, -14) for source in (old, new)]
                # prove checks the source identities of every product, increasing
                # K, one pop per output, disjoint DMA ownership, and drained DMA.
                self.assertEqual(after['load_bytes'], m * k + k * n)
                self.assertEqual(after['store_bytes'], 2 * m * n)
                self.assertEqual(before['counts']['vmatmul.mxu0'], after['counts']['vmatmul.mxu0'])
                self.assertEqual(before['counts']['vmatmul.acc.mxu0'], after['counts']['vmatmul.acc.mxu0'])
                self.assertEqual(before['counts']['vmatpop.bf16.acc.mxu0'], after['counts']['vmatpop.bf16.acc.mxu0'])
                self.assertEqual(Counter((src, size) for _, src, size in before['loads']),
                                 Counter((src, size) for _, src, size in after['loads']))
                self.assertEqual([(dst, size) for dst, _, size in before['stores']],
                                 [(dst, size) for dst, _, size in after['stores']])
                self.assertLess(after['delays'], before['delays'])

    def test_workspace_ranges_are_disjoint_within_physical_banks(self):
        for m, k, n in sorted(fast.SHAPES):
            with self.subTest(shape=(m, k, n)):
                plan = fast.layout(m, k, n)
                regions = [plan[key] for key in ('b_column_range', 'output_scratch_range',
                                                'a_bank1_range', 'a_bank0_range')]
                regions = [(start, end) for start, end in regions if start < end]
                for start, end in regions:
                    self.assertEqual(start % 1024, 0)
                    self.assertEqual(end % 1024, 0)
                    self.assertLessEqual(end, 2 * fast.BANK_BYTES)
                    self.assertEqual(start // fast.BANK_BYTES, (end - 1) // fast.BANK_BYTES)
                for i, (start, end) in enumerate(regions):
                    for other_start, other_end in regions[i + 1:]:
                        self.assertTrue(end <= other_start or other_end <= start)
                self.assertEqual(sum(end - start for start, end in
                                     (plan['a_bank1_range'], plan['a_bank0_range'])), m * k)
                self.assertEqual(plan['b_column_range'], [0, k * 32])
                self.assertEqual(plan['output_scratch_range'], [k * 32, k * 32 + 2048])

    def test_relocation_matches_exact_generated_variant(self):
        for m, k, n in sorted(fast.SHAPES):
            with self.subTest(shape=(m, k, n)):
                offset = m * k + k * n + 256
                old = atlas_program.assembly(m, k, n, offset, -15, traffic_reuse=True)
                relocated = relocate_assembly(old, DEFAULT_ARENA, 0x90000000)
                expected = relocate_assembly(fast.assembly(m, k, n, offset, -15), DEFAULT_ARENA, 0x90000000)
                self.assertEqual(fast.optimize(relocated, m, k, n), expected)

    def test_modified_original_hooks_and_repeated_optimization_are_rejected(self):
        m, k, n = 128, 768, 768
        offset = m * k + k * n + 256
        old = atlas_program.assembly(m, k, n, offset, 0, traffic_reuse=True)
        hook = Epilogue(m, n, offset, steps=(Step('relu'),))
        hooked = atlas_program.assembly(m, k, n, offset, 0, traffic_reuse=True, epilogue=hook)
        for source in (old.replace('delay 96', 'delay 95', 1),
                       old.replace('dma.load x19, x20, x5, ch0', 'dma.load x19, x20, x5, ch1', 1),
                       old.replace(f'li x23, {offset}', f'li x23, {offset + 1024}'),
                       old.replace('vli.all m4, 16256', 'vli.all m4, 16257'),
                       old.replace('vli.all m4, 16256', 'vli.all m4, 32640'),
                       old + 'nop\n', hooked, fast.optimize(old, m, k, n)):
            with self.subTest(source=source[:50]), self.assertRaises(ValueError):
                fast.optimize(source, m, k, n)

    def test_unqualified_shapes_offsets_and_scales_are_rejected(self):
        for shape in ((32, 768, 768), (64, 736, 736), (True, 768, 768)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                fast.layout(*shape)
        for offset, scale in ((688385, 0), (688384, True), (688384, None),
                              (688384, -127), (688384, 128)):
            with self.subTest(offset=offset, scale=scale), self.assertRaises(ValueError):
                fast.assembly(128, 768, 768, offset, scale)


if __name__ == '__main__':
    unittest.main()
