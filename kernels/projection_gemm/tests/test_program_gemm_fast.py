"""Prove preserved tile ownership and K order for the opt-in paired core."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[3]))
import atlas_kernels_path  # noqa: E402,F401
import atlas_program
import atlas_program_gemm_fast as fast
from reuse_proof import prove


class FastCoreTests(unittest.TestCase):
    def test_six_original_layouts_preserve_ordered_products_and_dma(self):
        for m, k, n in sorted(fast.SHAPES - {(128, 5632, 1024)}):
            with self.subTest(shape=(m, k, n)):
                offset = m * k + k * n + 256
                source = atlas_program.assembly(m, k, n, offset, -14, traffic_reuse=True)
                candidate = fast.assembly(m, k, n, offset, -14)
                old, new = [prove(text, m, k, n, -14) for text in (source, candidate)]
                for key in ('load_bytes', 'store_bytes', 'loads', 'stores', 'b_stores'):
                    self.assertEqual(old[key], new[key])
                self.assertLess(new['delays'], old['delays'])
                normalize = lambda text: [line.replace('m6', 'm0') for line in text.splitlines()
                                          if not line.startswith('delay ')]
                self.assertEqual(normalize(source), normalize(candidate))

    def test_full_a_rejects_a_hook_that_overwrites_resident_input(self):
        from atlas_epilogue import Epilogue
        from atlas_resident_output import retain_output

        class RetainedOutput(Epilogue):
            def store(self, output_reg):
                return retain_output(self, output_reg)

        m, k, n = 128, 5632, 1024
        offset = m * k + k * n + 256
        hook = RetainedOutput(m, n, offset)
        with self.assertRaises(ValueError):
            fast.assembly(m, k, n, offset, -14, epilogue=hook)
        original = atlas_program.assembly(m, k, n, offset, -14, traffic_reuse=True, epilogue=hook)
        with self.assertRaises(ValueError):
            fast.optimize(original, m, k, n)
        plain = atlas_program.assembly(m, k, n, offset, -14, traffic_reuse=True)
        with self.assertRaises(ValueError):
            fast.optimize(plain.removesuffix('ecall\n') + 'nop\necall\n', m, k, n)

    def test_decode_and_unqualified_shapes_are_rejected(self):
        for shape in ((32, 2048, 2048), (64, 736, 960), (True, 768, 768)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                fast.optimize('', *shape)

    def test_unexpected_schedule_is_rejected(self):
        shape = (128, 768, 768)
        old = atlas_program.assembly(*shape, 688384, 0, traffic_reuse=True)
        for source in ('ecall\n', old.replace('delay 96', 'delay 95'),
                       old.replace('vload m0, 0(x7)', 'vload m7, 0(x7)'),
                       old.replace('vmatpop.bf16.acc.mxu0 m2, acc0',
                                   'vmul.bf16 m2, m2, m6\nvmatpop.bf16.acc.mxu0 m2, acc0')):
            with self.subTest(source=source[:50]), self.assertRaises(ValueError):
                fast.optimize(source, *shape)
        with self.assertRaises(ValueError):
            fast.optimize(fast.optimize(old, *shape), *shape)

    def test_persistent_hook_scale_cannot_cross_row1_compute(self):
        shape = (128, 768, 768)
        old = atlas_program.assembly(*shape, 688384, 0, traffic_reuse=True)
        compute = 'vmatmul.mxu0 acc0, m0, w0\ndelay 96'
        source = old.replace(compute, compute + '\nvli.all m6, 16256\ndelay 66', 1)
        pop = 'vmatpop.bf16.acc.mxu0 m2, acc0\ndelay 34'
        source = source.replace(pop, pop + '\nvmul.bf16 m2, m2, m6\ndelay 66', 1)
        with self.assertRaisesRegex(ValueError, 'original m6 value is live'):
            fast.optimize(source, *shape)

    def test_hook_restore_must_not_be_skipped_by_forward_branch(self):
        shape = (128, 768, 768)
        old = atlas_program.assembly(*shape, 688384, 0, traffic_reuse=True)
        pop = 'vmatpop.bf16.acc.mxu0 m2, acc0\ndelay 34\n'
        restore = 'vli.all m6, 16256\ndelay 66\n'
        consume = 'vmul.bf16 m2, m2, m6\ndelay 66\n'
        # A straight-line full restore makes the hook independent of the core.
        fast.optimize(old.replace(pop, pop + restore + consume, 1), *shape)
        skipped = ('beq x0, x0, fast_skip_restore\nnop\n' + restore
                   + 'fast_skip_restore:\n' + consume)
        with self.assertRaisesRegex(ValueError, 'original m6 value is live'):
            fast.optimize(old.replace(pop, pop + skipped, 1), *shape)


if __name__ == '__main__':
    unittest.main()
