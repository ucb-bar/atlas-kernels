"""Streaming admission and exact constant-input checks independent of the VPU sequence."""
from pathlib import Path
import os
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[3]))
import atlas_kernels_path  # noqa: E402,F401
from atlas_rmsnorm_tiled import MAX_WIDTH, standalone_source, epsilon_bits


class TiledNormTests(unittest.TestCase):
    def source(self, rows, width):
        mask = 3 * rows * width + 256
        return standalone_source(rows, width, mask, mask + 65536, 0)

    def test_epsilon_encoding_and_rejection(self):
        self.assertEqual(epsilon_bits(1e-5), 0x3728)
        self.assertEqual(epsilon_bits(1e-6), 0x3586)
        for eps in (0, -1e-6, float('nan'), float('inf'), 1e-50, 1e-39, 3.4e38, 10**400, True, '1e-6'):
            with self.subTest(eps=eps), self.assertRaisesRegex(ValueError, 'eps'):
                standalone_source(32, 1536, 147712, 213248, eps=eps)
        # BF16 halfway values round to the even low bit.
        self.assertEqual(epsilon_bits(1 + 1 / 256), 0x3f80)
        self.assertEqual(epsilon_bits(1 + 3 / 256), 0x3f82)

    def test_epsilon_changes_only_loaded_constant(self):
        for rows, width in ((32, 1536), (128, 1536), (32, 960), (128, 2048)):
            mask = 3 * rows * width + 256
            old = standalone_source(rows, width, mask, mask + 65536)
            explicit = standalone_source(rows, width, mask, mask + 65536, eps=1e-5)
            changed = standalone_source(rows, width, mask, mask + 65536, eps=1e-6)
            self.assertEqual(old, explicit)
            self.assertEqual(changed, old.replace('vli.all m6, 14120', 'vli.all m6, 13702'))

    @unittest.skipUnless(os.environ.get('ATLAS_NPU_MODEL_ROOT'), 'set ATLAS_NPU_MODEL_ROOT for assembler')
    def test_cli_opt_in_and_invalid_epsilon(self):
        generator = Path(__file__).absolute().parents[1] / 'gen.py'
        env = dict(os.environ)
        env['PYTHONPATH'] = os.environ['ATLAS_NPU_MODEL_ROOT'] + os.pathsep + env.get('PYTHONPATH', '')
        with tempfile.TemporaryDirectory() as out:
            for width in (960, 1536, 2048):
                result = subprocess.run([sys.executable, str(generator), '--rows', '32', '--width', str(width),
                                         '--eps', '1e-6', '--out', out], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('vli.all m6, 13702', (Path(out) / f'rmsnorm-32x{width}.S').read_text())
            result = subprocess.run([sys.executable, str(generator), '--eps', 'nan', '--out', out,
                                     '--name', 'invalid'], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((Path(out) / 'invalid.bin').exists())

    def test_tile_shapes_and_partial_dma(self):
        # One tile is 2048 BF16 bytes, shorter than the normal DMA chunk.
        self.assertIn('li x17, 2048', self.source(32, 32))
        for rows, width in ((32, 32), (128, 1024), (128, 2304), (128, 4096), (32, MAX_WIDTH)):
            with self.subTest(rows=rows, width=width):
                self.assertTrue(self.source(rows, width).endswith('ecall\n'))

    def test_reject_shapes_before_assembly(self):
        for rows, width in ((0, 32), (33, 32), (32, 0), (32, 33), (32, MAX_WIDTH + 32)):
            with self.subTest(rows=rows, width=width), self.assertRaisesRegex(ValueError, 'multiple of 32'):
                self.source(rows, width)

    def test_arena_refuses_overlap_and_low32_overflow(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            standalone_source(32, 32, 1024, 65536)
        with self.assertRaisesRegex(ValueError, 'low32'):
            standalone_source(32, 32, 2**32, 2**32 + 65536)

    @unittest.skipUnless(os.environ.get('ATLAS_NPU_MODEL_ROOT'), 'set ATLAS_NPU_MODEL_ROOT for device execution')
    def test_constant_rows_exact_and_arena_guards(self):
        import numpy as np
        import kernel_bench as bench
        from atlas_execution import Executor
        # Unit-magnitude rows with unit gamma give signed unit output. Epsilon
        # is below one BF16 ulp; the independently known FP8 bytes are +/-1.
        for rows, width in ((32, 32), (32, 1024), (128, 2304), (128, 4096)):
            with self.subTest(rows=rows, width=width):
                h = np.full((rows, width), 0x3f80, dtype=np.uint16)
                h[1::2] = 0xbf80
                masks = np.zeros((32, 32, 32), dtype=np.uint16)
                for i in range(32):
                    masks[i, :, i] = 0x3f80
                output_offset, size = 2 * rows * width, rows * width
                arena = (bench.pack_bf16_tiles(h) + b'\xcd' * size + b'\x5a' * 256
                         + b''.join(bench.pack_bf16_tiles(t) for t in masks)
                         + bench.pack_bf16_tiles(np.full((32, width), 0x3f80, dtype=np.uint16)) + b'\xa5' * 256)
                result = Executor(os.environ['ATLAS_NPU_MODEL_ROOT']).execute(
                    bench.relocate(self.source(rows, width)), arena, output_offset, size)
                expected = np.full((rows, width), 0x38, dtype=np.uint8)
                expected[1::2] = 0xb8
                self.assertEqual(result.output, bench.pack_fp8_tiles(expected))
                self.assertEqual(result.arena[:output_offset], arena[:output_offset])
                self.assertEqual(result.arena[output_offset + size:], arena[output_offset + size:])


if __name__ == '__main__':
    unittest.main()
