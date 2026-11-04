"""Bench codecs are exact inverses, FP8 encoding is nearest-even, and every reference case builds."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
import kernel_bench as bench  # noqa: E402
from atlas_vector_arithmetic import decode_fp8  # noqa: E402


class KernelBenchTest(unittest.TestCase):
    def test_tile_codecs_round_trip(self):
        rng = np.random.default_rng(1)
        words = rng.integers(0, 0x7f00, (64, 96), dtype=np.uint16)
        codes = rng.integers(0, 0x7e, (64, 96), dtype=np.uint8)
        self.assertTrue(np.array_equal(bench.unpack_bf16_tiles(bench.pack_bf16_tiles(words), 64, 96), words))
        self.assertTrue(np.array_equal(bench.unpack_fp8_tiles(bench.pack_fp8_tiles(codes), 64, 96), codes))

    def test_fp8_encoding(self):
        codes = np.array([c for c in range(1, 127) if (c >> 3) & 15], dtype=np.uint8)
        values = decode_fp8(codes)
        self.assertTrue(np.array_equal(bench.encode_fp8(values), codes))
        self.assertTrue(np.array_equal(bench.encode_fp8(-values), codes | 0x80))
        self.assertEqual(int(bench.encode_fp8(1e6)), 0x7e)
        self.assertEqual(int(bench.encode_fp8(1e-6)), 0)
        midpoint = (values[8] + values[9]) / 2
        self.assertEqual(int(bench.encode_fp8(midpoint)), int(codes[8] if codes[8] % 2 == 0 else codes[9]))

    def test_every_case_builds(self):
        for family in bench.families():
            module = bench.load_reference(family)
            for spec in module.CASES:
                with self.subTest(family=family, case=spec['name']):
                    case = module.build(spec, np.random.default_rng(spec.get('seed', 0)))
                    self.assertLessEqual(case.output_offset + case.output_bytes, len(case.arena))
                    self.assertTrue(case.source.rstrip().endswith('ecall'))
                    if case.golden is not None:
                        self.assertEqual(len(case.golden), case.output_bytes)


if __name__ == '__main__':
    unittest.main()
