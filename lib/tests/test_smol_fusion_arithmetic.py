"""Directed arithmetic boundaries and end-to-end finite FC1 examples."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'golden'))
from smol_fusion_arithmetic import add_truncated, golden_arena, multiply, square, tanh


class SmolFusionArithmeticTest(unittest.TestCase):
    def test_add_truncates_and_multiply_preserves_zero_sign(self):
        # 1 + 3/256 is above the BF16 midpoint, but native add truncates.
        a = np.array([0x3f80], dtype=np.uint16)
        b = np.array([0x3c40], dtype=np.uint16)
        self.assertEqual(int(add_truncated(a, b)[0]), 0x3f81)
        self.assertEqual(int(multiply(np.array([0x8000], dtype=np.uint16), a)[0]), 0x8000)

    def test_square_truncates_its_integer_significand(self):
        # (181/128)^2 has significand 255.9453125 at exponent zero.
        self.assertEqual(square(0x3fb5), 0x3fff)
        self.assertEqual(square(0xbfb5), 0x3fff)
        self.assertEqual(square(1), 0)
        self.assertEqual(square(0), 0)

    def test_tanh_fixed_point_midpoint_and_saturation(self):
        self.assertEqual(tanh(0), 0)
        self.assertEqual(tanh(0x8000), 0x8000)
        self.assertEqual(tanh(0x3700), 0)  # 2**-17 collapses in Q3.16.
        # At x=1/16, interpolate halfway from knot 0 to knot 8150.
        self.assertEqual(tanh(0x3d80), 0x3d7f)
        self.assertEqual(tanh(0xbd80), 0xbd7f)
        self.assertEqual(tanh(0x4080), 0x3f80)
        self.assertEqual(tanh(0xc080), 0xbf80)
        # The final interpolation cell repeats the last knot through x<4.
        self.assertEqual(tanh(0x4078), tanh(0x407f))

    def test_nonfinite_and_overflow_domain_is_refused(self):
        for operation, bits in ((square, 0x7f80), (square, 0x7f00), (tanh, 0x7fc0)):
            with self.assertRaises(ValueError):
                operation(bits)
        with self.assertRaises(ValueError):
            multiply(np.array([0x7f00], dtype=np.uint16), np.array([0x7f00], dtype=np.uint16))

    def test_complete_fc1_and_consumer_scale_clipping(self):
        m = k = n = 32
        offset = m*k + k*n + 256
        extra = offset + m*n + 256
        initial = (bytes([0x38])*(m*k+k*n) + bytes([0xa5])*256
                   + bytes([0xcd])*(m*n) + b'Z'*256 + bytes(64*n) + b'Z'*256)
        fusion = dict(physical_mkn=[m, k, n], output_offset=offset,
                      output_dtype='fp8', operations=['bias', 'gelu_tanh'],
                      total_dram_bytes=len(initial), extra_offset=extra,
                      output_rescale_exponent=0, output_scale_exponent=0)
        golden, stats = golden_arena(initial, fusion)
        # A=B=1 gives ordered sum 32, tanh saturates to 1, GELU gives 32.
        self.assertEqual(golden[offset:offset+m*n], bytes([0x60])*(m*n))
        self.assertEqual(stats['saturation_values'], 0)
        self.assertEqual(golden[:offset], initial[:offset])
        self.assertEqual(golden[offset+m*n:], initial[offset+m*n:])
        fusion['output_scale_exponent'] = -5  # limit 448/32 = 14.
        golden, stats = golden_arena(initial, fusion)
        self.assertEqual(golden[offset:offset+m*n], bytes([0x7e])*(m*n))
        self.assertEqual(stats['saturation_values'], m*n)
        malformed = bytearray(initial)
        malformed[extra+32] = 1
        with self.assertRaisesRegex(ValueError, 'channel bias tile rows differ'):
            golden_arena(bytes(malformed), fusion)
        fusion['extra_offset'] += 2
        with self.assertRaisesRegex(ValueError, 'bias arena geometry differs'):
            golden_arena(initial, fusion)


if __name__ == '__main__':
    unittest.main()
