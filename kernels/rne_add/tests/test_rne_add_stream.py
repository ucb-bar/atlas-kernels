"""Execute the streamed exact add through the pinned physical instruction model."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[3]))
import atlas_kernels_path  # noqa: E402,F401
from atlas_execution import Executor  # noqa: E402
from atlas_rne_add_stream import assembly, flat_geometry  # noqa: E402
from kernel_bench import pack_bf16_tiles  # noqa: E402


class StreamAddTests(unittest.TestCase):
    def test_contiguous_stream_avoids_decode_row_padding(self):
        self.assertEqual(flat_geometry(2048), (32, 64))
        for count in (0, -1, True, 1.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                flat_geometry(count)
        count = 2053
        rows, width = flat_geometry(count)
        size = rows * width
        # Contiguous row-major host buffers need no tile permutation for an
        # elementwise op. Their shared lane interpretation cancels at output.
        rng = np.random.default_rng(12)
        a = np.zeros(size, dtype=np.uint16)
        b = np.zeros(size, dtype=np.uint16)
        values = rng.standard_normal((2, count), dtype=np.float32)
        bits = values.view(np.uint32)
        rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
        a[:count], b[:count] = rounded
        summed = ((a.astype(np.uint32) << 16).view(np.float32)
                  + (b.astype(np.uint32) << 16).view(np.float32)).view(np.uint32)
        expected = ((summed + np.uint32(0x7fff) + ((summed >> 16) & 1)) >> 16).astype('<u2').tobytes()
        left, right = a.astype('<u2').tobytes(), b.astype('<u2').tobytes()
        spec = importlib.util.find_spec('npu_model')
        if spec is None:
            raise AssertionError('npu_model is required on PYTHONPATH')
        root = Path(spec.origin).parent.parent
        arena = left + right + b'\xcd' * len(left) + b'\xa5' * 256
        wanted = left + right + expected + b'\xa5' * 256
        actual = Executor(root).execute(assembly(rows, width), arena, 2 * len(left), len(left),
                                        expected_arena=wanted,
                                        requested_dma=dict(load_bytes=2 * len(left), store_bytes=len(left)))
        self.assertEqual(actual.output, expected)

    def test_shape_and_address_admission(self):
        for rows, width in ((0, 32), (32, 31), (True, 32), (32.0, 32)):
            with self.subTest(rows=rows, width=width), self.assertRaises(ValueError):
                assembly(rows, width)
        for base in (-32, 1, 2**32 - 32):
            with self.subTest(base=base), self.assertRaises(ValueError):
                assembly(32, 32, dram_base=base)

    def test_two_row_tiles_preserve_inputs_and_match_fp32_add_rne(self):
        spec = importlib.util.find_spec('npu_model')
        if spec is None:
            raise AssertionError('npu_model is required on PYTHONPATH')
        root = Path(spec.origin).parent.parent
        rng = np.random.default_rng(7)
        left = rng.integers(0, 65536, (64, 64), dtype=np.uint16)
        right = rng.integers(0, 65536, (64, 64), dtype=np.uint16)
        # Finite BF16 operands include subnormals, ties, cancellation, and overflow.
        left[(left & 0x7f80) == 0x7f80] = 0x7f7f
        right[(right & 0x7f80) == 0x7f80] = 0xff7f
        left[0, :8] = [0x3f80, 0x3f81, 0xbf80, 0, 0x8000, 1, 0x7f7f, 0x3f80]
        right[0, :8] = [0x3b80, 0x3b80, 0x3f80, 0x8000, 0x8000, 1, 0x7f7f, 0xbf80]
        with np.errstate(over='ignore', invalid='ignore'):
            summed = ((left.astype(np.uint32) << 16).view(np.float32)
                      + (right.astype(np.uint32) << 16).view(np.float32))
            bits = summed.view(np.uint32)
            expected = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
        a, b, expected = map(pack_bf16_tiles, (left, right, expected))
        arena = a + b + b'\xcd' * len(a) + b'\xa5' * 256
        reference = a + b + expected + b'\xa5' * 256
        result = Executor(root).execute(assembly(64, 64), arena, 2 * len(a), len(a),
                                        expected_arena=reference,
                                        requested_dma=dict(load_bytes=2 * len(a), store_bytes=len(a)))
        self.assertEqual(result.output, expected)


if __name__ == '__main__':
    unittest.main()
