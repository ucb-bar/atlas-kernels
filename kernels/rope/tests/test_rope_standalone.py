"""Standalone BF16 RoPE exactness and staging admission, without captured data."""
import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).absolute().parents[3]
sys.path.insert(0, str(ROOT))
import atlas_kernels_path  # noqa: E402,F401
from atlas_execution import Executor  # noqa: E402
from atlas_rope_standalone import assembly, layout, prepare  # noqa: E402
from kernel_bench import pack_bf16_tiles  # noqa: E402


def bf(value):
    bits = np.asarray(value, np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def fp(words):
    return (np.asarray(words, np.uint16).astype(np.uint32) << 16).view(np.float32)


def oracle(h, cosine, sine):
    x = fp(h).reshape(len(h), -1, 64)
    c, s = fp(cosine)[:, None, :], fp(sine)[:, None, :]
    lo, hi = x[..., :32], x[..., 32:]
    # Independent host FP32 operations with explicit BF16 rounding points.
    low = bf(fp(bf(lo*c)) - fp(bf(hi*s)))
    high = bf(fp(bf(hi*c)) + fp(bf(lo*s)))
    return np.concatenate((low, high), axis=-1).reshape(h.shape)


class StandaloneRopeTests(unittest.TestCase):
    def inputs(self, m=64, n=128, rows=35):
        rng = np.random.default_rng(29)
        h = np.zeros((m, n), np.uint16)
        h[:rows] = bf(rng.standard_normal((rows, n), dtype=np.float32)*4)
        h[-1, 0] = 0x8000  # Signed-zero padding is admitted and preserved.
        h[0, :8] = [0, 0x8000, 0x3f80, 0x3f81, 0xbf80, 1, 0x8001, 0x3b80]
        angles = np.arange(rows*32, dtype=np.float32).reshape(rows, 32)*np.float32(0.03125)
        c = np.full((m, 32), 0x3f80, np.uint16)
        s = np.zeros((m, 32), np.uint16)
        c[:rows], s[:rows] = bf(np.cos(angles)), bf(np.sin(angles))
        return h, c, s, s ^ np.uint16(0x8000)

    def test_original47_independent_rounding_tails_guards_and_dma(self):
        spec = importlib.util.find_spec('npu_model')
        if spec is None:
            raise AssertionError('npu_model is required on PYTHONPATH')
        root = Path(spec.origin).parent.parent
        for m, n, rows in ((32, 64, 31), (64, 128, 35)):
            with self.subTest(m=m, n=n, rows=rows):
                h, c, s, neg = self.inputs(m, n, rows)
                staged = prepare(m, n, *(pack_bf16_tiles(v) for v in (h, c, s, neg)), logical_rows=rows)
                expected = pack_bf16_tiles(oracle(h, c, s))
                initial, offset = staged['initial'], staged['output_offset']
                self.assertEqual(staged['written_spans'], [dict(offset=offset, bytes=len(expected))])
                wanted = initial[:offset] + expected + initial[offset+len(expected):]
                actual = Executor(root).execute(staged['source'], initial, offset, len(expected),
                    expected_arena=wanted, requested_dma=dict(load_bytes=staged['dma_load_bytes'],
                                                              store_bytes=staged['dma_store_bytes']))
                self.assertEqual(actual.output, expected)
                self.assertEqual(actual.arena, wanted)
                self.assertEqual(actual.observed['load_bytes'], 2*m*n+192*m)
                self.assertEqual(actual.observed['store_bytes'], 2*m*n)
                self.assertEqual(initial[2*m*n:offset], b'\xa5'*256)
                self.assertEqual(initial[offset+len(expected):offset+len(expected)+256], b'\x5a'*256)

    def test_geometry_address_and_table_admission(self):
        for m, n in ((0, 64), (16, 64), (32, 32), (True, 64), (32.0, 64), (512, 2048), (1376, 64)):
            with self.subTest(m=m, n=n), self.assertRaises(ValueError):
                assembly(m, n)
        for base in (-64, 1, 2**32-64, True):
            with self.subTest(base=base), self.assertRaises(ValueError):
                layout(32, 64, dram_base=base)
        h, c, s, neg = self.inputs()
        def stage(h=h, c=c, s=s, neg=neg, rows=35):
            return prepare(64, 128, *(pack_bf16_tiles(v) for v in (h, c, s, neg)), logical_rows=rows)
        stage()
        for rows in (0, 65, True):
            with self.subTest(rows=rows), self.assertRaises(ValueError):stage(rows=rows)
        for which, matrix, at, value in (('h', h, (63, 127), 0x3f80),
                ('h', h, (0, 0), 0x7f80), ('c', c, (63, 0), 0),
                ('s', s, (63, 0), 0x3f80), ('neg', neg, (0, 0), 0),
                ('c', c, (0, 0), 0x4000)):
            bad = matrix.copy();bad[at] = value
            with self.subTest(which=which, at=at, value=value), self.assertRaises(ValueError):
                stage(**{which:bad})
        bad_s = s.copy();bad_s[63, 0] = 0x3f80
        with self.assertRaisesRegex(ValueError, 'padded coefficient'):
            stage(s=bad_s, neg=bad_s ^ np.uint16(0x8000))
        with self.assertRaises(ValueError):
            prepare(32, 64, b'', bytes(2048), bytes(2048), bytes(2048))

    def test_cli_opt_in_is_exclusive_and_has_no_rescale(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'invalid'
            base = [sys.executable, '-B', str(ROOT/'kernels/rope/gen.py'), '--out', str(out)]
            for options in (['--shape', '128x2048x2048', '--postpass-shape', '64x320'],
                            ['--postpass-shape', '64x320', '--rescale-exponent', '1'],
                            ['--postpass-shape', '16x64'],
                            ['--postpass-shape', '32x32'],
                            ['--postpass-shape', '32x64x32'],
                            ['--shape', '32x64']):
                result = subprocess.run(base+options, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn(b'Traceback', result.stderr)
                self.assertFalse(out.exists())
            subprocess.run(base+['--postpass-shape', '64x320', '--name', 'standalone'], check=True, capture_output=True)
            metadata = json.loads((out/'standalone.json').read_text())
            self.assertEqual(metadata['variant'], 'standalone-bf16-postpass')
            self.assertEqual((metadata['m'], metadata['n']), (64, 320))
            self.assertEqual((out/'standalone.S').read_text(), assembly(64, 320))


if __name__ == '__main__':
    unittest.main()
