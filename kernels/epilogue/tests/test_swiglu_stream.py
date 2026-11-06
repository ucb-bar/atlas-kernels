"""Original47 exact operator checks for the opt-in contiguous SiLU stream."""
import gzip
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).absolute().parents[3]
sys.path.insert(0, str(ROOT))
import atlas_kernels_path  # noqa: E402,F401
from atlas_execution import Executor  # noqa: E402
from atlas_swiglu_stream import assembly, layout  # noqa: E402


def fp32(words):
    return (np.asarray(words, np.uint16).astype(np.uint32) << 16).view(np.float32)


def rne(value):
    bits = np.asarray(value, np.float32).view(np.uint32)
    return ((bits + np.uint32(32767) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


class SwigluStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.find_spec('npu_model')
        if spec is None:
            raise AssertionError('npu_model is required on PYTHONPATH')
        model = Path(spec.origin).parent.parent
        cls.model = Path(model)
        cls.executor = Executor(cls.model)
        cls.exp, cls.rcp = [np.frombuffer(gzip.decompress(
            (cls.model / 'npu_model/configs/data' / (name + '.bin.gz')).read_bytes()), '<u2')
            for name in ('exp', 'rcp')]

    @classmethod
    def oracle(cls, gate, up):
        # Independent NumPy operations use the admitted original RTL LUT bytes.
        negative = rne(np.maximum(-fp32(gate), fp32(np.uint16(49840))))
        exponential = cls.exp[negative]
        denominator = ((fp32(exponential) + np.float32(1)).view(np.uint32) >> 16).astype(np.uint16)
        silu = rne(fp32(gate) * fp32(cls.rcp[denominator]))
        return rne(fp32(silu) * fp32(up))

    def test_tail_raw_order_complete_arena_guards_and_dma(self):
        rng = np.random.default_rng(29)
        for count in (1, 1023, 1024, 1025, 2053):
            with self.subTest(count=count):
                spec = layout(count)
                gate = np.zeros(spec['padded_elements'], '<u2')
                up = np.zeros_like(gate)
                gate[:count] = rne(rng.uniform(-4, 4, count).astype(np.float32))
                up[:count] = rne(rng.uniform(-2, 2, count).astype(np.float32))
                edges = np.array([0., -0., 1., -1., 16., -16., 96., -96.], np.float32)
                gate[:min(count, len(edges))] = rne(edges[:count])
                expected = self.oracle(gate, up).astype('<u2').tobytes()
                arena = bytearray(b'\xa5' * spec['arena_bytes'])
                size = spec['size']
                a, b, out = (spec[key] for key in ('gate_offset', 'up_offset', 'output_offset'))
                arena[a:a + size] = gate.tobytes()
                arena[b:b + size] = up.tobytes()
                arena[out:out + size] = b'\xcd' * size
                wanted = bytearray(arena)
                wanted[out:out + size] = expected
                source = assembly(count)
                # Positive unsigned immediates cannot silently change signed RV32 arithmetic.
                for line in source.splitlines():
                    fields = re.split(r'[\s,]+', line.strip())
                    if fields[0] in ('addi', 'andi', 'ori', 'xori', 'slti', 'sltiu'):
                        self.assertTrue(-2048 <= int(fields[-1], 0) <= 2047, line)
                program, _ = self.executor.assemble(source)
                self.assertTrue(program.assemble())
                result = self.executor.execute(source, bytes(arena), out, size,
                    expected_arena=bytes(wanted), requested_dma={
                        'load_bytes': 4 * spec['padded_elements'],
                        'store_bytes': 2 * spec['padded_elements']})
                self.assertEqual(result.output, expected)
                self.assertEqual(result.arena, bytes(wanted))
                self.assertFalse(np.any(np.frombuffer(result.output, '<u2')[count:]))

    def test_count_and_dma_window_admission(self):
        for count in (0, -1, True, 1.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                assembly(count)
        for base in (-32, 1, True, 2**32 - 32):
            with self.subTest(base=base), self.assertRaises(ValueError):
                assembly(1024, dram_base=base)
        with self.assertRaises(ValueError):
            assembly(2**32)
        spec = layout(2048)
        base = 2**32 - spec['arena_bytes'] - 32
        self.executor.assemble(assembly(2048, dram_base=base))

    def cli(self, arguments, out):
        return subprocess.run([sys.executable, '-B', str(ROOT / 'kernels/epilogue/gen.py'),
                               '--out', str(out), *arguments], text=True, capture_output=True)

    def test_stream_cli_writes_the_admitted_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'generated'
            result = self.cli(['--swiglu-elements', '1025'], out)
            self.assertEqual(result.returncode, 0, result.stderr)
            import json
            record = json.loads(result.stdout)
            self.assertEqual(record['variant'], 'compact-swiglu-stream')
            self.assertEqual((record['logical_elements'], record['padded_elements']), (1025, 2048))
            self.assertEqual(record['vmem_bytes'], 4096)
            self.assertEqual((out / record['source']).read_text(), assembly(1025))
            _, words_hash = self.executor.assemble(assembly(1025))
            self.assertEqual(hashlib.sha256((out / record['binary']).read_bytes()).hexdigest(), words_hash)

    def test_cli_requires_projection_arguments_and_rejects_stream_projection_flags(self):
        rejected = [[], ['--steps', 'silu'], ['--shape', '128x2048x1536'],
                    ['--shape', '128x2048x1536', '--steps', 'silu', '--swiglu-elements', '1024'],
                    ['--swiglu-elements', '0'], ['--swiglu-elements', '-1'],
                    ['--swiglu-elements', '1.5'], ['--swiglu-elements', str(2**32)]]
        for flag in (['--steps', 'silu'], ['--fp8'], ['--rescale-exponent', '0'],
                     ['--output-scale-exponent', '0'], ['--output-row-stride', '2048'],
                     ['--output-layout', 'tiles']):
            rejected.append(['--swiglu-elements', '1024', *flag])
        for arguments in rejected:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'rejected'
                result = self.cli(arguments, out)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn('Traceback', result.stderr + result.stdout)
                self.assertNotIn('AttributeError', result.stderr + result.stdout)
                self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
