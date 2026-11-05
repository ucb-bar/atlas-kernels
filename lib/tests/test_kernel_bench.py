"""Mutated execution outputs must fail the bench command and remain visible in receipts."""
from concurrent.futures import Future
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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


class InlinePool:
    """Exercise the real driver and run function without subprocess mock inheritance."""

    def __init__(self, *args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, fn, *args):
        future = Future()
        future.set_result(fn(*args))
        return future


class KernelBenchFailuresTest(unittest.TestCase):
    def run_mock_bench(self, functional=b'\x01', timed_output=b'\x01', golden=b'\x01', timed=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'kernels' / 'mock').mkdir(parents=True)
            (root / 'coverage').mkdir()
            case = bench.Case('ecall', b'\x00\x01\x00', 1, 1, np.array([1.]),
                              lambda data: np.frombuffer(data, dtype=np.uint8), golden,
                              'independent' if golden is not None else None)
            module = SimpleNamespace(CASES=[{'name': 'sample', 'model': 'test', 'shape': [1]}],
                                     build=lambda spec, rng: case)
            execution = SimpleNamespace(output=functional, program_words=1, instructions=1,
                                        model={'revision': 'mock', 'numerics': 'rtl'})
            argv = ['kernel_bench', '--model-root', str(root), '--out', str(root / 'out'), '--family', 'mock']
            if timed:
                argv.append('--timed')
            with patch.object(bench, 'ROOT', root), patch.object(bench, 'load_reference', return_value=module), \
                    patch.object(bench, 'ProcessPoolExecutor', InlinePool), \
                    patch('atlas_execution.Executor') as executor, \
                    patch.object(bench, 'timed_cycles', return_value=(4, b'\x00' + timed_output + b'\x00')), \
                    patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                executor.return_value.execute.return_value = execution
                status = bench.main()
            receipt = json.loads((root / 'out' / 'mock-sample.json').read_text())
            report = (root / 'coverage' / 'RESULTS.md').read_text()
            return status, receipt, report, err.getvalue()

    def test_functional_mutation_fails_independent_golden(self):
        status, receipt, report, stderr = self.run_mock_bench(functional=b'\x02', timed_output=b'\x02')
        self.assertEqual(status, 1)
        self.assertIs(receipt['bit_exact'], False)
        self.assertIs(receipt['timed_output_equal'], True)
        self.assertIn('independent golden mismatch', stderr)
        self.assertIn('DIFFERS | exact', report)

    def test_timed_mutation_fails_with_matching_golden(self):
        status, receipt, report, stderr = self.run_mock_bench(timed_output=b'\x02')
        self.assertEqual(status, 1)
        self.assertIs(receipt['bit_exact'], True)
        self.assertIs(receipt['timed_output_equal'], False)
        self.assertIn('timed/functional mismatch', stderr)
        self.assertIn('exact | DIFFERS', report)

    def test_timed_mutation_fails_without_golden(self):
        status, receipt, report, stderr = self.run_mock_bench(timed_output=b'\x02', golden=None)
        self.assertEqual(status, 1)
        self.assertIsNone(receipt['bit_exact'])
        self.assertIn('Cases without an independent golden: `mock/sample`.', report)
        self.assertIn('timed/functional mismatch', stderr)

    def test_matching_outputs_pass_and_untimed_is_explicit(self):
        self.assertEqual(self.run_mock_bench()[0], 0)
        status, receipt, report, stderr = self.run_mock_bench(timed=False)
        self.assertEqual(status, 0)
        self.assertNotIn('timed_output_equal', receipt)
        self.assertIn('exact | not run', report)
        self.assertEqual(stderr, '')

    def test_empty_case_selection_is_rejected_before_pool(self):
        module = SimpleNamespace(CASES=[{'name': 'present'}])
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(bench, 'load_reference', return_value=module), \
                patch.object(bench, 'ProcessPoolExecutor') as pool, \
                patch.object(sys, 'argv', ['kernel_bench', '--model-root', directory, '--out', directory,
                                          '--family', 'mock', '--case', 'absent']), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_status:
                bench.main()
            self.assertEqual(exit_status.exception.code, 2)
            pool.assert_not_called()

    def test_empty_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'no benchmark results'):
                bench.report(directory)

    def test_unchanged_report_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'RESULTS.md'
            bench.write_if_changed(path, 'same\n')
            with patch.object(Path, 'write_text') as write:
                bench.write_if_changed(path, 'same\n')
            write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
