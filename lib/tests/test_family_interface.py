"""Reject corrupted payloads and preserve the declared independent-golden scope."""
from contextlib import redirect_stderr
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
import family_interface as interface  # noqa: E402
import kernel_bench as bench  # noqa: E402


class GoldenScope(unittest.TestCase):
    def test_missing_arithmetic_remains_pending(self):
        case = SimpleNamespace(golden=None)
        self.assertEqual(interface.compare_golden({}, case, b'output')['status'], 'PENDING')

    def test_byte_difference_and_extent_mismatch_fail(self):
        case = SimpleNamespace(golden=b'ab', output_bytes=2, meta={})
        self.assertEqual(interface.compare_golden({}, case, b'ab')['status'], 'PASS')
        for output in (b'ac', b'a', b'abc'):
            with self.subTest(output=output), self.assertRaises(AssertionError):
                interface.compare_golden({}, case, output)

    def test_rope_zero_sign_exception_does_not_hide_nonzero_padding_or_logical_errors(self):
        values = np.zeros((32, 32), dtype=np.uint16)
        values[0] = 0x3f80
        case = SimpleNamespace(golden=bench.pack_bf16_tiles(values), output_bytes=2048,
                               meta={'golden_logical_rows': 1})
        spec = {'shape': [32, 32, 32]}
        actual = values.copy()
        actual[1, 0] = 0x8000
        result = interface.compare_golden(spec, case, bench.pack_bf16_tiles(actual))
        self.assertEqual((result['status'], result['padding_different_words']), ('PASS', 1))
        for row in (0, 1):
            with self.subTest(row=row), self.assertRaises(AssertionError):
                actual[row, 0] = 0x4000
                interface.compare_golden(spec, case, bench.pack_bf16_tiles(actual))
            actual[row, 0] = values[row, 0]


class InputBinding(unittest.TestCase):
    def manifest(self, target, family):
        (target / 'manifest.json').write_text(json.dumps(dict(family=family, seed=7, artifacts={}, dependencies={})))

    def test_corrupt_source_arena_fp32_and_golden_are_rejected(self):
        case = SimpleNamespace(source='ecall\n', arena=b'inputs', reference=np.array([1], dtype=np.float32),
                               golden=b'golden')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'projection_gemm'
            target.mkdir()
            self.manifest(target, 'projection_gemm')
            for changed in ('program.S', 'input_arena.bin', 'reference.npy', 'golden.bin'):
                with self.subTest(file=changed):
                    (target / 'program.S').write_text(case.source)
                    (target / 'input_arena.bin').write_bytes(case.arena)
                    np.save(target / 'reference.npy', case.reference, allow_pickle=False)
                    (target / 'golden.bin').write_bytes(case.golden)
                    if changed == 'reference.npy':
                        np.save(target / changed, np.array([2], dtype=np.float32), allow_pickle=False)
                    else:
                        (target / changed).write_bytes(b'corrupted')
                    with patch('family_data.build_case', return_value=({}, case)), self.assertRaises(ValueError):
                        interface.validated_case('projection_gemm', root, 7)

    def test_arithmetic_gap_cannot_consume_a_stale_golden(self):
        case = SimpleNamespace(source='ecall\n', arena=b'inputs', reference=np.array([1]), golden=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'attention'
            target.mkdir()
            self.manifest(target, 'attention')
            (target / 'program.S').write_text(case.source)
            (target / 'input_arena.bin').write_bytes(case.arena)
            np.save(target / 'reference.npy', case.reference, allow_pickle=False)
            (target / 'golden.bin').write_bytes(b'device output must not be an oracle')
            with patch('family_data.build_case', return_value=({}, case)), self.assertRaises(ValueError):
                interface.validated_case('attention', root, 7)


class Provisioning(unittest.TestCase):
    def test_generator_rejects_symlink_companions_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'projection_gemm'
            target.mkdir()
            original = root / 'original'
            original.write_bytes(b'untouched')
            (target / 'program.bin').symlink_to(original)
            with patch('family_data.build_case', return_value=({}, SimpleNamespace(source='ecall\n'))):
                with self.assertRaisesRegex(ValueError, 'symbolic-link'):
                    interface.generate('projection_gemm', root, 7)
            self.assertEqual(original.read_bytes(), b'untouched')
            self.assertFalse((target / 'program.S').exists())

    def test_model_compiler_and_rtl_resources_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = ['--family', 'projection_gemm', '--out', str(root / 'out')]
            for arguments in (['test', *base], ['rtl', *base, '--model-root', str(root / 'model'),
                                              '--atlas-mlir-bin', str(root / 'compiler')]):
                with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    interface.main(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_unpinned_simulator_never_reaches_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            simulator = root / 'simulator'
            simulator.write_bytes(b'wrong simulator')
            args = SimpleNamespace(seed=7, family='projection_gemm', out=root, simulator=str(simulator),
                                   build_receipt=str(root / 'build.json'))
            with patch.object(interface, 'validated_case'), patch('rtl_runner.main') as run:
                with self.assertRaisesRegex(ValueError, 'simulator differs'):
                    interface.rtl(args)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
