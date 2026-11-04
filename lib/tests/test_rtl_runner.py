"""Report incomplete qualification and mismatches without hiding failed families."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
import rtl_runner  # noqa: E402


class RtlReportTest(unittest.TestCase):
    def test_unrun_families_remain_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            rows = rtl_runner.report(out, out / 'report.md')
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(row['status'] == 'PENDING' for row in rows))
            self.assertEqual(json.loads((out / 'summary.json').read_text())['status'], 'INCOMPLETE_OR_FAILED')
            self.assertIn('| resnet_patch_gather | PENDING', (out / 'report.md').read_text())

    def test_mismatch_and_reduction_are_visible(self):
        row = dict(family='rope', shape=[32, 32, 960], reduced=True,
                   matched_rtl_cycles=79, synthetic_rtl_cycles=100, npu_model_cycles=80,
                   model_minus_matched_rtl_cycles=1, exact=False,
                   rtl_wall_seconds=1.5, rtl_revision='UNVERIFIED', status='FAIL')
        text = rtl_runner.table([row])
        self.assertIn('32x32x960 (reduced)', text)
        self.assertIn('| no | 80 | 79 | 1 | 100 | UNVERIFIED |', text)

    def test_custom_empty_output_is_rejected(self):
        from contextlib import redirect_stderr
        import io
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            argv = ['rtl_runner', '--out', directory, '--program', 'program.S',
                    '--input', 'input.bin', '--output-offset', '0', '--output-bytes', '0',
                    '--family', 'projection_gemm']
            with patch.object(sys, 'argv', argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                rtl_runner.main()
            self.assertEqual(error.exception.code, 2)

    def test_revision_cannot_mislabel_hash_checked_build(self):
        from contextlib import redirect_stdout
        import io
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            simulator = root / 'simulator'
            simulator.write_bytes(b'simulator')
            receipt = root / 'build.json'
            receipt.write_text(json.dumps(dict(binary_sha256=rtl_runner.digest(simulator),
                                               rtl_revision='a' * 40, returncode=0)))
            program, arena = root / 'program.S', root / 'input.bin'
            program.write_text('ecall\n')
            arena.write_bytes(b'0')
            args = SimpleNamespace(out=root, simulator=simulator, rtl_revision='b' * 40,
                                   build_receipt=receipt, program=program, input=arena,
                                   output_offset=0, output_bytes=1, shape='test')
            with patch('atlas_execution.Executor') as executor, redirect_stdout(io.StringIO()):
                row = rtl_runner.run_case(args, 'projection_gemm')
            executor.assert_not_called()
            self.assertEqual(row['status'], 'FAIL')
            self.assertIn('RTL revision does not match build provenance', row['error'])

    def test_combined_report_requires_both_memories_and_same_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            row = dict(family='resnet_pool', status='PASS', exact=True, shape=[32, 32],
                       rtl_revision='pin', simulator_sha256='sim', program_sha256='program',
                       input_sha256='input', model_output_sha256='output', npu_model_cycles=81,
                       rtl_cycles=80, output_bytes=32)
            matched = out / 'npu-model' / 'resnet_pool'
            matched.mkdir(parents=True)
            (matched / 'result.json').write_text(json.dumps(row))
            rows = rtl_runner.report(out, out / 'report.md')
            pool = next(r for r in rows if r['family'] == 'resnet_pool')
            self.assertEqual(pool['status'], 'PENDING')
            self.assertNotIn('exact', pool)
            synthetic = out / 'synthetic' / 'resnet_pool'
            synthetic.mkdir(parents=True)
            (synthetic / 'result.json').write_text(json.dumps(dict(row, rtl_cycles=40)))
            rows = rtl_runner.report(out, out / 'report.md')
            pool = next(r for r in rows if r['family'] == 'resnet_pool')
            self.assertEqual((pool['status'], pool['matched_rtl_cycles'],
                              pool['synthetic_rtl_cycles'], pool['model_minus_matched_rtl_cycles']),
                             ('PASS', 80, 40, 1))
            (synthetic / 'result.json').write_text(json.dumps(dict(row, input_sha256='different')))
            rows = rtl_runner.report(out, out / 'report.md')
            pool = next(r for r in rows if r['family'] == 'resnet_pool')
            self.assertEqual(pool['status'], 'FAIL')
            self.assertIn('different experiment identities', pool['error'])

    def test_missing_simulator_leaves_failed_case_receipt(self):
        from types import SimpleNamespace
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            args = SimpleNamespace(out=out, simulator=out / 'missing', rtl_revision='pin')
            with redirect_stdout(io.StringIO()):
                row = rtl_runner.run_case(args, 'resnet_pool')
            self.assertEqual(row['status'], 'FAIL')
            self.assertIn('FileNotFoundError', row['error'])
            self.assertTrue((out / 'synthetic' / 'resnet_pool' / 'result.json').is_file())

    def test_successful_single_memory_run_does_not_fail_for_missing_other_memory(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            argv = ['rtl_runner', '--out', directory, '--memory', 'synthetic',
                    '--simulator', 'sim', '--build-receipt', 'receipt', '--rtl-revision', 'pin',
                    '--model-root', 'model', '--atlas-mlir-bin', 'compiler']
            with patch.object(sys, 'argv', argv), patch.object(rtl_runner, 'run_case',
                    return_value=dict(status='PASS')) as run:
                self.assertEqual(rtl_runner.main(), 0)
            self.assertEqual(run.call_count, 9)

    def test_execution_requires_build_provenance(self):
        from unittest.mock import patch
        from contextlib import redirect_stderr
        import io
        with tempfile.TemporaryDirectory() as directory:
            argv = ['rtl_runner', '--out', directory, '--simulator', 'sim',
                    '--rtl-revision', 'pin', '--model-root', 'model', '--atlas-mlir-bin', 'compiler']
            with patch.object(sys, 'argv', argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                rtl_runner.main()
            self.assertEqual(error.exception.code, 2)

    def test_actual_model_shape_is_not_marked_reduced(self):
        row = dict(family='projection_gemm', shape=[32, 32, 960], reduced=False,
                   rtl_revision='UNVERIFIED', status='PENDING')
        self.assertNotIn('(reduced)', rtl_runner.table([row]))


if __name__ == '__main__':
    unittest.main()
