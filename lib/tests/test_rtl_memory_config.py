"""Check runtime derivation, provenance admission, and event replay failures."""
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import rtl_memory_config as memory
import rtl_validate_memory as validator

MODEL_ROOT = os.environ.get('ATLAS_NPU_MODEL_ROOT') or os.environ.get('NPU_MODEL')
MODEL = memory.checked(MODEL_ROOT) if MODEL_ROOT else None


class DerivationTest(unittest.TestCase):
    def test_header_uses_backend_values_and_hashes_exact_output(self):
        backend = SimpleNamespace(queue_depth=11, window_beats=7, beat_bytes=32,
                                  latency_mode='unloaded', converge=1.0, min_latency=2,
                                  enforce_peak=True, frequency_ghz=0.25, channels=3,
                                  _curves={100: [], 0: []},
                                  latency_at=lambda bw, pct: 12.5 + pct,
                                  curve_peak=lambda pct: (0.125 + pct / 1000, 20))

        @contextmanager
        def environment(root):
            yield None, backend, dict(model_revision='test', source_sha256={'source.py': 'digest'})

        with patch.object(memory, 'model_backend', environment):
            header, metadata = memory.config('unused')
            self.assertIn('atlas_dma_queue_depth = 11;', header)
            self.assertIn('atlas_dma_window_beats = 7;', header)
            self.assertIn('{0, 12.5, 0.125}', header)
            self.assertIn('{100, 112.5, 0.225}', header)
            self.assertEqual(metadata['header_sha256'], hashlib.sha256(header.encode()).hexdigest())
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / 'dma_config.h'
                self.assertEqual(memory.generate('unused', out), metadata)
                self.assertEqual(out.read_text(), header)
                self.assertEqual(json.loads(out.with_suffix('.json').read_text()), metadata)

    def test_modified_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source.py').write_bytes(b'modified')
            with patch.object(memory.subprocess, 'check_output', return_value=b'pinned'):
                with self.assertRaisesRegex(ValueError, 'differs from the pinned revision'):
                    memory._pinned_bytes(root, 'source.py')

    def test_restricted_and_symlink_roots_are_rejected_before_git(self):
        with patch.object(memory.subprocess, 'check_output') as git:
            with self.assertRaisesRegex(ValueError, 'restricted path'):
                memory.inspect_model('/tmp/forbidden-HaMmEr/model')
            with tempfile.TemporaryDirectory() as directory:
                link = Path(directory) / 'model'
                link.symlink_to(directory, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, 'symbolic-link'):
                    memory.inspect_model(link)
            git.assert_not_called()


@unittest.skipUnless(sys.version_info >= (3, 14) and MODEL is not None and MODEL.is_dir(),
                     'pinned model and its Python 3.14 interpreter required')
class PinnedBackendTest(unittest.TestCase):
    def trace_rows(self):
        rows = []
        with memory.model_backend(MODEL) as (module, backend, _):
            # A mixed burst spans the delivered-window boundary and applies
            # D backpressure, exercising both peak pacing and window updates.
            issued = 0
            cycle = 0
            while issued < 80 or backend.outstanding:
                if issued < 80 and backend.can_accept(cycle):
                    opcode = 0 if issued % 3 == 0 else 4
                    request = module.BeatRequest(source=issued % 64, address=0x80000000 + issued * 32,
                                                 is_store=opcode == 0, cycle=cycle)
                    backend.issue(request)
                    due = next(due for due, _, pending in backend._pending if pending is request)
                    rows.append(dict(event='request', absolute_cycle=cycle, source=request.source,
                                     address=request.address, opcode=opcode, due_cycle=due,
                                     accepted_cycle=cycle))
                    issued += 1
                request = backend.response(cycle)
                if request is not None and cycle % 5 == 0:
                    rows.append(dict(event='response', absolute_cycle=cycle, source=request.source,
                                     address=request.address, opcode=0 if request.is_store else 4,
                                     due_cycle=backend._pending[0][0], accepted_cycle=request.cycle))
                    backend.pop_response(cycle)
                cycle += 1
        return rows

    def replay(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / 'trace.csv'
            with trace.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=validator.FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            return validator.validate(MODEL, trace)

    def test_exact_header_interface_matches_previous_extraction(self):
        header, metadata = memory.config(MODEL)
        self.assertEqual(metadata['model_revision'], memory.MODEL_PIN)
        self.assertEqual(len(metadata['source_sha256']), 6)
        self.assertEqual(metadata['params']['queue_depth'], 64)
        self.assertIn('{0, 42.0, 0.04615622105588144}', header)
        self.assertIn('{50, 44.0, 0.04390243902439024}', header)
        self.assertIn('{100, 46.0, 0.04260068389057751}', header)

    def test_mixed_burst_and_delayed_responses(self):
        result = self.replay(self.trace_rows())
        self.assertEqual((result['requests'], result['responses']), (80, 80))
        self.assertEqual(result['peak_pending'], 64)
        self.assertEqual(result['status'], 'PASS')

    def test_wrong_due_cycle_is_rejected(self):
        rows = self.trace_rows()
        rows[0]['due_cycle'] += 1
        with self.assertRaisesRegex(ValueError, 'backend predicts'):
            self.replay(rows)

    def test_wrong_response_identity_is_rejected(self):
        rows = self.trace_rows()
        next(row for row in rows if row['event'] == 'response')['source'] += 1
        with self.assertRaisesRegex(ValueError, 'response identity'):
            self.replay(rows)

    def test_missing_response_is_rejected(self):
        rows = self.trace_rows()
        rows.pop()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.replay(rows)

    def test_imported_model_modules_are_restored(self):
        sentinel = SimpleNamespace(marker=True)
        with patch.dict(sys.modules, {'npu_model': sentinel}):
            memory.inspect_model(MODEL)
            self.assertIs(sys.modules['npu_model'], sentinel)


if __name__ == '__main__':
    unittest.main()
