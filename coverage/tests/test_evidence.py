"""Historical recovery preserves measurements and refuses corrupted evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

COVERAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COVERAGE))
import bench_model
import evidence


class EvidenceTest(unittest.TestCase):
    def test_variant_measurements_match_authenticated_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name in evidence.VARIANTS:
                with self.subTest(variant=name):
                    target = evidence.recover_variant(name, output)
                    manifest = json.loads((COVERAGE / f'{name}.json').read_text())
                    self.assertEqual(evidence.summarize_variant(json.loads(target.read_bytes())),
                                     manifest['summary'])
                    target.write_bytes(target.read_bytes() + b' ')
                    with self.assertRaisesRegex(FileExistsError, 'existing artifact differs'):
                        evidence.recover_variant(name, output)

    def test_recovered_reports_preserve_every_historical_table_row(self):
        expected = {'tinyllama': (26, 1038, 13, 3537507930),
                    'smolvla': (28, 2915, 23, 5160679740)}
        with tempfile.TemporaryDirectory() as temporary:
            for model, totals in expected.items():
                summary, destination = evidence.recover(model, Path(temporary))
                full = json.loads((destination / 'MODEL-BENCH.json').read_text())
                rows = full['launches']
                self.assertEqual((len(rows), sum(r['count'] for r in rows),
                                  sum(r['golden'] is not None for r in rows),
                                  sum(r['total_cycles'] for r in rows)), totals)
                # Compare all tables against the original full receipt's renderer,
                # including family shares, raw metrics, padding notes, and tiers.
                tables = lambda text: [line for line in text.splitlines() if line.startswith('|')]
                self.assertEqual(tables(bench_model.render(full)), tables(bench_model.render(summary)))
                committed = COVERAGE / 'examples' / model / 'MODEL-BENCH.md'
                self.assertEqual(committed.read_text(), bench_model.render(summary))

    def test_changed_cached_artifact_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            _, destination = evidence.recover('tinyllama', output)
            artifact = destination / 'MODEL-BENCH.json'
            artifact.write_bytes(artifact.read_bytes() + b' ')
            with self.assertRaisesRegex(FileExistsError, 'existing artifact differs'):
                evidence.recover('tinyllama', output)

    def test_corrupted_manifest_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'coverage' / 'examples' / 'tinyllama'
            shutil.copytree(COVERAGE / 'examples' / 'tinyllama', directory)
            path = directory / 'MODEL-BENCH.artifact.json'
            manifest = json.loads(path.read_text())
            manifest['artifact']['sha256'] = '0' * 64
            path.write_text(json.dumps(manifest))
            with patch.object(evidence, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    evidence.recover('tinyllama', root / 'out', repo=COVERAGE.parent)

    def test_rehashed_falsified_summary_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'coverage' / 'examples' / 'tinyllama'
            shutil.copytree(COVERAGE / 'examples' / 'tinyllama', directory)
            path = directory / 'MODEL-BENCH.summary.json'
            summary = json.loads(path.read_text())
            summary['launches'][0]['model_cycles'] += 1
            data = json.dumps(summary).encode()
            path.write_bytes(data)
            path = directory / 'MODEL-BENCH.artifact.json'
            manifest = json.loads(path.read_text())
            manifest['summary'].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            path.write_text(json.dumps(manifest))
            with patch.object(evidence, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'compact measurements differ'):
                    evidence.recover('tinyllama', root / 'out', repo=COVERAGE.parent)


if __name__ == '__main__':
    unittest.main()
