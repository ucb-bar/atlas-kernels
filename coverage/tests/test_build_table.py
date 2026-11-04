"""The committed coverage table regenerates byte for byte, and an unmapped op fails the build."""
import json
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

COVERAGE = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(COVERAGE))
try:
    import yaml
    import build_table
    import export_ops
except ImportError:
    yaml = None


@unittest.skipUnless(yaml, 'build_table.py requires PyYAML')
class BuildTableTest(unittest.TestCase):
    def test_committed_table_is_current(self):
        with tempfile.TemporaryDirectory() as out:
            subprocess.run([sys.executable, '-B', str(COVERAGE / 'build_table.py'), '--out', out], check=True,
                           stdout=subprocess.DEVNULL)
            for name in ('COVERAGE.md', 'coverage.json'):
                self.assertEqual((Path(out) / name).read_text(), (COVERAGE / name).read_text(), name)

    def test_every_rule_is_needed(self):
        mapping = yaml.safe_load((COVERAGE / 'mapping.yaml').read_text())
        for model in build_table.MODELS:
            inventory = json.loads((COVERAGE / 'fixture' / f'{model}.json').read_text())
            rules = mapping['rules'][model]
            baseline, unmapped, hits = build_table.place(model, inventory, rules, mapping['kernels'])
            self.assertEqual(unmapped, [])
            placements = lambda rows: [(row['op'], row.get('component'), row['where']) for row in rows]
            for index in range(len(rules)):
                self.assertGreater(hits[index], 0, f'{model} rule {index}')
                changed, unmapped, _ = build_table.place(model, inventory, rules[:index] + rules[index + 1:], mapping['kernels'])
                self.assertTrue(unmapped or placements(changed) != placements(baseline), f'{model} rule {index}')

    def test_device_work_matches_kit_launches(self):
        summary = json.loads((COVERAGE / 'coverage.json').read_text())
        for model, value in summary['models'].items():
            self.assertEqual(value['derived_launches'], build_table.REQUIRED_LAUNCHES[model])
            self.assertEqual(build_table.launch_failures(model, value['derived_launches'], value['kit_launches']), [])

    def test_absent_device_category_fails(self):
        mapping = yaml.safe_load((COVERAGE / 'mapping.yaml').read_text())
        for model, expected in build_table.REQUIRED_LAUNCHES.items():
            inventory = json.loads((COVERAGE / 'fixture' / f'{model}.json').read_text())
            placed, _, _ = build_table.place(model, inventory, mapping['rules'][model], mapping['kernels'])
            for key in expected:
                with self.subTest(model=model, category=key):
                    # Move the complete category to the host, reproducing the missing-key regression.
                    mutated = copy.deepcopy(placed)
                    for row in mutated:
                        if key in build_table.derived_launches([row]):
                            row['kind'] = 'host'
                    derived = build_table.derived_launches(mutated)
                    self.assertNotIn(key, derived)
                    failures = build_table.launch_failures(model, derived, mapping['builds'][model]['launches'])
                    self.assertTrue(any(f'derived {key} 0' in failure for failure in failures), failures)
                    changed_expected = {name: count for name, count in expected.items() if name != key}
                    self.assertTrue(build_table.launch_failures(model, derived, changed_expected))

    def test_projection_bias_components_have_complete_mixed_placement(self):
        mapping = yaml.safe_load((COVERAGE / 'mapping.yaml').read_text())
        inventory = json.loads((COVERAGE / 'fixture' / 'smolvla.json').read_text())
        placed, unmapped, _ = build_table.place('smolvla', inventory, mapping['rules']['smolvla'], mapping['kernels'])
        self.assertEqual(unmapped, [])
        bias = [row for row in placed if row.get('component') == 'bias']
        totals = {kind: sum(row['count'] * row['flops_each'] for row in bias if row['kind'] == kind)
                  for kind in ('host', 'atlas')}
        self.assertEqual(totals, {'host': 97041664, 'atlas': 75497472})
        self.assertEqual(sum(row['count'] * row['flops_each'] for row in placed), inventory['flops'])
        self.assertTrue(all('calls' not in row for row in bias))
        self.assertTrue(all(row['where'] == 'atlas:smol_fusion' and row['module'].endswith('.mlp.fc1')
                            for row in bias if row['kind'] == 'atlas'))
        for row in inventory['ops']:
            self.assertEqual(export_ops.flops(row['op'], row['inputs'], row['outputs']), row['flops_each'])

    def test_unmapped_report_requires_explicit_opt_in(self):
        inventory = json.loads((COVERAGE / 'fixture' / 'tinyllama.json').read_text())
        inventory['model'] = 'unregistered'
        inventory['ops'] = [dict(inventory['ops'][0], flops_each=10, count=2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'inventory.json'
            path.write_text(json.dumps(inventory))
            command = [sys.executable, '-B', str(COVERAGE / 'build_table.py'), '--inventory',
                       f'unregistered={path}', '--out', str(root / 'report')]
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('unmapped', failed.stderr)
            self.assertFalse((root / 'report' / 'coverage.json').exists())
            subprocess.run(command + ['--allow-unmapped'], check=True, stdout=subprocess.DEVNULL)
            report = json.loads((root / 'report' / 'coverage.json').read_text())
            self.assertEqual(report['status'], 'INCOMPLETE')
            self.assertEqual(report['models']['unregistered']['flops']['unmapped'], 20)
            self.assertEqual(len(report['models']['unregistered']['unmapped']), 1)
            self.assertIn('**INCOMPLETE: 1 unmapped rows', (root / 'report' / 'COVERAGE.md').read_text())

    def test_allow_unmapped_does_not_bypass_launch_gate(self):
        mapping = yaml.safe_load((COVERAGE / 'mapping.yaml').read_text())
        for rule in mapping['rules']['tinyllama']:
            if rule['where'] == 'atlas:rmsnorm':
                rule['where'] = 'host:atlas-mlir/tools/tiny_full/tiny_full_model.c'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'mapping.yaml'
            path.write_text(yaml.safe_dump(mapping))
            failed = subprocess.run([sys.executable, '-B', str(COVERAGE / 'build_table.py'), '--mapping', str(path),
                                     '--out', str(root), '--allow-unmapped'], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('derived rmsnorm 0 differs from qualified build 132', failed.stderr)


if __name__ == '__main__':
    unittest.main()
