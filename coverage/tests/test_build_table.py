"""The committed coverage table regenerates byte for byte, and an unmapped op fails the build."""
import json
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
            _, unmapped, _ = build_table.place(model, inventory, rules, mapping['kernels'])
            self.assertEqual(unmapped, [])
            for index in (0, len(rules) - 1):
                _, unmapped, _ = build_table.place(model, inventory, rules[:index] + rules[index + 1:], mapping['kernels'])
                self.assertTrue(unmapped, f'{model} rule {index}')

    def test_device_work_matches_kit_launches(self):
        summary = json.loads((COVERAGE / 'coverage.json').read_text())
        for model, value in summary['models'].items():
            for key, count in value['derived_launches'].items():
                self.assertEqual(count, value['kit_launches'][key], f'{model} {key}')


if __name__ == '__main__':
    unittest.main()
