"""The committed coverage table regenerates byte for byte, and an unmapped op fails the build."""
import json
import os
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import quote, unquote

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
    def test_capture_failure_fails_cli_and_not_run_is_distinct(self):
        inventory = json.loads((COVERAGE / 'fixture' / 'tinyllama.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'inventory.json'
            command = [sys.executable, '-B', str(COVERAGE / 'build_table.py'), '--inventory',
                       f'tinyllama={path}', '--out', str(root / 'report'), '--allow-unmapped']
            inventory['capture_crosscheck']['status'] = 'FAIL'
            path.write_text(json.dumps(inventory))
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('capture cross-check FAIL', failed.stderr)
            self.assertFalse((root / 'report').exists())
            for status in ('NOT_RUN', 'PASS'):
                inventory['capture_crosscheck']['status'] = status
                path.write_text(json.dumps(inventory))
                passed = subprocess.run(command, check=True, capture_output=True, text=True)
                report = json.loads((root / 'report' / 'coverage.json').read_text())
                self.assertEqual(report['models']['tinyllama']['capture_crosscheck'], status)
                self.assertIn(f'capture {status}', passed.stdout)
                self.assertIn(f'Merlin capture cross-check: {status}',
                              (root / 'report' / 'COVERAGE.md').read_text())

    def test_citations_require_destinations_and_family_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = [root / repo for repo in ('atlas-mlir', 'merlin', 'atlas-kernels')]
            for path in roots:
                path.mkdir()
            mapping = {'kernels': {'attention': 'attention'}, 'rules': {'example': [
                {'where': 'host:atlas-mlir/missing.c', 'bias_where': 'build:merlin/missing.py'},
                {'where': 'atlas:attention'},
            ]}}
            expected = ['atlas-kernels:kernels/attention/kernel.yaml family attention',
                        'build:merlin/missing.py', 'host:atlas-mlir/missing.c']
            self.assertEqual(build_table.check_cites(mapping, roots), expected)
            (roots[0] / 'missing.c').touch()
            (roots[1] / 'missing.py').touch()
            manifest = roots[2] / 'kernels' / 'attention' / 'kernel.yaml'
            manifest.parent.mkdir(parents=True)
            manifest.write_text('family: wrong\n')
            self.assertEqual(build_table.check_cites(mapping, roots), expected[:1])
            manifest.write_text('family: attention\n')
            self.assertEqual(build_table.check_cites(mapping, roots), [])
            mapping['rules']['example'][0]['where'] = 'host:unknown/missing.c'
            self.assertEqual(build_table.check_cites(mapping, roots), ['host:unknown/missing.c'])

    def test_explicit_missing_citation_checkouts_fail_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = subprocess.run([sys.executable, '-B', str(COVERAGE / 'build_table.py'),
                                     '--out', str(root / 'report'), '--check-cites',
                                     str(root / 'missing'), str(root / 'missing'), str(COVERAGE.parent)],
                                    capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('missing cited file', failed.stderr)
            self.assertFalse((root / 'report').exists())

    def test_committed_table_is_current(self):
        with tempfile.TemporaryDirectory(prefix='coverage report ') as directory:
            source, out = Path(directory) / 'coverage source', Path(directory) / 'report output'
            source.symlink_to(COVERAGE, target_is_directory=True)
            subprocess.run([sys.executable, '-B', str(source / 'build_table.py'), '--out', str(out)], check=True,
                           stdout=subprocess.DEVNULL)
            self.assertEqual((Path(out) / 'coverage.json').read_bytes(), (COVERAGE / 'coverage.json').read_bytes())
            guide = quote(Path(os.path.relpath(source / 'README.md', out)).as_posix())
            self.assertIn('%20', guide)
            expected = (COVERAGE / 'COVERAGE.md').read_text().replace('(README.md#', f'({guide}#')
            self.assertEqual((Path(out) / 'COVERAGE.md').read_text(), expected)
            anchors = ('files', 'how-mappingyaml-relates-to-merlins-selectionyaml', 'regenerate',
                       'adding-a-model', 'limits', 'coverage')
            self.assertEqual(expected.count(f']({guide}#'), len(anchors))
            for anchor in anchors:
                self.assertIn(f']({guide}#{anchor})', expected)
            self.assertEqual((Path(out) / unquote(guide)).resolve(), (COVERAGE / 'README.md').resolve())

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

    def test_export_walk_retains_python_compute_and_batching_wrappers(self):
        import operator
        import torch
        graph = torch.fx.Graph()
        x = graph.placeholder('x')
        x.meta['val'] = torch.zeros(2, 3)
        batch = graph.call_function(torch._C._functorch._add_batch_dim, (x, 0, 1))
        batch.meta['val'] = torch.zeros(3)
        unbatch = graph.call_function(torch._C._functorch._remove_batch_dim, (batch, 1, 2, 0))
        unbatch.meta['val'] = torch.zeros(2, 3)
        neg = graph.call_function(operator.neg, (unbatch,))
        neg.meta['val'] = torch.zeros(2, 3)
        add = graph.call_function(torch.ops.aten.add.Tensor, (neg, x))
        add.meta['val'] = torch.zeros(2, 3)
        graph.output(add)
        records = []
        export_ops.walk(torch.fx.GraphModule(torch.nn.Module(), graph), '', records)
        self.assertEqual([row['op'] for row in records], [
            'torch._C._functorch._add_batch_dim', 'torch._C._functorch._remove_batch_dim',
            '_operator.neg', 'aten.add.Tensor'])
        self.assertEqual([row['flops'] for row in records], [0, 0, 6, 6])
        self.assertEqual(export_ops.flops('dotless', [[[2, 3], 'float32']],
                                         [[[2, 3], 'float32']]), 6)
        self.assertEqual(export_ops.flops('custom.view.compute', [[[2, 3], 'float32']],
                                         [[[2, 3], 'float32']]), 6)

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

    def test_new_model_device_work_is_reported_not_compared(self):
        self.assertEqual(build_table.launch_failures('new_model', {'projection_calls': 6}, {}), [])
        self.assertEqual(build_table.launch_failures('new_model', {'projection_calls': 5}, {'projection_calls': 6}),
                         ['new_model: derived projection_calls 5 differs from qualified build 6'])
        self.assertTrue(build_table.launch_failures('tinyllama', {'projection_calls': 465}, {}))

    def test_atlas_rows_carry_launch_hints(self):
        summary = json.loads((COVERAGE / 'coverage.json').read_text())
        for model, value in summary['models'].items():
            rows = value['atlas_rows']
            self.assertEqual(sum(row['count'] for row in rows),
                             sum(item['ops'] for where, item in value['by_where'].items() if where.startswith('atlas:')), model)
            self.assertTrue(all(row['where'].startswith('atlas:') for row in rows))
        hinted = {row['launch']['family'] for row in summary['models']['tinyllama']['atlas_rows'] if isinstance(row['launch'], dict)}
        self.assertEqual(hinted, {'attention', 'rmsnorm', 'rope', 'rne_add'})

    def test_causal_decoder_example_mapping(self):
        """The starter mapping places Linear, RMSNorm, SiLU, the gate multiply, and residual adds, and nothing else."""
        mapping = yaml.safe_load((COVERAGE / 'examples' / 'causal_decoder' / 'mapping.yaml').read_text())
        rows = [dict(phase='inference', module=module, source=f'loader.py:forward:{line}', op=op, inputs=[],
                     outputs=[[shape, 'float32']], flops_each=1, count=1)
                for module, line, op, shape in (('qkv', 22, 'aten.linear.default', [1, 8, 96]),
                                                ('norm', 22, 'aten.rms_norm.default', [1, 8, 32]),
                                                ('', 30, 'aten.silu.default', [1, 8, 64]),
                                                ('', 30, 'aten.mul.Tensor', [1, 8, 64]),
                                                ('', 30, 'aten.add.Tensor', [1, 8, 32]),
                                                ('', 27, 'aten.add.Tensor', [1, 4, 8, 8]),
                                                ('', 27, 'aten.softmax.int', [1, 4, 8, 8]))]
        placed, unmapped, _ = build_table.place('causal_decoder', {'ops': rows}, mapping['rules']['causal_decoder'], mapping['kernels'])
        self.assertEqual([row['where'] for row in placed], ['atlas:projection_gemm', 'atlas:rmsnorm', 'atlas:epilogue',
                                                            'atlas:epilogue', 'atlas:rne_add'])
        self.assertEqual([row['op'] for row in unmapped], ['aten.add.Tensor', 'aten.softmax.int'])


if __name__ == '__main__':
    unittest.main()
