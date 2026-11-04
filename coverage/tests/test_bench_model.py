"""bench_model.py turns coverage rows into the kits' launches, reports no-variant shapes and unmapped ops, and runs."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

COVERAGE = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(COVERAGE))
import bench_model  # noqa: E402

MODEL_ROOT = os.environ.get('ATLAS_NPU_MODEL_ROOT')
CAN_RUN = bool(MODEL_ROOT) and all(importlib.util.find_spec(name) for name in ('numpy', 'torch', 'npu_model'))


def linear(module, m, k, n, **extra):
    return dict(phase='inference', module=module, source='toy.py:forward:1', op='aten.linear.default',
                inputs=[[[1, m, k], 'float32'], [[n, k], 'float32']], outputs=[[[1, m, n], 'float32']], count=1,
                flops_each=2 * m * k * n, where='atlas:projection_gemm', rule=0, launch=None, **extra)


def toy_model():
    """One Linear the default schedule runs, one width-32 RMSNorm the generator rejects, one unmapped softmax."""
    norm = dict(phase='inference', module='norm', source='toy.py:forward:2', op='aten.rms_norm.default',
                inputs=[[[1, 8, 32], 'float32'], [[32], 'float32']], outputs=[[[1, 8, 32], 'float32']], count=2,
                flops_each=1024, where='atlas:rmsnorm', rule=1,
                launch={'family': 'rmsnorm', 'per': {'op': 'aten.rms_norm.default'}})
    unmapped = dict(phase='inference', module='', source='toy.py:forward:3', op='aten.softmax.int',
                    inputs=[[[1, 4, 8, 8], 'float32']], outputs=[[[1, 4, 8, 8], 'float32']], count=1, flops_each=1280)
    return {'schema': 'atlas.coverage.summary.v1', 'status': 'INCOMPLETE', 'models': {'toy': {
        'status': 'INCOMPLETE', 'kit_launches': {}, 'bench': {'panel_rows': 128},
        'atlas_rows': [linear('qkv', 8, 32, 64), norm], 'unmapped': [unmapped]}}}


class PlanTest(unittest.TestCase):
    def test_focus_models_reproduce_kit_launch_counts(self):
        summary = json.loads((COVERAGE / 'coverage.json').read_text())
        for model, expected in (('tinyllama', dict(projection_panels=840, attention=66, rmsnorm=132)),
                                ('smolvla', dict(projection_panels=2715, attention=200))):
            with self.subTest(model=model):
                launches, uncovered = bench_model.plan(summary['models'][model])
                self.assertEqual(uncovered, [])
                checks = {key: (derived, kit) for key, derived, kit in bench_model.checks(summary['models'][model], launches)}
                self.assertEqual({key: value[0] for key, value in checks.items()}, expected)
                self.assertTrue(all(derived == kit for derived, kit in checks.values()))

    def test_fusions_replace_plain_panels(self):
        summary = json.loads((COVERAGE / 'coverage.json').read_text())
        launches, _ = bench_model.plan(summary['models']['tinyllama'])
        families = {entry['family'] for entry in launches if 'model.layers.*.mlp.gate_proj' in entry['modules']}
        self.assertEqual(families, {'epilogue'})
        decode_o = [entry for entry in launches if 'model.layers.*.self_attn.o_proj' in entry['modules'] and entry['params']['m'] == 32]
        self.assertEqual({entry['family'] for entry in decode_o}, {'projection_gemm'})  # decode residual adds are on the host
        launches, _ = bench_model.plan(summary['models']['smolvla'])
        fused = [entry for entry in launches if entry['family'] == 'smol_fusion']
        self.assertEqual([(entry['count'], entry['params']['m'], entry['params']['k'], entry['params']['n']) for entry in fused],
                         [(192, 128, 768, 3072)])

    def test_panel_split(self):
        tiny = {'panel_rows': 128, 'panel_columns': [{'max_k': 2048, 'n': 2048}, {'n': 1024}]}
        self.assertEqual(sum(bench_model.panels(128, 2048, 32000, tiny).values()), 16)
        self.assertEqual(bench_model.panels(128, 5632, 2048, tiny), {(128, 128, 5632, 1024, 5632, 1024): 2})
        self.assertEqual(bench_model.panels(177, 720, 720, {'panel_rows': 128}),
                         {(128, 128, 736, 736, 720, 720): 1, (49, 64, 736, 736, 720, 720): 1})

    def test_rows_without_a_launch_are_reported(self):
        model = toy_model()['models']['toy']
        orphan = dict(model['atlas_rows'][1], op='aten.mul.Tensor', launch=None, where='atlas:epilogue')
        model['atlas_rows'].append(orphan)
        launches, uncovered = bench_model.plan(model)
        self.assertEqual([row['op'] for row in uncovered], ['aten.mul.Tensor'])
        self.assertEqual(sorted(entry['family'] for entry in launches), ['projection_gemm', 'rmsnorm'])

    def test_plan_only_lists_unmapped_ops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'coverage.json'
            path.write_text(json.dumps(toy_model()))
            self.assertEqual(bench_model.main(['--plan-only', '--coverage', str(path), '--out', str(Path(tmp) / 'out')]), 0)
            text = (Path(tmp) / 'out' / 'MODEL-BENCH.md').read_text()
            self.assertIn('## Unmapped ops (1 rows, 1 instances, 1,280 FLOPs)', text)
            self.assertIn('`aten.softmax.int`', text)

    def test_mismatches_fail(self):
        rows = [dict(key='a', status='ok', bit_exact=False), dict(key='b', status='ok', timed_output_equal=False),
                dict(key='c', status='no-variant'), dict(key='d', status='ok', bit_exact=True, timed_output_equal=True)]
        self.assertEqual(bench_model.failures(rows), ['a: independent golden mismatch', 'b: timed/functional mismatch'])


@unittest.skipUnless(CAN_RUN, 'needs NumPy, PyTorch, and npu_model (ATLAS_NPU_MODEL_ROOT)')
class RunTest(unittest.TestCase):
    def test_runs_reports_no_variant_and_unmapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'coverage.json'
            path.write_text(json.dumps(toy_model()))
            out = Path(tmp) / 'out'
            status = bench_model.main(['--coverage', str(path), '--model-root', MODEL_ROOT, '--out', str(out), '--workers', '4'])
            self.assertEqual(status, 0)
            document = json.loads((out / 'MODEL-BENCH.json').read_text())
            rows = {row['family']: row for row in document['launches']}
            gemm, norm = rows['projection_gemm'], rows['rmsnorm']
            self.assertEqual(gemm['status'], 'ok')
            self.assertTrue(gemm['bit_exact'])
            self.assertTrue(gemm['timed_output_equal'])
            self.assertGreater(gemm['model_cycles'], 0)
            self.assertEqual(gemm['total_cycles'], gemm['model_cycles'])
            self.assertTrue(gemm['variant'].startswith('default schedule'))
            self.assertEqual(norm['status'], 'no-variant')
            self.assertIn("invalid choice: '32'", norm['error'])
            self.assertEqual(norm['count'], 2)
            text = (out / 'MODEL-BENCH.md').read_text()
            self.assertIn('no-variant', text)
            self.assertIn('npu_model model cycles', text)
            self.assertIn('ignore host time, launch overhead', text)
            self.assertIn('`aten.softmax.int`', text)

    def test_generator_rejection_is_no_variant(self):
        entry = dict(key='epilogue:toy', family='epilogue',
                     params=dict(m=32, k=32, n=64, rows=8, logical_k=32, logical_n=64, steps=['silu'], output='bf16'))
        with tempfile.TemporaryDirectory() as tmp:
            result = bench_model.run_job(copy.deepcopy(entry), 'functional', Path(MODEL_ROOT), Path(tmp), 10_000_000, 10_000_000)
        self.assertEqual(result['status'], 'no-variant')
        self.assertIn('traffic reuse is supported only for the qualified shapes', result['error'])


if __name__ == '__main__':
    unittest.main()
