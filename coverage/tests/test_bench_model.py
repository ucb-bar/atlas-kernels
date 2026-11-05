"""bench_model.py turns coverage rows into the kits' launches, reports no-variant shapes and unmapped ops, and runs."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

COVERAGE = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(COVERAGE))
import bench_model  # noqa: E402

MODEL_ROOT = os.environ.get('ATLAS_NPU_MODEL_ROOT')
CAN_RUN = bool(MODEL_ROOT) and Path(MODEL_ROOT).is_dir() and all(
    importlib.util.find_spec(name) for name in ('numpy', 'torch', 'npu_model'))
if 'ATLAS_NPU_MODEL_ROOT' in os.environ and not CAN_RUN:
    raise RuntimeError('ATLAS_NPU_MODEL_ROOT requires a model directory and importable NumPy, PyTorch, and npu_model')
MLIR_BIN = os.environ.get('ATLAS_MLIR_BIN')
CAN_COMPILE = bool(MLIR_BIN) and all((Path(MLIR_BIN) / name).is_file() for name in ('atlas-opt', 'atlas-emit'))
if 'ATLAS_MLIR_BIN' in os.environ and not CAN_COMPILE:
    raise RuntimeError('ATLAS_MLIR_BIN requires a nonempty directory containing atlas-opt and atlas-emit')


def linear(module, m, k, n, **extra):
    return dict(phase='inference', module=module, source='toy.py:forward:1', op='aten.linear.default',
                inputs=[[[1, m, k], 'float32'], [[n, k], 'float32']], outputs=[[[1, m, n], 'float32']], count=1,
                flops_each=2 * m * k * n, where='atlas:projection_gemm', rule=0, launch=None, **extra)


def toy_model():
    """One Linear the default schedule runs, one width-33 RMSNorm the generator rejects, one unmapped softmax."""
    norm = dict(phase='inference', module='norm', source='toy.py:forward:2', op='aten.rms_norm.default',
                inputs=[[[1, 8, 33], 'float32'], [[33], 'float32']], outputs=[[[1, 8, 33], 'float32']], count=2,
                flops_each=1024, where='atlas:rmsnorm', rule=1,
                launch={'family': 'rmsnorm', 'per': {'op': 'aten.rms_norm.default'}})
    unmapped = dict(phase='inference', module='', source='toy.py:forward:3', op='aten.softmax.int',
                    inputs=[[[1, 4, 8, 8], 'float32']], outputs=[[[1, 4, 8, 8], 'float32']], count=1, flops_each=1280)
    return {'schema': 'atlas.coverage.summary.v1', 'status': 'INCOMPLETE', 'models': {'toy': {
        'status': 'INCOMPLETE', 'kit_launches': {}, 'bench': {'panel_rows': 128},
        'atlas_rows': [linear('qkv', 8, 32, 64), norm], 'unmapped': [unmapped]}}}


def job_args(directory, **overrides):
    root = Path(directory)
    defaults = dict(out=root, resume=True, atlas_mlir_bin=root / 'bin', min_tier='stream-verified',
                    backend_tools={}, model_root=root / 'model', workers=2, max_cycles=100,
                    max_instructions=100, job_timeout=1, subset_after=1)
    return SimpleNamespace(**(defaults | overrides))


def job_destination(command):
    return command[command.index('--mode') + 1], Path(command[command.index('--result') + 1])


def write_cached_job(args, entry, mode, result):
    jobs = args.out / 'jobs'
    jobs.mkdir(exist_ok=True)
    name = hashlib.sha256((entry['key'] + mode).encode()).hexdigest()[:16]
    (jobs / f'{name}.{mode}.json').write_text(json.dumps(result))


def execute_jobs(entry, args, callback, context=None):
    with mock.patch.object(bench_model.subprocess, 'run', side_effect=callback) as process, \
            mock.patch.object(bench_model, 'cache_context', return_value=context or {}), \
            mock.patch('builtins.print'):
        return bench_model.execute([entry], args), process


class PlanTest(unittest.TestCase):
    def test_explicit_configuration_rejects_missing_prerequisites(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(importlib.util, 'find_spec', return_value=None):
            optional = runpy.run_path(__file__)
            self.assertFalse(optional['CAN_RUN'] or optional['CAN_COMPILE'])
            for variable in ('ATLAS_NPU_MODEL_ROOT', 'ATLAS_MLIR_BIN'):
                for value in ('', '/nonexistent', str(COVERAGE)):
                    with self.subTest(variable=variable, value=value), mock.patch.dict(os.environ, {variable: value}):
                        with self.assertRaisesRegex(RuntimeError, variable):
                            runpy.run_path(__file__)

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

    def test_subset_compiler_error_is_a_launch_failure(self):
        entry = dict(key='attention:toy', family='attention', count=2,
                     params=dict(rows=256, keys=32, heads=1, kv_heads=1, head_dim=64,
                                 mask='none', output_fp8=False))
        error = 'BackendError: program tier encode-only is below required verified'
        results = {entry['key']: dict(functional=dict(status='ok'),
                                     timed=dict(status='timeout', seconds=1),
                                     subset=dict(rows=128, factor=2, result=dict(status='error', error=error)))}
        row, = bench_model.merge([entry], results)
        self.assertEqual(row['status'], 'error')
        self.assertIn(error, row['error'])
        self.assertTrue(bench_model.failures([row]))
        self.assertNotIn('total_cycles', row)

    def test_no_variant_does_not_hide_other_job_error(self):
        entry, = [item for item in bench_model.plan(toy_model()['models']['toy'])[0]
                  if item['family'] == 'projection_gemm']
        error = 'BackendError: atlas-emit words differ from the assembler'
        for functional, timed in ((dict(status='no-variant', error='unsupported shape'),
                                   dict(status='error', error=error)),
                                  (dict(status='error', error=error),
                                   dict(status='no-variant', error='unsupported shape'))):
            with self.subTest(functional_status=functional['status']):
                row, = bench_model.merge([entry], {entry['key']: dict(functional=functional, timed=timed)})
                self.assertEqual(row['status'], 'error')
                self.assertIn(error, row['error'])
                self.assertTrue(bench_model.failures([row]))

    def test_resume_reruns_incomplete_or_compilerless_timeout_jobs(self):
        entry = dict(key='projection_gemm:resume', family='projection_gemm')
        tools = {'atlas-opt': {'sha256': 'test-opt'}, 'atlas-emit': {'sha256': 'test-emit'}}
        backend = dict(tier='verified', min_tier='stream-verified', tools=tools)
        for cached_status, cached_backend in (('running', backend), ('timeout', None)):
            with self.subTest(cached_status=cached_status), tempfile.TemporaryDirectory() as tmp:
                args = job_args(tmp, backend_tools=tools)
                for mode in ('functional', 'timed'):
                    write_cached_job(args, entry, mode, dict(
                        key=entry['key'], mode=mode, status=cached_status, backend=cached_backend, seconds=1))

                def rerun(command, **kwargs):
                    mode, destination = job_destination(command)
                    destination.write_text(json.dumps(dict(key=entry['key'], mode=mode, status='no-variant')))
                    return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

                results, process = execute_jobs(entry, args, rerun)
                self.assertEqual(process.call_count, 2)
                self.assertEqual(sorted(call.args[0][call.args[0].index('--mode') + 1]
                                        for call in process.call_args_list), ['functional', 'timed'])
                for call in process.call_args_list:
                    command = call.args[0]
                    self.assertEqual(command[command.index('--atlas-mlir-bin') + 1], str(args.atlas_mlir_bin))
                    self.assertEqual(command[command.index('--min-tier') + 1], args.min_tier)
                self.assertEqual({mode: result['status'] for mode, result in results[entry['key']].items()},
                                 {'functional': 'no-variant', 'timed': 'no-variant'})

    def test_timeout_retains_verified_checkpoint_identity(self):
        entry = dict(key='projection_gemm:timeout', family='projection_gemm', params={})
        identity = dict(backend=dict(tier='verified', min_tier='stream-verified', tools={}),
                        program_words=7, words_sha256='words-digest', arena_sha256='arena-digest')
        for checkpoint in (True, False):
            with self.subTest(checkpoint=checkpoint), tempfile.TemporaryDirectory() as tmp:
                args = job_args(tmp, resume=False, subset_rows=128)

                def timeout(command, **kwargs):
                    if checkpoint:
                        mode, destination = job_destination(command)
                        bench_model.write_job_result(destination, dict(
                            key=entry['key'], mode=mode, status='running', **identity))
                    raise subprocess.TimeoutExpired(command, kwargs['timeout'])

                results, _ = execute_jobs(entry, args, timeout)
                self.assertEqual(set(results[entry['key']]), {'functional', 'timed'})
                for result in results[entry['key']].values():
                    if checkpoint:
                        self.assertEqual(result['status'], 'timeout')
                        for field, value in identity.items():
                            self.assertEqual(result[field], value)
                    else:
                        self.assertEqual(result['status'], 'error')
                        self.assertIn('before compiler verification completed', result['error'])

    def test_resume_rebuilds_and_rejects_stale_cache_closure(self):
        entry = dict(key='projection_gemm:closure', family='projection_gemm', params={})
        context = dict(implementation='staging-v1', model='model-v1', compiler='policy-v1')
        staged = dict(words_sha256='words-v1', arena_sha256='arena-v1')
        cases = ('match', 'words', 'arena', 'implementation', 'model', 'compiler', 'legacy', 'timeout',
                 'negative-match', 'negative-implementation', 'negative-now-supported')
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                args = job_args(tmp, atlas_mlir_bin=None, backend_tools=None, workers=1)
                for mode in ('functional', 'timed'):
                    binding = bench_model.cache_binding(entry, mode, copy.deepcopy(context), 100, 100, 1)
                    if case in ('implementation', 'model', 'compiler', 'negative-implementation'):
                        binding['context'][case.removeprefix('negative-')] = 'stale'
                    cached = dict(key=entry['key'], mode=mode, cache_binding=binding, cache_identity=staged,
                                  status='no-variant' if case.startswith('negative') else 'ok',
                                  error='unsupported', output_sha256='historical-output')
                    if case == 'legacy':
                        cached.pop('cache_binding')
                    if case == 'timeout':
                        cached['status'] = 'timeout'
                        binding['timeout'] = 0.5
                    write_cached_job(args, entry, mode, cached)

                reruns, checks = [], []

                def subprocess_job(command, **kwargs):
                    mode, destination = job_destination(command)
                    identity = copy.deepcopy(staged)
                    if case in ('words', 'arena'):
                        identity[case + '_sha256'] = 'changed'
                    binding = bench_model.cache_binding(entry, mode, context, 100, 100, 1)
                    result = dict(key=entry['key'], mode=mode, cache_binding=binding,
                                  cache_identity=identity, status='identity', error='unsupported')
                    if '--identity-only' in command:
                        checks.append(mode)
                        if case in ('negative-match',):
                            result['status'] = 'no-variant'
                    else:
                        reruns.append(mode)
                        result.update(status='ok', output_sha256='fresh-output')
                    destination.write_text(json.dumps(result))
                    return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

                results, _ = execute_jobs(entry, args, subprocess_job, context)
                reused = case in ('match', 'negative-match')
                self.assertEqual(len(reruns), 0 if reused else 2)
                self.assertEqual(len(checks), 2 if case in ('match', 'words', 'arena', 'negative-match',
                                                         'negative-now-supported') else 0)
                if reused:
                    self.assertTrue(all(r['output_sha256'] == 'historical-output' for r in results[entry['key']].values()))
                else:
                    self.assertTrue(all(r['output_sha256'] == 'fresh-output' for r in results[entry['key']].values()))

    def test_source_closure_tracks_golden_and_model_data_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'reference.py').write_text('golden = 1\n')
            (root / 'table.bin').write_bytes(b'lookup-v1')
            original = bench_model.source_identity(root)
            (root / 'reference.py').write_text('golden = 2\n')
            changed_golden = bench_model.source_identity(root)
            self.assertNotEqual(original, changed_golden)
            (root / 'table.bin').write_bytes(b'lookup-v2')
            self.assertNotEqual(changed_golden, bench_model.source_identity(root))
            (root / 'link.py').symlink_to(root / 'reference.py')
            with self.assertRaisesRegex(ValueError, 'symbolic links'):
                bench_model.source_identity(root)
            (root / 'link.py').unlink()
            before = bench_model.source_identity(root)
            (root / '__pycache__').mkdir()
            (root / '__pycache__' / 'cached.pyc').write_bytes(b'ignored')
            self.assertEqual(before, bench_model.source_identity(root))

    def test_render_preserves_reuse_beside_totals_and_timeout_provenance(self):
        entry = dict(key='attention:timeout-provenance', family='attention', count=24, params=dict(
            rows=1024, keys=1024, heads=12, kv_heads=12, head_dim=64, mask='none', output_fp8=False),
            modules=[], dtypes=['float32'])
        note = 'timeout marker reconstructed from bench-smolvla.log after the merge crash (retained log: 1,200.1 s)'
        results = {entry['key']: dict(functional=dict(status='ok', seconds=1),
                                     timed=dict(status='timeout', seconds=1200.1, note=note),
                                     subset=dict(rows=128, factor=8, result=dict(status='ok', model_cycles=5)))}
        rows = bench_model.merge([entry], results)
        self.assertEqual(rows[0]['timed_provenance_note'], note)
        document = dict(model='smolvla', command='test', coverage='coverage.json', coverage_sha256='test',
                        coverage_status='COMPLETE', npu_model='test', workers=1, launch_checks=[],
                        launches=rows, unmapped=[], uncovered=[], reuse_proof=dict(path='reuse-proof.json',
                            sha256='7745da6b8410d7fb6a2456e4699af8c048f340be4d51be47521208d5bec1437f'))
        report = bench_model.render(document)
        summary = report.split('## Summary', 1)[1].split('Compiler tier counts', 1)[0]
        self.assertIn('includes 24 estimated launches', summary)
        self.assertIn('zero functional or timed bench jobs were rerun', summary)
        self.assertIn('`reuse-proof.json` beside the generated JSON', summary)
        self.assertIn(note, report)
        # The retained report lacks the timed-job note, so only its exact historical proof enables the fallback.
        rows[0].pop('timed_provenance_note')
        document['reuse_proof']['sha256'] = '7745da6b8410d7fb6a2456e4699af8c048f340be4d51be47521208d5bec1437f'
        self.assertIn(note, bench_model.render(document))
        document['reuse_proof']['sha256'] = 'unrelated-proof'
        unrelated = bench_model.render(document)
        self.assertNotIn('timeout marker reconstructed', unrelated)
        self.assertNotIn('zero functional or timed bench jobs were rerun', unrelated)


@unittest.skipUnless(CAN_RUN, 'needs NumPy, PyTorch, and npu_model (ATLAS_NPU_MODEL_ROOT)')
class RunTest(unittest.TestCase):
    def test_resume_regenerates_identity_and_reruns_stale_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'coverage.json'
            path.write_text(json.dumps(toy_model()))
            out = Path(tmp) / 'out'
            command = ['--coverage', str(path), '--model-root', MODEL_ROOT, '--out', str(out),
                       '--workers', '1', '--family', 'projection_gemm']
            with mock.patch('builtins.print'):
                self.assertEqual(bench_model.main(command), 0)
            real_subprocess = bench_model.subprocess.run
            with mock.patch.object(bench_model.subprocess, 'run', wraps=real_subprocess) as process, \
                    mock.patch('builtins.print'):
                self.assertEqual(bench_model.main([*command, '--resume']), 0)
            jobs = [call.args[0] for call in process.call_args_list if '--job' in call.args[0]]
            self.assertEqual(len(jobs), 2)
            self.assertTrue(all('--identity-only' in job for job in jobs))
            functional, = (out / 'jobs').glob('*.functional.json')
            cached = json.loads(functional.read_text())
            cached['cache_identity']['words_sha256'] = 'stale'
            functional.write_text(json.dumps(cached))
            with mock.patch.object(bench_model.subprocess, 'run', wraps=real_subprocess) as process, \
                    mock.patch('builtins.print'):
                self.assertEqual(bench_model.main([*command, '--resume']), 0)
            jobs = [call.args[0] for call in process.call_args_list if '--job' in call.args[0]]
            self.assertEqual(len(jobs), 3)
            rerun, = [job for job in jobs if '--identity-only' not in job]
            self.assertEqual(rerun[rerun.index('--mode') + 1], 'functional')
            repaired = json.loads(functional.read_text())
            self.assertNotEqual(repaired['cache_identity']['words_sha256'], 'stale')

    def test_causal_reference_mask_uses_explicit_query_start(self):
        import numpy as np
        actual = bench_model.attention_mask(dict(rows=2, keys=5, mask='causal', query_start=1))
        np.testing.assert_array_equal(actual, [[True, True, False, False, False],
                                              [True, True, True, False, False]])

    def test_runs_reports_no_variant_and_unmapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'coverage.json'
            path.write_text(json.dumps(toy_model()))
            out = Path(tmp) / 'out'
            # Tool-discovery environment variables must not opt an ordinary bench into compilation.
            with mock.patch.dict(os.environ, {'ATLAS_MLIR_BIN': str(Path(tmp) / 'missing-compiler')}):
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
            self.assertIsNone(document['backend'])
            self.assertIsNone(gemm['backend'])
            self.assertEqual(norm['status'], 'no-variant')
            self.assertIn("multiple of 32", norm['error'])
            self.assertEqual(norm['count'], 2)
            text = (out / 'MODEL-BENCH.md').read_text()
            self.assertIn('no-variant', text)
            self.assertIn('npu_model model cycles', text)
            self.assertIn('ignore host time, launch overhead', text)
            self.assertIn('`aten.softmax.int`', text)
            self.assertIn('Compiler tier counts by distinct launch: none.', text)

    def run_backend_cli(self, temporary, compiler, *options):
        path = temporary / 'coverage.json'
        path.write_text(json.dumps(toy_model()))
        out = temporary / 'out'
        process = subprocess.run([sys.executable, '-B', str(COVERAGE / 'bench_model.py'),
                                  '--coverage', str(path), '--model-root', MODEL_ROOT, '--out', str(out),
                                  '--workers', '2', '--family', 'projection_gemm',
                                  '--atlas-mlir-bin', str(compiler), *options],
                                 capture_output=True, text=True, timeout=120)
        self.assertTrue((out / 'MODEL-BENCH.json').is_file(), process.stdout + process.stderr)
        return process, json.loads((out / 'MODEL-BENCH.json').read_text()), (out / 'MODEL-BENCH.md').read_text()

    def test_backend_mismatch_and_tier_rejection_fail_cli(self):
        for reject_stream, expected in ((False, 'atlas-emit words differ from the assembler'),
                                        (True, 'program tier encode-only is below required verified')):
            with self.subTest(reject_stream=reject_stream), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                compiler = root / 'bin'
                compiler.mkdir()
                # These tools exercise the real adapter and job subprocesses. A successful emitter
                # with incorrect words must fail just as a stream verifier rejection must fail.
                for name in ('atlas-opt', 'atlas-emit'):
                    tool = compiler / name
                    tool.write_text(f'#!{sys.executable}\nimport sys\n'
                                    'if "--version" in sys.argv:\n    print("test compiler")\n'
                                    + ('elif True:\n    print("00000000")\n' if name == 'atlas-emit' else
                                       ('else:\n    print("kernel.mlir:1: error: forced stream rejection", file=sys.stderr)\n'
                                        '    sys.exit(1)\n' if reject_stream else 'else:\n    pass\n')))
                    tool.chmod(0o755)
                options = ['--min-tier', 'verified'] if reject_stream else []
                process, document, text = self.run_backend_cli(root, compiler, *options)
                self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
                self.assertEqual(len(document['launches']), 1)
                row = document['launches'][0]
                self.assertEqual(row['status'], 'error')
                self.assertIn(expected, row['error'])
                self.assertIn(expected, process.stderr)
                self.assertIsNone(row['backend'])
                self.assertIn('Compiler tier counts by distinct launch: none.', text)

    @unittest.skipUnless(CAN_COMPILE, 'needs ATLAS_MLIR_BIN with atlas-opt and atlas-emit')
    def test_backend_cli_records_equal_words_tier_and_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            process, document, text = self.run_backend_cli(Path(tmp), MLIR_BIN)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(len(document['launches']), 1)
        row = document['launches'][0]
        self.assertEqual(row['status'], 'ok')
        self.assertTrue(row['bit_exact'])
        self.assertTrue(row['timed_output_equal'])
        backend = row['backend']
        self.assertTrue(backend['words_equal_assembler'])
        self.assertEqual(backend['min_tier'], 'stream-verified')
        self.assertIn(backend['tier'], ('verified', 'stream-verified'))
        self.assertEqual(backend['words'], row['program_words'])
        self.assertEqual(backend['words_sha256'], row['words_sha256'])
        self.assertEqual(backend['tools'], document['backend']['tools'])
        for tool in ('atlas-opt', 'atlas-emit'):
            self.assertEqual(Path(backend['tools'][tool]['path']).parent, Path(MLIR_BIN).absolute())
            self.assertRegex(backend['tools'][tool]['sha256'], '^[0-9a-f]{64}$')
        for field in ('source_sha256', 'words_sha256', 'arena_sha256'):
            self.assertRegex(row[field], '^[0-9a-f]{64}$')
        self.assertGreater(row['arena_bytes'], 0)
        self.assertIn(f"Compiler tier counts by distinct launch: 1 `{backend['tier']}`.", text)

    def test_generator_rejection_is_no_variant(self):
        entry = dict(key='epilogue:toy', family='epilogue',
                     params=dict(m=32, k=32, n=64, rows=8, logical_k=32, logical_n=64, steps=['silu'], output='bf16'))
        with tempfile.TemporaryDirectory() as tmp:
            result = bench_model.run_job(copy.deepcopy(entry), 'functional', Path(MODEL_ROOT), Path(tmp), 10_000_000, 10_000_000)
        self.assertEqual(result['status'], 'no-variant')
        self.assertIn('traffic reuse is supported only for the qualified shapes', result['error'])


if __name__ == '__main__':
    unittest.main()
