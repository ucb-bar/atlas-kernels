"""Translator and Atlas compiler backend checks.

Translator tests need nothing beyond Python. Backend tests need atlas-opt and
atlas-emit (ATLAS_MLIR_BIN, ATLAS_OOT_BIN_DIR, or PATH); assembler comparisons
also need npu_model 47b8599 on PYTHONPATH. Missing tools skip, never pass.
"""

import io
import os
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401

import atlas_asm_mlir as t  # noqa: E402
import atlas_backend  # noqa: E402

try:
    atlas_backend.tools()
    TOOLS = True
except atlas_backend.BackendUnavailable:
    TOOLS = False

try:
    from npu_model.configs import isa_definition  # noqa: F401
    from npu_model.isa import IsaSpec
    from npu_model.util.converter import input_to_program
    MODEL = True
except ImportError:
    MODEL = False


def fields(text):
    return [(i.op, i.fields) for i in t.parse(text)]


class TranslatorTest(unittest.TestCase):
    def test_li_matches_the_assembler_expansion(self):
        self.assertEqual(fields('li x5, 17'), [('atlas.alu_imm', dict(kind='addi', dst=5, src=0, immediate=17))])
        self.assertEqual(fields('li x5, -2048'), [('atlas.alu_imm', dict(kind='addi', dst=5, src=0, immediate=-2048))])
        self.assertEqual(fields('li x31, 2415919104'), [('atlas.upper', dict(kind='lui', dst=31, immediate=0x90000))])
        self.assertEqual(fields('li x6, 0x12345fff'),
                         [('atlas.upper', dict(kind='lui', dst=6, immediate=0x12346)),
                          ('atlas.alu_imm', dict(kind='addi', dst=6, src=6, immediate=-1))])
        self.assertEqual(fields('nop'), [('atlas.alu_imm', dict(kind='addi', dst=0, src=0, immediate=0))])

    def test_labels_count_words_including_pseudo_expansions(self):
        program = t.parse('top:\nli x5, 0x12345678\nbne x5, x0, top  # back\nnop\njal x0, end\nnop\nend:\necall\n')
        self.assertEqual(program[2].fields['offset_bytes'], -4)   # two LI words back
        self.assertEqual(program[4].fields['offset'], 4)          # forward two words
        self.assertEqual(program[0].labels, ['top'])
        self.assertEqual(program[6].labels, ['end'])
        self.assertEqual(fields('beq x1, x2, -3')[0][1]['offset_bytes'], -6)

    def test_operand_forms_and_immediate_normalization(self):
        self.assertEqual(fields('vload m3, 4095(x7)')[0][1], dict(dst=3, base=7, offset=-1, format='raw'))
        self.assertEqual(fields('sw a0, -4(x2)')[0][1], dict(kind='sw', src=10, base=2, offset=-4))
        self.assertEqual(fields('vli.all m4, -1')[0][1], dict(mode='all', dst=4, immediate=0xffff))
        self.assertEqual(fields('lui x1, -1')[0][1], dict(kind='lui', dst=1, immediate=0xfffff))
        self.assertEqual(fields('srai x1, x2, 31')[0][1], dict(kind='srai', dst=1, src=2, immediate=31))
        self.assertEqual(fields('csrrwi x0, 0xC10, 7')[0][1], dict(kind='rrwi', dst=0, source=7, address=0xc10))
        self.assertEqual(fields('seli e3, 127')[0][1], dict(kind='seli', dst=3, base=0, offset=127))

    def test_unit_slot_and_dma_fields(self):
        self.assertEqual(fields('dma.load x15, x16, x17, ch3')[0][1],
                         dict(direction='load', channel=3, reg=15, dram=16, size=17))
        self.assertEqual(fields('dma.store x16, x15, x17, ch0')[0][1],
                         dict(direction='store', channel=0, reg=15, dram=16, size=17))
        self.assertEqual(fields('vmatmul.acc.mxu1 acc1, m4, w0')[0][1],
                         dict(unit=1, src=4, weight_slot=0, acc_slot=1, accumulate=True))
        self.assertEqual(fields('vmatpop.bf16.acc.mxu0 m2, acc1')[0][1],
                         dict(format='bf16', unit=0, dst=2, slot=1, scale_reg=0))
        self.assertEqual(fields('vpack.bf16.fp8 m9, m2, e1')[0][1],
                         dict(direction='bf16_to_fp8', dst=9, src=2, scale_reg=1))

    def test_rejects_what_the_assembler_rejects(self):
        for text in ('frob x1', 'beq x1, x2, nowhere', 'addi x1, x2', 'add x32, x0, x0', 'vload m64, 0(x0)',
                     'addi x1, x0, 4096', 'slli x1, x1, 32', 'dma.wait ch8', 'vmatmul.mxu0 acc2, m0, w0', 'sw a0, -4(sp)'):
            with self.subTest(text=text), self.assertRaises(t.AsmError):
                t.parse(text)

    def test_module_has_one_op_per_word_and_source_locations(self):
        module = t.translate('li x31, 2415919104\nli x5, 0x12345678\necall\n', name='k.S')
        ops = re.findall(r'"(atlas\.\w+)"', module)
        self.assertEqual(ops, ['atlas.start', 'atlas.upper', 'atlas.upper', 'atlas.alu_imm', 'atlas.trap'])
        self.assertIn('loc("k.S":2:1)', module)

    def test_mnemonic_list_matches_npu_model(self):
        if not MODEL:
            self.skipTest('npu_model 47b8599 not importable')
        self.assertEqual(set(t.MNEMONICS) - {'li', 'nop'}, set(IsaSpec.operations))


def sample(mnemonic, cls):
    """One legal instance of every mnemonic, using even BF16 pair bases."""
    special = {'csrrw': 'csrrw x5, 0xC10, x6', 'csrrs': 'csrrs x5, 0xC00, x0', 'csrrc': 'csrrc x5, 0xC02, x0',
               'csrrwi': 'csrrwi x0, 0xC11, 9', 'csrrsi': 'csrrsi x7, 0xC03, 0', 'csrrci': 'csrrci x7, 0xC01, 0',
               'seli': 'seli e4, 200', 'jalr': 'jalr x1, x0, 2', 'jal': 'jal x1, 2'}
    if mnemonic in special:
        return special[mnemonic]
    out = []
    for param in cls.params:
        if type(param).__name__ == 'Bundled':
            out.append('-32(x9)')
            continue
        fmt = getattr(param.inner, 'fmt', '')
        out.append({'x': 'x11', 'm': 'm6', 'e': 'e3', 'w': 'w1', 'acc': 'acc1', 'ch': 'ch5'}.get(fmt) or
                   ('2' if param.repr == 'imm' and mnemonic.startswith('b') else '21'))
    return (mnemonic + ' ' + ', '.join(out)).strip()


class SourceIdentityTest(unittest.TestCase):
    def test_git_tree_prunes_before_descending(self):
        import hashlib, subprocess, tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            build = Path(directory) / 'build'
            (source / '.git').mkdir(parents=True)
            (source / 'safe').mkdir()
            build.mkdir()
            content = b'original'
            (source / 'safe/file.txt').write_bytes(content)
            blob = hashlib.sha1(b'blob 8\0' + content).hexdigest()
            child = 'a' * 40
            forbidden = 'b' * 40
            (build / 'CMakeCache.txt').write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={source}\n')
            visited = []
            def command(argv, **kwargs):
                if argv[-2:] == ['rev-parse', 'HEAD']:
                    return subprocess.CompletedProcess(argv, 0, stdout='c' * 40 + '\n')
                self.assertEqual(argv[3:5], ['ls-tree', '-z'])
                visited.append(argv[5])
                if argv[5] == 'HEAD':
                    raw = (f'040000 tree {child}\tsafe\0'
                           f'040000 tree {forbidden}\tHaMmEr-data\0'
                           f'040000 tree {forbidden}\tVLSI-data\0').encode()
                elif argv[5] == child:
                    raw = f'100644 blob {blob}\tfile.txt\0'.encode()
                else:
                    self.fail('descended into restricted tree')
                return subprocess.CompletedProcess(argv, 0, stdout=raw)
            with patch('atlas_backend.subprocess.run', side_effect=command) as process:
                result = atlas_backend._build_source(build)
            self.assertEqual(visited, ['HEAD', child])
            self.assertEqual(process.call_count, 3)
            self.assertFalse(result['modified'])

    def test_manifest_tracks_working_bytes_and_missing_symlinks(self):
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            build = Path(directory) / 'build'
            source.mkdir()
            build.mkdir()
            (source / 'file.txt').write_text('original')
            (source / 'link.txt').symlink_to('file.txt')
            def git(*args):
                return subprocess.run(['git', '-C', str(source), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init', '-q')
            git('add', 'file.txt', 'link.txt')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                'commit', '-qm', 'test: fixture')
            (build / 'CMakeCache.txt').write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={source}\n')
            clean = atlas_backend._build_source(build)
            self.assertEqual(clean['commit'], git('rev-parse', 'HEAD'))
            self.assertFalse(clean['modified'])
            self.assertRegex(clean['tracked_source_sha256'], '^[0-9a-f]{64}$')
            (source / 'file.txt').write_text('modified')
            (source / 'link.txt').unlink()
            dirty = atlas_backend._build_source(build)
            self.assertTrue(dirty['modified'])
            self.assertNotEqual(dirty['tracked_source_sha256'], clean['tracked_source_sha256'])


@unittest.skipUnless(TOOLS, 'atlas-opt and atlas-emit not found')
class BackendTest(unittest.TestCase):
    def test_compile_verifies_and_encodes(self):
        result = atlas_backend.compile('li x5, 3\nloop:\naddi x5, x5, -1\nbne x5, x0, loop\nnop\necall\n')
        self.assertEqual((result.tier, result.stream, result.timing), ('verified', 'pass', 'pass'))
        self.assertEqual(result.words, (0x00300293, 0xfff28293, 0xfe029fe3, 0x00000013, 0x00000073))
        self.assertEqual((result.stream_diagnostics, result.timing_diagnostics), ((), ()))

    def test_timing_rejection_is_stream_verified_with_every_diagnostic(self):
        # The vload reads the line a dma.load on channel 0 is still filling.
        source = 'li x5, 0\nli x6, 0\nli x7, 1024\ndma.load x5, x6, x7, ch0\nvload m1, 0(x5)\ndma.wait ch0\necall\n'
        result = atlas_backend.compile(source)
        self.assertEqual((result.tier, result.stream, result.timing), ('stream-verified', 'pass', 'reject'))
        self.assertEqual(len(result.timing_diagnostics), 1)
        self.assertIn('RAW conflict with atlas.dma on channel 0', result.timing_diagnostics[0])
        self.assertEqual(atlas_backend.rule(result.timing_diagnostics[0]).split(' op ')[0], "'atlas.vload'")

    def test_stream_rejection_is_encode_only_with_the_rule(self):
        result = atlas_backend.compile('beq x0, x0, 3\nnop\necall\n', min_tier='encode-only')   # target past the program end
        self.assertEqual(result.tier, 'encode-only')
        self.assertIn('escapes the LLVM inline assembly block', result.stream_diagnostics[0])
        self.assertEqual(len(result.words), 3)

    def test_stream_rejection_fails_by_default(self):
        with self.assertRaisesRegex(atlas_backend.BackendError, 'below required stream-verified'):
            atlas_backend.compile('beq x0, x0, 3\nnop\necall\n')

    def test_verified_minimum_rejects_timing_failure(self):
        source = 'li x5, 0\nli x6, 0\nli x7, 1024\ndma.load x5, x6, x7, ch0\nvload m1, 0(x5)\ndma.wait ch0\necall\n'
        with self.assertRaisesRegex(atlas_backend.BackendError, 'below required verified'):
            atlas_backend.compile(source, min_tier='verified')

    def test_skipping_verification_requires_explicit_tier(self):
        with self.assertRaisesRegex(atlas_backend.BackendError, 'below required stream-verified'):
            atlas_backend.compile('ecall\n', verify=False)

    def test_no_verify_is_encode_only(self):
        result = atlas_backend.compile('fence\necall\n', verify=False, min_tier='encode-only')
        self.assertEqual((result.tier, result.stream, result.timing, result.words), ('encode-only', None, None, (0x0f, 0x73)))

    def test_field_rejection_raises(self):
        with self.assertRaises(atlas_backend.BackendError):
            atlas_backend.compile('vadd.bf16 m1, m2, m4\necall\n')    # odd BF16 pair base

    def test_cross_check_refuses_different_words(self):
        with self.assertRaises(atlas_backend.BackendError):
            atlas_backend.cross_checked('ecall\n', [0x00100073])

    def test_missing_tools_raise_instead_of_falling_back(self):
        with self.assertRaises(atlas_backend.BackendUnavailable):
            atlas_backend.compile('ebreak\necall\n', bin_dir='/nonexistent')

    def test_identity_records_both_binaries(self):
        identity = atlas_backend.tools().identity()
        for name in ('atlas-opt', 'atlas-emit'):
            self.assertRegex(identity[name]['sha256'], '^[0-9a-f]{64}$')

    @unittest.skipUnless(MODEL, 'npu_model 47b8599 not importable')
    def test_every_mnemonic_matches_the_assembler(self):
        for mnemonic, cls in sorted(IsaSpec.operations.items()):
            line = sample(mnemonic, cls)
            text = '\n'.join(['nop', line, 'nop', 'nop', 'ecall']) + '\n'
            with self.subTest(line=line):
                words = [int(w) for w in input_to_program(io.StringIO(text)).assemble()]
                # Lone instructions are not timing-correct programs, so only the stream verifier must accept them.
                self.assertEqual(atlas_backend.cross_checked(text, words, min_tier='encode-only').stream, 'pass')

    @unittest.skipUnless(MODEL, 'npu_model 47b8599 not importable')
    def test_generated_projection_kernel(self):
        from atlas_program import assembly
        source = assembly(64, 64, 64, 4 * 64 * 64 + 8192, -8)
        words = [int(w) for w in input_to_program(io.StringIO(source)).assemble()]
        result = atlas_backend.cross_checked(source, words)
        self.assertEqual(list(result.words), words)
        self.assertIn(result.tier, ('verified', 'stream-verified'))


@unittest.skipUnless(TOOLS and MODEL, 'atlas-opt, atlas-emit, and npu_model 47b8599 needed')
class RunWrapperTest(unittest.TestCase):
    """`atlas_backend.py run` cross-checks every assembly a script performs and writes a receipt."""

    def run_script(self, body, backend='atlas-mlir', options=()):
        import json, subprocess, tempfile
        directory = Path(tempfile.mkdtemp(prefix='atlas-backend-run-'))
        script = directory / 'job.py'
        script.write_text(body)
        receipt = directory / 'receipt.json'
        lib = Path(atlas_backend.__file__).absolute().parent
        completed = subprocess.run([sys.executable, str(lib / 'atlas_backend.py'), 'run', '--backend', backend,
                                    '--receipt', str(receipt), *options, '--', str(script), '--flag', 'value'],
                                   capture_output=True, text=True, env=dict(os.environ))
        return completed, json.loads(receipt.read_text())

    JOB = (
        'import io, sys\n'
        'assert sys.argv[1:] == ["--flag", "value"], sys.argv\n'
        'from npu_model.util.converter import input_to_program\n'
        'for text in ("fence\\necall\\n", "fence\\necall\\n", "li x5, 3\\naddi x5, x5, 1\\necall\\n"):\n'
        '    input_to_program(io.StringIO(text)).assemble()\n'
    )

    def test_receipt_records_backend_identity_and_tiers(self):
        completed, receipt = self.run_script(self.JOB)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual((receipt['backend'], receipt['exit_code']), ('atlas-mlir', 0))
        self.assertRegex(receipt['tools']['atlas-emit']['sha256'], '^[0-9a-f]{64}$')
        self.assertEqual(receipt['summary']['programs'], 2)
        self.assertEqual(receipt['summary']['tiers'], {'verified': 2})
        self.assertTrue(all(p['words_equal_assembler'] for p in receipt['programs']))

    def test_empty_wrapped_script_fails(self):
        completed, receipt = self.run_script('pass\n')
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('encoded 0 distinct programs', receipt['error'])

    def test_fresh_python_subprocess_does_not_satisfy_coverage(self):
        body = ('import subprocess, sys\n'
                'subprocess.run([sys.executable, "-c", '
                '\'import io; from npu_model.util.converter import input_to_program; '
                'input_to_program(io.StringIO("ecall\\n")).assemble()\'], check=True)\n')
        completed, receipt = self.run_script(body)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(receipt['summary']['programs'], 0)
        self.assertIn('encoded 0 distinct programs', receipt['error'])

    def test_expected_program_count_fails_even_with_successful_encodes(self):
        completed, receipt = self.run_script(self.JOB, options=('--expect-min-programs', '3'))
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(receipt['summary']['programs'], 2)
        self.assertIn('expected at least 3', receipt['error'])

    def test_run_enforces_minimum_tier_and_keeps_rejected_record(self):
        job = ('import io\nfrom npu_model.util.converter import input_to_program\n'
               'input_to_program(io.StringIO("beq x0, x0, 3\\nnop\\necall\\n")).assemble()\n')
        completed, receipt = self.run_script(job)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(receipt['summary']['tiers'], {'encode-only': 1})
        self.assertIn('below required stream-verified', receipt['error'])
        completed, receipt = self.run_script(job, options=('--min-tier', 'encode-only'))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(receipt['summary']['tiers'], {'encode-only': 1})

    def test_assembler_backend_records_without_the_compiler(self):
        completed, receipt = self.run_script(self.JOB, backend='assembler')
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual((receipt['tools'], receipt['summary']['tiers']), (None, {'assembler': 2}))

    def test_backend_gap_fails_the_script(self):
        completed, receipt = self.run_script(
            'import io\nfrom npu_model.util.converter import input_to_program\n'
            'input_to_program(io.StringIO("vadd.bf16 m1, m2, m4\\necall\\n")).assemble()\n')
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('BackendError', receipt['error'])


if __name__ == '__main__':
    unittest.main()
