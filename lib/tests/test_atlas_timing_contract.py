"""Image-bound input contracts and conditional backend tiers, without external tools."""
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
import atlas_backend as backend  # noqa: E402
import atlas_timing_contract as contracts  # noqa: E402


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.image = self.directory / 'input.bin'
        self.image.write_bytes(bytes.fromhex('ffffffff010000000200000003000000'))
        self.contract = dict(name='attention-metadata-v1',
                             segments=[dict(address=0x90000000, bytes_hex='ffffffff01000000')],
                             image=dict(path=str(self.image), address=0x90000000))
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop(contracts.ENVIRONMENT, None)
        backend._cache.clear()

    def manifest(self, contract=None, source='ecall\n'):
        path = self.directory / 'contracts.json'
        path.write_text(json.dumps(dict(schema=contracts.SCHEMA,
                                       contracts={hashlib.sha256(source.encode()).hexdigest():
                                                  self.contract if contract is None else contract})))
        os.environ[contracts.ENVIRONMENT] = str(path)
        return path

    def test_words_are_unsigned_little_endian_and_sorted(self):
        self.contract['segments'] = [dict(address=0x90000008, bytes_hex='02000000'),
                                     dict(address=0x90000000, bytes_hex='ffffffff01000000')]
        result = contracts.validate(self.contract)
        self.assertEqual(result.words, ((0x90000000, 0xffffffff), (0x90000004, 1), (0x90000008, 2)))
        self.assertRegex(result.sha256, '^[0-9a-f]{64}$')
        self.assertEqual(result.record()['assumptions'], list(contracts.ASSUMPTIONS))
        mlir = result.annotate('module {\n  body\n}\n')
        self.assertIn('address = 2415919104 : i64, value = 4294967295 : i64', mlir)
        self.assertTrue(mlir.endswith('  body\n}\n'))

    def test_changed_image_rejects_declared_bytes(self):
        contracts.validate(self.contract)
        self.image.write_bytes(bytes(16))
        with self.assertRaisesRegex(contracts.ContractError, 'differ from the supplied image'):
            contracts.validate(self.contract)

    def test_duplicate_and_overlapping_segments_reject(self):
        for address in (0x90000000, 0x90000004):
            with self.subTest(address=address):
                candidate = copy.deepcopy(self.contract)
                candidate['segments'].append(dict(address=address, bytes_hex='01000000'))
                with self.assertRaisesRegex(contracts.ContractError, 'overlapping'):
                    contracts.validate(candidate)

    def test_invalid_segments_reject(self):
        candidates = [dict(address=0x90000001, bytes_hex='00000000'),
                      dict(address=True, bytes_hex='00000000'),
                      dict(address=-4, bytes_hex='00000000'),
                      dict(address=2**32, bytes_hex='00000000'),
                      dict(address=0xfffffffc, bytes_hex='0000000000000000'),
                      dict(address=0x90000000, bytes_hex=''),
                      dict(address=0x90000000, bytes_hex='0000'),
                      dict(address=0x90000000, bytes_hex='zzzzzzzz'),
                      dict(address=0x90000000, bytes_hex='00 00 00 00')]
        for segment in candidates:
            with self.subTest(segment=segment):
                candidate = copy.deepcopy(self.contract)
                candidate['segments'] = [segment]
                with self.assertRaises(contracts.ContractError):
                    contracts.validate(candidate)

    def test_outside_image_rejects(self):
        self.contract['segments'][0]['address'] += 16
        with self.assertRaisesRegex(contracts.ContractError, 'outside the supplied image'):
            contracts.validate(self.contract)

    def test_name_is_nonempty_and_mlir_escaped(self):
        for name in ('', '   ', 'bad\nname', None):
            with self.subTest(name=name):
                self.contract['name'] = name
                with self.assertRaises(contracts.ContractError):
                    contracts.validate(self.contract)
        self.contract['name'] = 'metadata " \\ \u2603'
        self.assertIn(r'name = "metadata \22 \5C \E2\98\83"',
                      contracts.validate(self.contract).annotate('module {\n}\n'))

    def test_path_checks_before_reads(self):
        for path in ('../input.bin', '/path/to/ha' + 'mmer/input.bin', '/path/to/vl' + 'si/input.bin'):
            with self.subTest(path=path):
                self.contract['image']['path'] = path
                with self.assertRaises(contracts.ContractError):
                    contracts.validate(self.contract)
        link = self.directory / 'link.bin'
        link.symlink_to(self.image)
        self.contract['image']['path'] = str(link)
        with self.assertRaisesRegex(contracts.ContractError, 'symbolic-link'):
            contracts.validate(self.contract)

    def test_manifest_selects_only_supplied_source_and_relative_image(self):
        self.contract['image']['path'] = 'input.bin'
        self.manifest()
        self.assertIsNone(contracts.select('fence\necall\n'))
        self.assertEqual(contracts.select('ecall\n').name, 'attention-metadata-v1')

    def test_bad_manifest_and_selected_null_reject(self):
        path = self.manifest()
        valid = path.read_text()
        for text in ('{}', valid.replace(contracts.SCHEMA, 'unknown'),
                     '{"schema":"x","schema":"y","contracts":{}}',
                     json.dumps(dict(schema=contracts.SCHEMA, contracts={'bad': self.contract})),
                     json.dumps(dict(schema=contracts.SCHEMA,
                                     contracts={hashlib.sha256(b'ecall\n').hexdigest(): None}))):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(contracts.ContractError):
                    contracts.select('ecall\n')

    def fake_compile(self, **options):
        def process(argv, **kwargs):
            class Completed:
                returncode = 0
                def communicate(self):
                    return ('00000073\n', '') if argv[0] == 'emit' else ('', '')
            return Completed()
        with patch.object(backend, 'tools', return_value=backend.Tools('opt', 'emit')), \
             patch.object(backend.subprocess, 'Popen', side_effect=process):
            return backend.compile('ecall\n', **options)

    def test_default_stays_unconditional(self):
        result = self.fake_compile()
        self.assertEqual(result.tier, 'verified')
        self.assertNotIn('dram_contract', result.record())
        self.assertNotIn('atlas.dram_contract', result.mlir)

    def test_direct_contract_is_conditional_and_recorded(self):
        result = self.fake_compile(dram_contract=self.contract, min_tier='contract-verified')
        self.assertEqual(result.tier, 'contract-verified')
        self.assertIn('atlas.dram_contract', result.mlir)
        self.assertEqual(result.record()['dram_contract']['name'], 'attention-metadata-v1')
        self.assertEqual(result.record()['dram_contract']['assumptions'], list(contracts.ASSUMPTIONS))
        with self.assertRaisesRegex(backend.BackendError, 'below required verified'):
            self.fake_compile(dram_contract=self.contract, min_tier='verified')

    def test_environment_contract_is_revalidated_before_cache(self):
        self.fake_compile()  # Populate the unconditional cache for the same source.
        self.manifest()
        self.assertEqual(self.fake_compile().tier, 'contract-verified')
        self.image.write_bytes(bytes(16))
        with self.assertRaisesRegex(backend.BackendError, 'differ from the supplied image'):
            self.fake_compile()

    def test_manifest_edits_are_reread(self):
        path = self.manifest()
        first = self.fake_compile()
        self.contract['name'] = 'second-contract'
        path.write_text(json.dumps(dict(schema=contracts.SCHEMA,
                                       contracts={hashlib.sha256(b'ecall\n').hexdigest(): self.contract})))
        second = self.fake_compile()
        self.assertNotEqual(first.dram_contract['sha256'], second.dram_contract['sha256'])
        self.assertEqual(second.dram_contract['name'], 'second-contract')

    def test_skipped_verification_keeps_encode_only_tier(self):
        result = self.fake_compile(dram_contract=self.contract, verify=False, min_tier='encode-only')
        self.assertEqual(result.tier, 'encode-only')
        self.assertIsNotNone(result.dram_contract)

    def test_environment_session_does_not_hide_contract_changes_by_deduplication(self):
        first = self.fake_compile(dram_contract=self.contract)
        self.contract['name'] = 'other-assumptions'
        second = self.fake_compile(dram_contract=self.contract)
        self.manifest()
        with patch.object(backend, 'tools', return_value=backend.Tools('opt', 'emit')), \
             patch.object(backend, 'cross_checked', side_effect=[first, second]):
            session = backend.Session('atlas-mlir', self.directory / 'records.jsonl')
            session.encode('ecall\n', [0x73])
            with self.assertRaisesRegex(backend.BackendError, 'contract changed'):
                session.encode('ecall\n', [0x73])

    def test_merge_keeps_conflicting_outcomes_under_the_same_contract(self):
        first = dict(source_sha256=hashlib.sha256(b'ecall\n').hexdigest(),
                     **self.fake_compile(dram_contract=self.contract).record())
        second = dict(first, tier='stream-verified', timing='reject', timing_diagnostics=['rejected'])
        spool = self.directory / 'records.jsonl'
        spool.write_text('\n'.join(json.dumps(p) for p in (first, second, first)) + '\n')
        with patch.object(backend, 'tools', return_value=backend.Tools('opt', 'emit')):
            programs = backend.Session('atlas-mlir', spool).programs()
        self.assertCountEqual(programs, [first, second])

    def real_tools(self):
        try:
            return backend.tools()
        except backend.BackendUnavailable:
            self.skipTest('atlas-opt and atlas-emit not found')

    def input_word(self, value):
        contents = value.to_bytes(4, 'little') + bytes(28)
        self.image.write_bytes(contents)
        self.contract['segments'] = [dict(address=0x90000000, bytes_hex=contents.hex())]

    def fork_policy(self, source, values, tiers, minimum):
        self.real_tools()
        words = backend.compile(source, min_tier='encode-only').words
        manifests = []
        for number, value in enumerate(values):
            if value is None:
                manifests.append(None)
                continue
            contents = value.to_bytes(4, 'little') + bytes(28)
            image = self.directory / f'child-{number}.bin'
            image.write_bytes(contents)
            contract = dict(name=f'child-{number}',
                            segments=[dict(address=0x90001000, bytes_hex=contents.hex())],
                            image=dict(path=str(image), address=0x90001000))
            manifest = self.directory / f'child-{number}.json'
            manifest.write_text(json.dumps(dict(schema=contracts.SCHEMA, contracts={
                hashlib.sha256(source.encode()).hexdigest(): contract})))
            manifests.append(str(manifest))
        for order in ((0, 1), (1, 0)):
            with self.subTest(order=order):
                script = self.directory / 'children.py'
                raw = self.directory / 'raw.json'
                # Repeat the first invocation in another child to check exact deduplication.
                script.write_text(
                    'import json, os, sys\nfrom pathlib import Path\n'
                    'session = next(f.session for f in sys.meta_path '
                    'if f.__class__.__name__ == "_ConverterHook")\n'
                    f'for manifest in {[manifests[i] for i in (order[0], *order)]!r}:\n'
                    '    child = os.fork()\n'
                    '    if child == 0:\n'
                    f'        os.environ.pop({contracts.ENVIRONMENT!r}, None)\n'
                    f'        if manifest: os.environ[{contracts.ENVIRONMENT!r}] = manifest\n'
                    f'        session.encode({source!r}, {words!r})\n'
                    '        os._exit(0)\n'
                    '    assert os.waitpid(child, 0)[1] == 0\n'
                    f'Path({str(raw)!r}).write_text(json.dumps(['
                    'json.loads(line) for line in session.spool.read_text().splitlines()]))\n')
                receipt = self.directory / 'receipt.json'
                done = subprocess.run([sys.executable, backend.__file__, 'run', '--backend', 'atlas-mlir',
                                       '--receipt', str(receipt), '--min-tier', minimum, str(script)],
                                      capture_output=True, text=True)
                records = json.loads(receipt.read_text())
                self.assertEqual([p['tier'] for p in json.loads(raw.read_text())],
                                 [tiers[i] for i in (order[0], *order)])
                self.assertNotEqual(done.returncode, 0, done.stderr)
                self.assertCountEqual([p['tier'] for p in records['programs']], tiers)
                self.assertEqual(len({p['source_sha256'] for p in records['programs']}), 1)
                self.assertIn(f'below required {minimum}', '; '.join(records['policy_errors']))
                if 'stream-verified' in tiers:
                    self.assertIn('RAW conflict', ' '.join(d for p in records['programs']
                                                          for d in p['timing_diagnostics']))

    def test_fork_cannot_hide_conditional_proof_from_verified_policy(self):
        self.fork_policy('ecall\n', (None, 0), ('verified', 'contract-verified'), 'verified')

    def test_fork_cannot_hide_rejected_contract_from_conditional_policy(self):
        source = ('dma.config x0, ch0\nli x4, 256\nli x7, 0x90000000\nli x9, 1024\n'
                  'li x14, 0x2000\nli x15, 0x90001000\nli x16, 32\nli x17, 0x8000\n'
                  'dma.load x14, x15, x16, ch2\ndma.wait ch2\nlw x5, 0(x17)\ndelay 4\nnop\n'
                  'dma.load x4, x7, x9, ch0\nvload m0, 0(x5)\ndelay 66\nnop\ndma.wait ch0\necall\n')
        self.fork_policy(source, (0, 256), ('contract-verified', 'stream-verified'), 'contract-verified')

    METADATA_LOAD = ('dma.config x0, ch0\nli x4, 0\nli x7, 0x90000000\nli x9, 32\n'
                     'dma.load x4, x7, x9, ch0\ndma.wait ch0\nlw x10, 0(x0)\ndelay 8\nnop\n')
    POINTER_ACCESS = ('li x4, 256\nli x9, 1024\ndma.load x4, x7, x9, ch0\n'
                      'vload m0, 0(x10)\ndelay 66\nnop\ndma.wait ch0\necall\n')

    def test_real_contract_proof_has_conditional_tier_and_identical_words(self):
        tools = self.real_tools()
        self.input_word(0)
        source = self.METADATA_LOAD + self.POINTER_ACCESS
        ordinary = backend.compile(source, min_tier='encode-only')
        conditional = backend.compile(source, dram_contract=self.contract, min_tier='contract-verified')
        self.assertEqual((ordinary.tier, conditional.tier), ('stream-verified', 'contract-verified'))
        self.assertEqual(ordinary.words, conditional.words)
        with self.assertRaisesRegex(backend.BackendError, 'below required verified'):
            backend.compile(source, dram_contract=self.contract, min_tier='verified')
        source_mlir, timed_mlir = self.directory / 'source.mlir', self.directory / 'timed.mlir'
        source_mlir.write_text(conditional.mlir)
        subprocess.run([tools.atlas_opt, '--verify-atlas-timing', str(source_mlir), '-o', str(timed_mlir)],
                       check=True, capture_output=True, text=True)
        for mode in ('--program-json', '--map-json'):
            output = subprocess.run([tools.atlas_emit, mode, str(timed_mlir)],
                                    check=True, capture_output=True, text=True)
            record = json.loads(output.stdout)
            self.assertIn('attention-metadata-v1', record['input_dram_contract'])
            self.assertEqual(record['timing_assumptions'], ['immutable-dram:attention-metadata-v1'])

    def test_real_contract_cannot_approve_unsafe_loaded_mask_pointer(self):
        self.real_tools()
        self.input_word(256)  # Word address 256 aliases the pending transfer at byte 1024.
        result = backend.compile(self.METADATA_LOAD + self.POINTER_ACCESS,
                                 dram_contract=self.contract, min_tier='encode-only')
        self.assertEqual(result.tier, 'stream-verified')
        self.assertIn('RAW conflict', ' '.join(result.timing_diagnostics))

    def test_real_loaded_transfer_bound_past_vmem_rejects(self):
        self.real_tools()
        self.input_word(64)
        # The destination has only 32 bytes left in the 1536 KiB VMEM.
        source = self.METADATA_LOAD + ('li x4, 393208\ndma.load x4, x7, x10, ch0\n'
                                       'dma.wait ch0\necall\n')
        result = backend.compile(source, dram_contract=self.contract, min_tier='encode-only')
        self.assertEqual(result.tier, 'stream-verified')
        self.assertIn('past the end of VMEM', ' '.join(result.timing_diagnostics))

if __name__ == '__main__':
    unittest.main()
