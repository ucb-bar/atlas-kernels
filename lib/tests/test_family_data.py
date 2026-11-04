"""Generated data binds deterministic inputs to host references and true oracles."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
import family_data  # noqa: E402
import kernel_bench as bench  # noqa: E402
import rtl_cases  # noqa: E402
from atlas_vector_arithmetic import ordered_matmul, rescale_bf16_array  # noqa: E402


class FamilyDataTest(unittest.TestCase):
    def test_default_staging_matches_existing_rtl_case(self):
        for family in family_data.FAMILIES:
            with self.subTest(family=family):
                spec, case = family_data.build_case(family)
                previous_spec, previous = rtl_cases.load_case(family)
                self.assertEqual(spec, previous_spec)
                self.assertEqual((case.source, case.arena, case.golden),
                                 (previous.source, previous.arena, previous.golden))
                np.testing.assert_array_equal(case.reference, previous.reference)

    def test_projection_inputs_reference_and_golden_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            first = family_data.produce_family('projection_gemm', Path(directory) / 'one', seed=19)
            second = family_data.produce_family('projection_gemm', Path(directory) / 'two', seed=19)
            self.assertEqual(first, second)
            out = Path(directory) / 'one' / 'projection_gemm'
            a = np.load(out / 'activations_fp32.npy', allow_pickle=False)
            w = np.load(out / 'weights_fp32.npy', allow_pickle=False)
            codes_a = np.load(out / 'activations_fp8.npy', allow_pickle=False)
            codes_w = np.load(out / 'weights_fp8.npy', allow_pickle=False)
            expected = bench.load_reference('projection_gemm').reference(a, w)
            np.testing.assert_array_equal(np.load(out / 'reference.npy', allow_pickle=False), expected)
            scales = first['case_metadata']
            golden = bench.pack_bf16_tiles(rescale_bf16_array(
                ordered_matmul(codes_a, codes_w), scales['a_scale'] + scales['w_scale']))
            self.assertEqual((out / 'golden.bin').read_bytes(), golden)
            self.assertEqual(first['independent_arithmetic']['status'], 'PASS')
            self.assertEqual(first['device_validation']['status'], 'PENDING')
            self.assertEqual(first['seed'], 19)
            self.assertEqual(first['logical_shape'], [1, 32, 960])
            for name, info in first['artifacts'].items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), info['sha256'])
            for dependency in ('kernels/projection_gemm/atlas_program_reuse_blocked.py',
                               'lib/fallback/tile_admission.py', 'lib/fallback/chipyard/arena.py'):
                self.assertEqual(first['dependencies'][dependency], hashlib.sha256(
                    (family_data.ROOT / dependency).read_bytes()).hexdigest())
            self.assertEqual((out / 'input_arena.bin').read_bytes()[:codes_a.size],
                             bench.pack_fp8_tiles(codes_a))
            self.assertEqual((out / 'input_arena.bin').read_bytes()[codes_a.size:codes_a.size + codes_w.size],
                             bench.pack_fp8_weights(codes_w))
            different = family_data.produce_family('projection_gemm', Path(directory) / 'other', seed=20)
            self.assertNotEqual(first['artifacts']['input_arena.bin']['sha256'],
                                different['artifacts']['input_arena.bin']['sha256'])

    def test_missing_oracles_remain_pending_with_real_fidelity_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            for family in ('attention', 'rmsnorm', 'epilogue', 'smol_fusion'):
                with self.subTest(family=family):
                    manifest = family_data.produce_family(family, directory)
                    out = Path(directory) / family
                    self.assertEqual(manifest['status'], 'PASS')
                    self.assertEqual(manifest['independent_arithmetic']['status'], 'PENDING')
                    self.assertIsNone(manifest['independent_arithmetic']['artifact'])
                    self.assertFalse((out / 'golden.bin').exists())
                    actual = np.load(out / 'reference.npy', allow_pickle=False)
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertGreater(actual.size, 0)
                    self.assertGreater(float(np.linalg.norm(actual)), 0)
                    module = bench.load_reference(family)
                    load = lambda name: np.load(out / (name + '.npy'), allow_pickle=False)
                    if family == 'attention':
                        expected = module.reference(*(bench.bf16_values(load(name))
                            for name in ('q_bf16', 'k_bf16', 'v_bf16')), load('allowed_mask'))
                    elif family == 'rmsnorm':
                        expected = module.reference(bench.bf16_values(load('hidden_bf16')),
                                                    bench.bf16_values(load('gamma_bf16')))
                    elif family == 'epilogue':
                        expected = module.reference(load('activations_fp32'), load('weights_fp32'))
                    else:
                        expected = module.reference(load('activations_fp32'), load('weights_fp32'),
                                                    bench.bf16_values(load('bias_bf16')))
                    np.testing.assert_array_equal(actual, expected)

    def test_resnet_exact_byte_references_derive_from_saved_image(self):
        with tempfile.TemporaryDirectory() as directory:
            for family in ('resnet_pool', 'resnet_patch_gather'):
                with self.subTest(family=family):
                    manifest = family_data.produce_family(family, directory, seed=23)
                    out = Path(directory) / family
                    channels, height, width = manifest['physical_shape']
                    if family == 'resnet_pool':
                        image = np.load(out / 'image_fp8.npy', allow_pickle=False)
                        padded = np.pad(image, ((1, 1), (1, 1), (0, 0)))
                        expected = np.zeros((height // 2, width // 2, channels), dtype=np.uint8)
                        for y in range(height // 2):
                            for x in range(width // 2):
                                expected[y, x] = padded[2*y:2*y + 3, 2*x:2*x + 3].max(axis=(0, 1))
                        self.assertEqual((out / 'golden.bin').read_bytes(), expected.tobytes())
                    else:
                        image = np.load(out / 'image_bytes.npy', allow_pickle=False)
                        expected = np.zeros((64, 9 * channels), dtype=np.uint8)
                        padded = np.pad(image, ((1, 1), (1, 1), (0, 0)))
                        for row in range(height * width):
                            y, x = divmod(row, width)
                            expected[row] = padded[y:y+3, x:x+3].reshape(-1)
                        self.assertEqual((out / 'golden.bin').read_bytes(), bench.pack_fp8_tiles(expected))
                    self.assertEqual(manifest['independent_arithmetic']['status'], 'PASS')

    def test_rerun_removes_obsolete_golden_and_preserves_program_companions(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'epilogue'
            out.mkdir()
            (out / 'golden.bin').write_bytes(b'obsolete device bytes')
            (out / 'golden_decoded.npy').write_bytes(b'obsolete device tensor')
            (out / 'program.bin').write_bytes(b'generator companion')
            family_data.produce_family('epilogue', directory)
            self.assertFalse((out / 'golden.bin').exists())
            self.assertFalse((out / 'golden_decoded.npy').exists())
            self.assertEqual((out / 'program.bin').read_bytes(), b'generator companion')
            self.assertIn('lib/golden/atlas_vector_arithmetic.py',
                          json.loads((out / 'manifest.json').read_text())['dependencies'])

    def test_invalid_seed_or_unknown_family_is_rejected(self):
        for seed in (-1, True, 2.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                family_data.build_case('projection_gemm', seed=seed)
        with self.assertRaises(ValueError):
            family_data.build_case('unknown')
        with self.assertRaises(ValueError):
            family_data.produce_families('/unused', families=['unknown'])
        with self.assertRaises(ValueError):
            family_data.admit_output_root(family_data.ROOT / 'unignored-data')

    def test_source_dependencies_and_output_paths_reject_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'target').mkdir()
            (root / 'link').symlink_to(root / 'target', target_is_directory=True)
            with self.assertRaises(ValueError):
                family_data.produce_family('resnet_pool', root / 'link')
        with tempfile.TemporaryDirectory() as directory:
            # Device execution is not needed even for the data producer's
            # independent arithmetic goldens.
            with patch('atlas_execution.Executor') as executor:
                family_data.produce_family('rne_add', directory)
            executor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
