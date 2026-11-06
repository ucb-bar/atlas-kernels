"""Wide attention admission, tiled codecs, fidelity, and execution agreement."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[3]))
import atlas_kernels_path  # noqa: E402,F401
import numpy as np
from atlas_attention_wide_program import (
    AttentionWideConfig, assembly_wide, pack_output_wide, stage_wide, unpack_output_wide, query_key_counts,
)


class WideAttentionLayoutTests(unittest.TestCase):
    def test_key_loop_bounds_come_from_static_geometry(self):
        import struct
        for rows, keys, start, mask in ((32, 32, 0, 'causal'), (1, 33, 32, 'causal'),
                                       (1, 34, 33, 'causal'), (65, 65, 0, 'causal'),
                                       (33, 65, 7, 'causal'), (33, 65, 0, 'none')):
            c = AttentionWideConfig(rows, keys, 12, 2, mask=mask, query_start=start)
            words = [np.zeros(shape, dtype=np.uint16) for shape in
                     ((12, rows, 128), (2, keys, 128), (2, keys, 128))]
            arena, offsets = stage_wide(c, *words)
            counts = struct.unpack_from('<' + 'I' * c.query_tiles, arena, offsets['key_counts'])
            self.assertEqual(query_key_counts(c), counts)
            source = assembly_wide(c, offsets)
            self.assertNotIn('lw x18, 0(x17)', source)
            self.assertIn(f'li x18, {counts[-1]}', source)

    def test_output_layout_roundtrip_with_row_padding_and_gqa(self):
        for dim in (128, 256):
            for fp8 in (False, True):
                with self.subTest(dim=dim, fp8=fp8):
                    c = AttentionWideConfig(33, 35, 4, 2, head_dim=dim, output_fp8=fp8)
                    dtype = np.uint8 if fp8 else np.uint16
                    value = np.arange(4 * 33 * dim).astype(dtype).reshape(4, 33, dim)
                    packed = pack_output_wide(c, value)
                    self.assertEqual(len(packed), 2 * 4 * dim * (32 if fp8 else 64))
                    np.testing.assert_array_equal(unpack_output_wide(c, packed), value)

    def test_reject_physical_capacity_and_unsupported_paths(self):
        for kwargs in (
            dict(head_dim=64), dict(head_dim=256, keys=1025),
            dict(head_dim=128, keys=2049), dict(head_dim=128, query_rows=1057, keys=2048),
            dict(head_dim=256, query_rows=4097), dict(head_dim=128, mask='arbitrary'),
            dict(head_dim=128, packed_decode=True), dict(head_dim=128, query_start=-1),
            dict(head_dim=128, query_rows=33, keys=32, mask='causal'),
            dict(head_dim=128, query_heads=3, kv_heads=2),
        ):
            args = dict(query_rows=1, keys=1, query_heads=2, kv_heads=1) | kwargs
            with self.subTest(args=args), self.assertRaises(ValueError):
                AttentionWideConfig(**args)
        self.assertEqual(AttentionWideConfig(1, 2048, 2, 1, head_dim=128).key_tiles, 64)
        self.assertEqual(AttentionWideConfig(1, 1024, 2, 1, head_dim=256).key_tiles, 32)

    def test_default_scale_is_bf16_round_to_nearest_even(self):
        self.assertEqual(AttentionWideConfig(1, 1, 1, 1, head_dim=128).scale, 0.08837890625)
        self.assertEqual(AttentionWideConfig(1, 1, 1, 1, head_dim=256).scale, 0.0625)


@unittest.skipUnless(os.environ.get('ATLAS_NPU_MODEL_ROOT'), 'requires pinned npu_model')
class WideAttentionExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from atlas_execution import Executor
        cls.executor = Executor(os.environ['ATLAS_NPU_MODEL_ROOT'])

    def single_key_case(self, dim, fp8=False, mask='causal'):
        import kernel_bench as bench
        # With Q=K=0, the sole visible key has score zero and yields exactly V.
        # V varies across every output panel and GQA group.
        c = AttentionWideConfig(1, 1, 4, 2, head_dim=dim, mask=mask, output_fp8=fp8)
        q = np.zeros((4, 1, dim), dtype=np.uint16)
        k = np.zeros((2, 1, dim), dtype=np.uint16)
        v_values = ((np.arange(2 * dim).reshape(2, 1, dim) % 16 - 8) / 8).astype(np.float32)
        v_values[1] *= 2
        v = bench.bf16_words(v_values)
        arena, offsets = stage_wide(c, q, k, v)
        expected = np.repeat(bench.encode_fp8(v_values) if fp8 else v, 2, axis=0)
        return c, arena, offsets, assembly_wide(c, offsets), pack_output_wide(c, expected)

    @unittest.skipUnless(os.environ.get('ATLAS_MLIR_BIN'), 'requires explicit Atlas compiler tools')
    def test_qwen_d128_passes_verified_compiler_and_assembler_crosscheck(self):
        import tempfile
        sys.path.insert(0, str(Path(__file__).absolute().parents[3] / 'coverage'))
        import bench_model
        p = dict(rows=32, keys=32, heads=12, kv_heads=2, head_dim=128, mask='causal',
                 query_start=0, output_fp8=False)
        entry = dict(family='attention', params=p, key=bench_model.launch_key('attention', p))
        with tempfile.TemporaryDirectory() as temporary:
            built = bench_model.build(entry, Path(temporary), os.environ['ATLAS_NPU_MODEL_ROOT'])
            result = bench_model.program_identity(built, Path(os.environ['ATLAS_MLIR_BIN']), 'verified')
        self.assertEqual(result['backend']['tier'], 'verified')
        self.assertTrue(result['backend']['words_equal_assembler'])
        self.assertTrue(built.source.endswith('delay 128\nnop\necall\n'))

    def test_runtime_count_metadata_cannot_expand_resident_loop(self):
        import struct
        import kernel_bench as bench
        c, arena, offsets, source, expected = self.single_key_case(128)
        altered = bytearray(arena)
        struct.pack_into('<I', altered, offsets['key_counts'], 0xffffffff)
        result = self.executor.execute(source, bytes(altered), offsets['output'], offsets['output_bytes'])
        cycles, timed = bench.timed_cycles(os.environ['ATLAS_NPU_MODEL_ROOT'], source, bytes(altered), 1_000_000)
        self.assertGreater(cycles, 0)
        self.assertEqual(result.output, expected)
        self.assertEqual(timed[offsets['output']:offsets['output'] + offsets['output_bytes']], expected)

    def test_single_visible_key_is_exact_v_for_both_widths_and_outputs(self):
        for dim in (128, 256):
            for fp8 in (False, True):
                for mask in ('none', 'causal'):
                    with self.subTest(dim=dim, fp8=fp8, mask=mask):
                        c, arena, offsets, source, expected = self.single_key_case(dim, fp8, mask)
                        result = self.executor.execute(source, arena, offsets['output'], offsets['output_bytes'])
                        self.assertEqual(result.output, expected)

    def test_random_causal_tail_with_multiple_kv_heads_and_query_start(self):
        import kernel_bench as bench
        import torch
        rng = np.random.default_rng(7)
        words = [bench.bf16_words(rng.standard_normal(shape, dtype=np.float32))
                 for shape in ((4, 33, 256), (2, 65, 256), (2, 65, 256))]
        q, k, v = map(bench.bf16_values, words)
        c = AttentionWideConfig(33, 65, 4, 2, head_dim=256, mask='causal', query_start=7,
                                q_exponent=bench.fp8_scale(q), k_exponent=bench.fp8_scale(k),
                                v_exponent=bench.fp8_scale(v))
        arena, offsets = stage_wide(c, *words)
        result = self.executor.execute(assembly_wide(c, offsets), arena,
                                       offsets['output'], offsets['output_bytes'])
        scores = torch.from_numpy(q) @ torch.from_numpy(np.repeat(k, 2, axis=0)).transpose(1, 2)
        scores *= c.scale
        visible = torch.arange(65)[None, :] <= 7 + torch.arange(33)[:, None]
        scores.masked_fill_(~visible, float('-inf'))
        reference = (torch.softmax(scores, -1) @ torch.from_numpy(np.repeat(v, 2, axis=0))).numpy()
        actual = bench.bf16_values(unpack_output_wide(c, result.output))
        fidelity = bench.fidelity(actual, reference)
        self.assertGreater(fidelity['cosine'], 0.99)
        self.assertLess(fidelity['relative_error'], 0.12)

    def test_small_random_gqa_fidelity_and_exact_timed_agreement(self):
        import kernel_bench as bench
        import torch
        rng = np.random.default_rng(19)
        words = [bench.bf16_words(rng.standard_normal(shape, dtype=np.float32))
                 for shape in ((4, 3, 128), (2, 5, 128), (2, 5, 128))]
        q, k, v = map(bench.bf16_values, words)
        c = AttentionWideConfig(3, 5, 4, 2, head_dim=128,
                                q_exponent=bench.fp8_scale(q), k_exponent=bench.fp8_scale(k),
                                v_exponent=bench.fp8_scale(v))
        arena, offsets = stage_wide(c, *words)
        source = assembly_wide(c, offsets)
        result = self.executor.execute(source, arena, offsets['output'], offsets['output_bytes'])
        scores = torch.from_numpy(q) @ torch.from_numpy(np.repeat(k, 2, axis=0)).transpose(1, 2)
        probabilities = torch.softmax(scores * c.scale, -1)
        # Ensure the seeded operands exercise a nonuniform multi-key softmax.
        self.assertGreater(float(probabilities.max() - probabilities.min()), 0.1)
        reference = (probabilities @ torch.from_numpy(np.repeat(v, 2, axis=0))).numpy()
        actual = bench.bf16_values(unpack_output_wide(c, result.output))
        fidelity = bench.fidelity(actual, reference)
        self.assertGreater(fidelity['cosine'], 0.99)
        self.assertLess(fidelity['relative_error'], 0.12)
        cycles, timed = bench.timed_cycles(os.environ['ATLAS_NPU_MODEL_ROOT'], source, arena, 1_000_000)
        self.assertGreater(cycles, 0)
        self.assertEqual(timed[offsets['output']:offsets['output'] + offsets['output_bytes']], result.output)

    def test_timed_256_head_dma_segments_match_exact_functional_result(self):
        from kernel_bench import timed_cycles
        c, arena, offsets, source, expected = self.single_key_case(256)
        result = self.executor.execute(source, arena, offsets['output'], offsets['output_bytes'])
        cycles, timed = timed_cycles(os.environ['ATLAS_NPU_MODEL_ROOT'], source, arena, 1_000_000)
        self.assertGreater(cycles, 0)
        self.assertEqual(result.output, expected)
        self.assertEqual(timed[offsets['output']:offsets['output'] + offsets['output_bytes']], expected)


if __name__ == '__main__':
    unittest.main()
