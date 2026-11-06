"""Role, physical shape, exact rounding, and retained hook admission checks."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[3]))
import atlas_kernels_path  # noqa: E402,F401
from atlas_epilogue import Epilogue, Step, TileInput  # noqa: E402
from atlas_program_checks import check_hook_footprint  # noqa: E402
from atlas_program_qwen_fused import ChannelRNE, assembly, layout, stage  # noqa: E402
from qualify_qwen_fusions import exact_sum  # noqa: E402


class QwenFusedTests(unittest.TestCase):
    def test_roles_full_shapes_scales_and_selected_residency(self):
        for role in ('q', 'k', 'v', 'o', 'down'):
            k = 8960 if role == 'down' else 1536
            with self.subTest(role=role):
                plan = layout(role, 32, k, 128)
                source = assembly(role, 32, k, 128, -14)
                plain = assembly(role, 32, k, 128, -14, fused=False)
                self.assertEqual(plan['resident_a'], k == 1536)
                self.assertEqual(plan['fused_dma']['load_bytes'] - plan['plain_dma']['load_bytes'], 8192)
                self.assertEqual(plan['fused_dma']['store_bytes'], 8192)
                self.assertIn('vload m56,', source)
                self.assertNotIn('vload m56,', plain)
                self.assertIn('vli.all m4, 14464', source)
                self.assertEqual(source.count('vmatmul.acc.mxu0'), k // 32 - 1)
        self.assertEqual(assembly('q', 32, 1536, 128, -14), assembly('k', 32, 1536, 128, -14))
        self.assertEqual(assembly('q', 32, 1536, 128, -14), assembly('v', 32, 1536, 128, -14))

    def test_reject_unqualified_shapes_role_scales_and_addresses(self):
        for role, shape in [('logits', (32,1536,128)), ('q', (32,8960,128)),
                            ('down',(32,1536,128)), ('q',(1,1536,128)),
                            ('q',(64,1536,128)), ('q',(32,1536,256)), ('q',(32.0,1536,128))]:
            with self.subTest(role=role, shape=shape), self.assertRaises(ValueError):
                layout(role, *shape)
        for scale in (True, None, -127, 128, -14.0):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                assembly('q', 32, 1536, 128, scale)
        for base in (True, 1, -32, 2**32 - 32):
            with self.subTest(base=base), self.assertRaises(ValueError):
                layout('q', 32, 1536, 128, dram_base=base)

    def test_channel_contract_survives_scratch_remap_and_clobber_guard(self):
        plan = layout('q', 32, 1536, 128)
        hook = ChannelRNE(32,128,plan['output_offset'],
                          (Step('add_rne', TileInput(plan['operand_offset'], 'channel')),))
        remapped = replace(hook, scratch_registers=(27,28))
        self.assertEqual(remapped.steps[0].operand.layout, 'channel')
        self.assertIn('vload m56,', '\n'.join(remapped.apply(12)))
        with self.assertRaisesRegex(ValueError, 'overwrites recipe'):
            check_hook_footprint('vload m6, 0(x1)\n', remapped.apply(12))
        with self.assertRaises(ValueError):
            ChannelRNE(32,128,plan['output_offset'],
                       (Step('add', TileInput(plan['operand_offset'], 'channel')),))
        # The unchanged default deliberately still refuses channel add_rne.
        with self.assertRaises(ValueError):
            Epilogue(32,128,plan['output_offset'], hook.steps)

    def test_bias_broadcast_stage_and_operand_layout_admission(self):
        a = np.zeros((32,1536), dtype=np.uint8)
        w = np.zeros((1536,128), dtype=np.uint8)
        bias = np.arange(128, dtype=np.uint16)
        _, arena, plan = stage('q', a, w, bias, -14)
        from kernel_bench import unpack_bf16_tiles
        actual = unpack_bf16_tiles(arena[plan['operand_offset']:plan['operand_offset'] + 8192],32,128)
        np.testing.assert_array_equal(actual, np.broadcast_to(bias, (32,128)))
        for role, addend in [('q', actual), ('o', bias)]:
            with self.subTest(role=role), self.assertRaises(ValueError):
                stage(role,a,w,addend,-14)
        a[0,0] = 0x7f
        with self.assertRaises(ValueError):
            stage('q',a,w,bias,-14)

    def test_independent_exact_sum_ties_subnormals_overflow_and_nan(self):
        a = np.array([0x3f80,0x3f81,0xbf80,1,0x7f7f,0x7f80,0x8000], dtype=np.uint16)
        b = np.array([0x3b80,0x3b80,0x3f80,1,0x7f7f,0xff80,0x8000], dtype=np.uint16)
        np.testing.assert_array_equal(exact_sum(a,b),
            np.array([0x3f80,0x3f82,0,2,0x7f80,0x7fc0,0x8000],dtype=np.uint16))


if __name__ == '__main__':
    unittest.main()
