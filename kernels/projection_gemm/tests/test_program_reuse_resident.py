"""Admission, symbolic semantic proofs, mutations, and assembler checks for reuse recipes."""
import importlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[3]))
import atlas_kernels_path  # noqa: E402,F401
import atlas_program as program
import atlas_program_reuse_blocked as blocked
import atlas_program_reuse_smol_connector as connector
import atlas_program_reuse_stream as stream
import atlas_program_reuse_tiny as tiny
import atlas_program_reuse_tiny_breadth as breadth
from atlas_program_reuse_resident import admission, assembly, dma_load_bytes
from reuse_proof import prove
from reuse_proof import prove as prove_program


class ResidentReuseTests(unittest.TestCase):
    def test_target_capacity_and_compulsory_dma(self):
        r = admission(128,768,768,688384,-19)
        self.assertEqual(r['a_byte_range'], [262144,360448])
        self.assertEqual(r['b_column_byte_range'], [360448,385024])
        self.assertEqual(r['resident_bytes'], 122880)
        self.assertEqual(dma_load_bytes(128,768,768), 688128)
        self.assertEqual(r['dma_store_bytes'], 196608)
    def test_complete_address_and_k_semantics(self):
        for shape in ((64,32,32),(64,64,64),(128,768,768)):
            m, k, n = shape
            with self.subTest(shape=shape):
                source = assembly(m, k, n, m*k+k*n+256, -19)
                counts = prove(source, m, k, n, -19, m*k+k*n)['counts']
                self.assertEqual(counts['vli.all'], 1)
                self.assertEqual(counts['vtrpose.xlu'], k*n//1024)
                self.assertEqual(counts['vmatpush.weight.mxu0'], (m//64)*(k//32)*(n//32))
    def test_reject_unsupported_layouts(self):
        for args in ((32,768,768,0,-19),(128,2048,768,0,-19),
                     (128,768,768,688383,-19),(128,768,768,688384,None),
                     (128,768,768,688384,-127)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError): admission(*args)
    def test_original_rescale_patch_protocol(self):
        # Optional dependency supplied by the existing model environment.
        from npu_model.util.converter import input_to_program
        words = []
        for exponent in (-126,-19,0,127):
            p = input_to_program(io.StringIO(assembly(128,768,768,688384,exponent)))
            raw = p.assemble()
            self.assertLessEqual(len(raw), 8192)
            indices = [i for i, w in enumerate(raw) if (w & ((1<<23)-1)) == 0x25f]
            self.assertEqual(len(indices), 1)
            index = indices[0]
            self.assertEqual(raw[index], ((exponent+127)<<23)|0x25f)
            words.append((index,raw))
        for index, raw in words:
            self.assertEqual(index, words[0][0])
            changed = [i for i,(a,b) in enumerate(zip(raw,words[0][1])) if a!=b]
            self.assertTrue(changed in ([],[index]))


class BlockedReuseTests(unittest.TestCase):
    def test_strict_admission(self):
        for shape in ((224., 2304, 256), (225, 2304, 256), (32, 32, 32),
                      (32, 2048, 1024), (128., 768, 3072), (129, 768, 3072),
                      (64, 12288, 960), (128, 768, 768), (128, 3072, 768), (64, 736, 32)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                blocked.layout(*shape)
        for m, k, n in ((224, 2304, 256), (64, 320, 320)):
            offset = m * k + k * n + 256
            for bad_offset in (0, offset + 1, float(offset), True):
                with self.subTest(offset=bad_offset), self.assertRaises(ValueError):
                    blocked.assembly(m, k, n, bad_offset, -13)
            for exponent in (None, -127, 128, True, 0.0):
                with self.subTest(exponent=exponent), self.assertRaises(ValueError):
                    blocked.assembly(m, k, n, offset, exponent)


class ConnectorTest(unittest.TestCase):

    def test_rejected_inputs(self):
        for shape in ((64., 12288, 960), (32, 12288, 960), (64, 12256, 960)):
            with self.assertRaises(ValueError):
                connector.layout(*shape)
        args = (64, 12288, 960, 12583168)
        for exponent in (None, True, -127, 128):
            with self.assertRaises(ValueError):
                connector.assembly(*args, exponent)


class StreamReuseTests(unittest.TestCase):
    def test_order_addresses_capacity_and_compulsory_traffic(self):
        for shape in ((128,3072,768), (32,32,32), (64,64,96), (96,3072,64), (128,2048,32)):
            for exponent in (None, -15):
                m,k,n = shape
                with self.subTest(shape=shape, exponent=exponent):
                    result = prove(stream.assembly(m,k,n,m*k+k*n+256,exponent),m,k,n,exponent,m*k+k*n)
                    self.assertEqual((result['load_bytes'],result['store_bytes']), (m*k+k*n,2*m*n))

    def test_row2_bank_transition_and_disjoint_regions(self):
        plan = stream.layout(128,3072,768)
        self.assertEqual(plan, dict(a_bank1_bytes=262144,a_bank0_bytes=131072,
                                   b_word=32768,b_bytes=98304,output_word=57344))
        addresses = [stream.a_word(i)*4 for i in range(0,393216,1024)]
        self.assertEqual(len(set(addresses)),384)
        self.assertEqual(addresses[255:257],[523264,0])
        self.assertTrue(all(a >= 262144 or a + 1024 <= 131072 for a in addresses))

    def test_refuses_overflow_and_bad_abi(self):
        for shape in ((160,3072,768),(128,4096,768),(0,32,32),(32,31,32)):
            with self.assertRaises(ValueError):
                stream.assembly(*shape,10000000)
        for offset in (0,8193,2**32-32):
            with self.assertRaises(ValueError):
                stream.assembly(64,64,64,offset)


SHAPE = (128, 2048, 2048)
OUTPUT = 4456704


def prove_tiny(source, exponent=-14):
    return prove_program(source, *SHAPE, exponent)


class TinyReuseTests(unittest.TestCase):


    def test_negative_controls_detect_semantic_breakage(self):
        source = tiny.assembly(*SHAPE, OUTPUT, -14)
        changes = {
            'A bank alias': ('li x1, 0x10000', 'li x1, 0'),
            'incomplete preload': ('li x21, 256', 'li x21, 255'),
            'B DRAM stride': ('addi x20, x20, 1024', 'addi x20, x20, 2048'),
            'missing transpose': ('vtrpose.xlu m1, m1', 'nop'),
            'missing cached transpose': ('vstore m1, 0(x19)', 'nop'),
            'K order': ('addi x7, x7, 256', 'addi x7, x7, 512'),
            'accumulator alias': ('vmatmul.acc.mxu0 acc1', 'vmatmul.acc.mxu0 acc0'),
            'reset continuation': ('vmatmul.acc.mxu0', 'vmatmul.mxu0'),
            'wrong pop': ('vmatpop.bf16.acc.mxu0 m2, acc1', 'vmatpop.bf16.acc.mxu0 m2, acc0'),
            'missing scale': ('vmul.bf16 m2, m2, m4', 'nop'),
            'wrong scale': ('vli.all m4, 14464', 'vli.all m4, 14592'),
            'output scratch alias': ('li x3, 0x4000', 'li x3, 0x3f00'),
            'swapped BF16 halves': ('vstore m3, 0(x9)', 'vstore m2, 0(x9)'),
            'output row stride': ('li x22, 131072', 'li x22, 2048'),
            'output column alias': ('add x12, x12, x8', 'add x12, x12, x0'),
            'undrained DMA': ('dma.wait ch0', 'nop'),
            'unwaited prefetch': ('dma.wait ch1', 'nop'),
            'wrong DRAM page': ('li x30, 1', 'li x30, 0'),
        }
        for name, (old, new) in changes.items():
            self.assertIn(old, source)
            with self.subTest(control=name), self.assertRaises((AssertionError, KeyError)):
                prove_tiny(source.replace(old, new))

    def test_admission(self):
        for shape in ((64, 2048, 2048), (128, 1024, 2048), (128, 2048, 512), (160, 2048, 2048)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                tiny.assembly(*shape, OUTPUT, -14)
            with self.assertRaises(ValueError):
                tiny.dma_load_bytes(*shape)
        for exponent in (True, False, 0.0, float("inf"), float("nan"), -127, 128):
            with self.assertRaises(ValueError):
                tiny.assembly(*SHAPE, OUTPUT, exponent)

        for offset in (0, OUTPUT - 1, OUTPUT + 1, float(OUTPUT), True):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                tiny.assembly(*SHAPE, offset, -14)


class BreadthTests(unittest.TestCase):
    def test_admission(self):
        for shape in [(32.,2048,1024), (32,2048,1024.), (32,2048,1056)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                breadth.allocation(*shape)
        for args in [(128,2048,2048,4456704,-14),(128,5632,1024,0,-18),
                     (32,2048,2048,4260096,False),(32,2048,2048,4260096,-127)]:
            with self.assertRaises(ValueError):breadth.assembly(*args)


def dispatch_assembly(shape, exponent=-13):
    m, k, n = shape
    return program.assembly(m, k, n, m * k + k * n + 256, exponent, traffic_reuse=True)


class ReuseProofTests(unittest.TestCase):
    def test_all_dispatch_shapes(self):
        for m, k, n in program._TRAFFIC_REUSE_MODULES:
            with self.subTest(shape=(m, k, n)):
                result = prove(dispatch_assembly((m, k, n)), m, k, n, -13)
                self.assertEqual(result['store_bytes'], 2 * m * n)

    def test_optional_direct_rescale(self):
        for shape, name in program._TRAFFIC_REUSE_MODULES.items():
            if name not in ('atlas_program_reuse_tiny', 'atlas_program_reuse_tiny_breadth',
                            'atlas_program_reuse_stream'):
                continue
            m, k, n = shape
            source = importlib.import_module(name).assembly(m, k, n, m * k + k * n + 256, None)
            with self.subTest(shape=shape):
                prove(source, m, k, n, None)

    def test_one_static_rescale_patch_for_every_shape(self):
        for shape in program._TRAFFIC_REUSE_MODULES:
            original = dispatch_assembly(shape, 0).splitlines()
            sites = [i for i, line in enumerate(original) if line.startswith('vli.all ')]
            with self.subTest(shape=shape):
                self.assertEqual(len(sites), 1)
                for exponent in (-126, -13, 127):
                    expected = original.copy()
                    expected[sites[0]] = f'vli.all m4, {(exponent + 127) << 7}'
                    self.assertEqual(expected, dispatch_assembly(shape, exponent).splitlines())

    def reject_mutations(self, shape, changes):
        source = dispatch_assembly(shape)
        for old, new in changes:
            self.assertIn(old, source)
            with self.subTest(shape=shape, mutation=old), self.assertRaises((AssertionError, KeyError)):
                prove(source.replace(old, new), *shape, -13)

    def test_blocked_negative_controls(self):
        for shape in ((224, 2304, 256), (128, 2560, 960)):
            self.reject_mutations(shape, [
                ('vtrpose.xlu m1, m1', 'nop'),
                ('vmatmul.acc.mxu0', 'vmatmul.mxu0'),
                ('acc1', 'acc0'),
                ('vstore m3, 0(x9)', 'vstore m2, 0(x9)'),
                ('add x22, x22, x16', 'add x22, x22, x8'),
                ('addi x19, x19, 256', 'addi x19, x19, 512'),
                ('dma.wait ch0', 'nop'),
                ('dma.wait ch1', 'nop'),
                ('vmul.bf16 m2, m2, m4', 'nop')])

    def test_compact_resnet_pair_loop_negative_controls(self):
        self.reject_mutations((12544, 160, 64), [
            ('add x23, x23, x25', 'add x23, x23, x8'),
            ('addi x24, x24, -1', 'addi x24, x24, -2'),
            ('add x7, x7, x23', 'add x7, x7, x0')])

    def test_tiny_and_prefetch_negative_controls(self):
        for shape in ((128, 2048, 256), (32, 5632, 1024), (128, 5632, 1024)):
            changes = [
                ('vtrpose.xlu m1, m1', 'nop'),
                ('vmatmul.acc.mxu0', 'vmatmul.mxu0'),
                ('vstore m3, 0(x9)', 'vstore m2, 0(x9)'),
                ('dma.wait ch0', 'nop'),
                ('dma.wait ch1', 'nop'),
                ('vmul.bf16 m2, m2, m4', 'nop'),
                ('li x30, 1', 'li x30, 0')]
            if shape[0] != 32:
                changes += [('add x23, x23, x22', 'add x23, x23, x8')]
            if shape[1] == 5632:
                changes += [('dma.config x30, ch1', 'nop')]
            if shape == (128, 5632, 1024):
                changes += [('addi x7, x7, 256', 'addi x7, x7, 512'), ('add x12, x12, x22', 'add x12, x12, x0')]
            self.reject_mutations(shape, changes)

    def test_column_prefetch_ownership(self):
        for shape in ((64, 736, 2048), (128, 768, 768), (128, 2048, 2048),
                      (128, 3072, 768), (12544, 160, 64)):
            self.reject_mutations(shape, [
                ('dma.load x24, x20, x5, ch1', 'dma.load x19, x20, x5, ch1'),
                ('addi x24, x19, 256', 'addi x24, x19, 512'),
                ('dma.wait ch1', 'nop')])

    def test_scale_and_memory_boundary_negative_controls(self):
        self.reject_mutations((128, 2048, 256), [
            ('vli.all m4, 14592', 'vli.all m4, 14720'),
            ('vli.all m4, 14592', 'vli.all m4, 14592\nvli.all m4, 14592'),
            # Correct values scaled too late must not validate earlier stores.
            ('vmul.bf16 m2, m2, m4\ndelay 66\nvstore m2, 0(x3)\ndelay 34\nvstore m3, 0(x9)',
             'vstore m2, 0(x3)\ndelay 34\nvstore m3, 0(x9)\ndelay 34\nvmul.bf16 m2, m2, m4'),
            ('li x1, 0x10000', 'li x1, 0x20000'),
            ('li x3, 0x4000', 'li x3, 0xff00'),
            ('li x9, 0x4100', 'li x9, 0x10000')])
        self.reject_mutations((32, 5632, 1024), [
            ('li x19, 65536', 'li x19, 65280')])  # 4-KiB DMA crosses banks.


if __name__ == '__main__':
    unittest.main()
