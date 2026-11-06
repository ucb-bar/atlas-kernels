"""Opt-in DMA serialization preserves default assembly and drains each transfer."""
import io
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).absolute().parents[3]
sys.path.insert(0,str(ROOT))
import atlas_kernels_path  # noqa: E402,F401
from atlas_attention_sync import synchronous_dma  # noqa: E402
from atlas_attention_v2_program import AttentionV2Config, assembly_v2, stage_v2  # noqa: E402


class SynchronousDmaTest(unittest.TestCase):
    def test_rewrite_has_exact_transfer_waits_and_scalar_halt_guard(self):
        source='dma.load x2, x3, x4, ch1\nvload m0, 0(x2)\ndma.store x1, x2, x3, ch0\ndma.wait ch0\ndma.wait ch1\ndelay 128\necall\n'
        expected='dma.load x2, x3, x4, ch1\ndma.wait ch1\nvload m0, 0(x2)\ndma.store x1, x2, x3, ch0\ndma.wait ch0\ndma.wait ch0\ndma.wait ch1\ndelay 128\nnop\necall\n'
        self.assertEqual(synchronous_dma(source),expected)
        for changed in (source.replace('128','127'),source+'nop\n',source.replace('ch1\nvload','ch2\nvload')):
            with self.assertRaisesRegex(ValueError,'changed'):
                synchronous_dma(changed)

    @unittest.skipUnless(os.environ.get('ATLAS_NPU_MODEL_ROOT'),'requires explicit npu_model root')
    def test_default_program_words_remain_identical_and_variant_assembles(self):
        import numpy as np
        sys.path.insert(0,os.environ['ATLAS_NPU_MODEL_ROOT'])
        from npu_model.util.converter import input_to_program
        c=AttentionV2Config(2,3,2,1,mask='none')
        q=np.full((2,2,64),0x3f00,dtype='<u2');kv=np.full((1,3,64),0x3e80,dtype='<u2')
        _,offsets=stage_v2(c,q,kv,kv)
        original=assembly_v2(c,offsets)
        original_words=input_to_program(io.StringIO(original)).assemble()
        variant=synchronous_dma(original)
        self.assertEqual(input_to_program(io.StringIO(assembly_v2(c,offsets))).assemble(),original_words)
        variant_words=input_to_program(io.StringIO(variant)).assemble()
        dma_count=sum(line.startswith(('dma.load ','dma.store ')) for line in original.splitlines())
        self.assertEqual(len(variant_words),len(original_words)+dma_count+1)
        self.assertEqual([line for line in variant.splitlines() if not line.startswith('dma.wait ') and line!='nop'],
                         [line for line in original.splitlines() if not line.startswith('dma.wait ') and line!='nop'])


if __name__=='__main__':unittest.main()
