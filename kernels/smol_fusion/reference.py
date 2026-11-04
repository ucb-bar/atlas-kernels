"""PyTorch reference and real-shape bench case for SmolVLA's fused vision fc1 bias and GELU.

The fc1-bias-gelu-fp8 profile ends the 128x768x3072 vision fc1 panel with a per-channel BF16 bias add and
GELU (tanh), written as FP8 tiles that fc2 reads (atlas-kernels smol_fusion_fp8.py stage_fc1_fp8, reproduced
here with synthetic operands). There is no independent Atlas-arithmetic golden; fidelity is against FP32 PyTorch.
"""
import numpy as np
import torch

from atlas_epilogue import Epilogue, Step, TileInput
from smol_fusion_program import channel_tiles
import kernel_bench as bench

CASES = [dict(name='sv-vision-fc1-bias-gelu', model='SV', shape=[128, 768, 3072],
              note='vision fc1 panel with bias and GELU (tanh), FP8 output for fc2')]


def reference(a, w, bias):
    product = torch.from_numpy(a) @ torch.from_numpy(w) + torch.from_numpy(bias)
    return torch.nn.functional.gelu(product, approximate='tanh').numpy()


def build(spec, rng):
    m, k, n = spec['shape']
    a, w, (a_codes, a_scale), (w_codes, w_scale) = bench.projection_operands(spec, rng)
    bias_words = bench.bf16_words(0.1 * rng.standard_normal(n, dtype=np.float32))
    expected = reference(a, w, bench.bf16_values(bias_words))
    scale = bench.fp8_scale(expected)
    output_offset = m * k + k * n + 256
    extra = output_offset + m * n + 256
    epilogue = Epilogue(m, n, output_offset, (Step('add', TileInput(extra, 'channel')), Step('gelu_tanh')),
                        output_dtype='fp8', output_scale_exponent=scale, scratch_registers=(10, 11))
    source, arena, offset, extent = bench.stage_projection(m, k, n, a_codes, w_codes, a_scale + w_scale, epilogue,
                                                           channel_tiles(bias_words.tobytes(), n) + b'Z' * 256)
    return bench.Case(source, arena, offset, extent, expected,
                      lambda raw: bench.fp8_values(bench.unpack_fp8_tiles(raw, m, n), scale),
                      meta=dict(a_scale=a_scale, w_scale=w_scale, output_scale=scale))
