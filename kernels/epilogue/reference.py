"""PyTorch references and real-shape bench cases for the fused TinyLlama MLP epilogues.

gate: Y = SiLU(P) in BF16 (Step('silu')). up: Y = FP8(P * G / 2**s) for the BF16 gate tile G, written as FP8 tiles
with the down projection's consumer row stride (Step('mul'), atlas-mlir tools/tiny_full/fusion.py roles 4 and 5).
There is no independent Atlas-arithmetic golden for these steps; fidelity is against FP32 PyTorch.
"""
import numpy as np
import torch

from atlas_epilogue import Epilogue, Step, TileInput
import kernel_bench as bench

CASES = [
    dict(name='tl-prefill-gate-silu', model='TL', shape=[128, 2048, 2048], step='silu',
         note='prefill gate_proj panel with SiLU'),
    dict(name='tl-prefill-up-mul-fp8', model='TL', shape=[128, 2048, 2048], step='mul',
         note='prefill up_proj panel, multiply by the SiLU gate, FP8 output in the 5632-wide down input layout'),
]
WIDTH = 5632


def reference(a, w, gate=None):
    product = torch.from_numpy(a) @ torch.from_numpy(w)
    return (torch.nn.functional.silu(product) if gate is None else product * torch.from_numpy(gate)).numpy()


def build(spec, rng):
    m, k, n = spec['shape']
    a, w, (a_codes, a_scale), (w_codes, w_scale) = bench.projection_operands(spec, rng)
    output_offset = m * k + k * n + 256
    if spec['step'] == 'silu':
        epilogue = Epilogue(m, n, output_offset, (Step('silu'),), scratch_registers=(10, 11))
        source, arena, offset, extent = bench.stage_projection(m, k, n, a_codes, w_codes, a_scale + w_scale, epilogue)
        return bench.Case(source, arena, offset, extent, reference(a, w),
                          lambda raw: bench.bf16_values(bench.unpack_bf16_tiles(raw, m, n)), meta=dict(a_scale=a_scale, w_scale=w_scale))
    gate_words = bench.bf16_words(torch.nn.functional.silu(torch.from_numpy(rng.standard_normal((m, n), dtype=np.float32))).numpy())
    gate = bench.bf16_values(gate_words)
    expected = reference(a, w, gate)
    scale = bench.fp8_scale(expected)
    stride = 32 * WIDTH
    common = dict(output_dtype='fp8', output_scale_exponent=scale, output_row_stride=stride, scratch_registers=(10, 11))
    extra = output_offset + Epilogue(m, n, output_offset, **common).output_bytes + 256
    epilogue = Epilogue(m, n, output_offset, (Step('mul', TileInput(extra)),), **common)
    source, arena, offset, extent = bench.stage_projection(m, k, n, a_codes, w_codes, a_scale + w_scale, epilogue,
                                                           bench.pack_bf16_tiles(gate_words))

    def decode(raw):
        rows = [bench.unpack_fp8_tiles(raw[r * stride:r * stride + n * 32], 32, n) for r in range(m // 32)]
        return bench.fp8_values(np.concatenate(rows), scale)
    return bench.Case(source, arena, offset, extent, expected, decode,
                      meta=dict(a_scale=a_scale, w_scale=w_scale, output_scale=scale))
