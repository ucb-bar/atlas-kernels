"""PyTorch reference and real-shape bench case for the exact BF16 residual add fused into a projection.

TinyLlama's prefill o and down projections end with Step('add_rne'): Y = RNE_bf16(P + R) for the BF16 projection
P and the BF16 residual tile R, the host's FP32 add plus round-to-nearest-even. The FP32 reference is A @ W + R on
the unquantized operands; the golden is ordered_matmul, the exact rescale, and the exact sum rounded to BF16.
"""
import numpy as np
import torch

from atlas_epilogue import Epilogue, Step, TileInput
from atlas_vector_arithmetic import decode_bf16, ordered_matmul, rescale_bf16_array, round_bf16
import kernel_bench as bench

CASES = [dict(name='tl-prefill-o-proj', model='TL', shape=[128, 2048, 2048],
              note='prefill o_proj with the residual add (atlas-mlir tools/tiny_full/fusion.py role 3, exact_residual)')]


def reference(a, w, residual):
    return (torch.from_numpy(a) @ torch.from_numpy(w) + torch.from_numpy(residual)).numpy()


def build(spec, rng):
    m, k, n = spec['shape']
    a, w, (a_codes, a_scale), (w_codes, w_scale) = bench.projection_operands(spec, rng)
    residual_words = bench.bf16_words(rng.standard_normal((m, n), dtype=np.float32))
    residual = bench.bf16_values(residual_words)
    output_offset = m * k + k * n + 256
    extra_offset = output_offset + 2 * m * n + 256
    epilogue = Epilogue(m, n, output_offset, (Step('add_rne', TileInput(extra_offset)),), scratch_registers=(10, 11))
    source, arena, offset, extent = bench.stage_projection(m, k, n, a_codes, w_codes, a_scale + w_scale, epilogue,
                                                           bench.pack_bf16_tiles(residual_words))
    projected = rescale_bf16_array(ordered_matmul(a_codes, w_codes), a_scale + w_scale)
    golden = round_bf16(decode_bf16(projected) + decode_bf16(residual_words))
    return bench.Case(source, arena, offset, extent, reference(a, w, residual),
                      lambda raw: bench.bf16_values(bench.unpack_bf16_tiles(raw, m, n)),
                      bench.pack_bf16_tiles(golden), 'ordered_matmul + exact sum rounded to BF16',
                      dict(a_scale=a_scale, w_scale=w_scale))
