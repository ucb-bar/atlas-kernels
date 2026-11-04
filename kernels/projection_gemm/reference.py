"""PyTorch reference and real-shape bench cases for the FP8 projection GEMM.

The kernel computes Y = A @ W for FP8 E4M3 A [M, K] and W [K, N] with power-of-two scales, ascending-K BF16
accumulation on MXU0, and one BF16 output rescale by 2**(a_scale + w_scale). The FP32 reference is the same
product on the unquantized inputs; the golden is lib/golden ordered_matmul plus the exact rescale.
"""
import torch

from atlas_vector_arithmetic import ordered_matmul, rescale_bf16_array
import kernel_bench as bench

# One representative physical panel per model, plus the decode padding and blocked-recipe variants.
CASES = [
    dict(name='tl-prefill-lm-head', model='TL', shape=[128, 2048, 2048], note='prefill lm_head panel, tiny recipe'),
    dict(name='tl-decode-lm-head', model='TL', shape=[32, 2048, 2048], logical_rows=1,
         note='decode lm_head panel, one row padded to 32, tiny_breadth recipe'),
    dict(name='sv-vision-q', model='SV', shape=[128, 768, 768], note='vision q_proj panel, resident recipe (bias on host)'),
    dict(name='sv-text-q', model='SV', shape=[128, 960, 960], note='VLM prefix q_proj panel, blocked recipe'),
]


def reference(a, w):
    """FP32 PyTorch product of the unquantized operands."""
    return (torch.from_numpy(a).float() @ torch.from_numpy(w).float()).numpy()


def build(spec, rng):
    m, k, n = spec['shape']
    a, w, (a_codes, a_scale), (w_codes, w_scale) = bench.projection_operands(spec, rng)
    source, arena, offset, extent = bench.stage_projection(m, k, n, a_codes, w_codes, a_scale + w_scale)
    golden = rescale_bf16_array(ordered_matmul(a_codes, w_codes), a_scale + w_scale)
    return bench.Case(source, arena, offset, extent, reference(a, w),
                      lambda raw: bench.bf16_values(bench.unpack_bf16_tiles(raw, m, n)),
                      bench.pack_bf16_tiles(golden), 'ordered_matmul', dict(a_scale=a_scale, w_scale=w_scale))
