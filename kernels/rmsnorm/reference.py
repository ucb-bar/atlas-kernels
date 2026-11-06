"""PyTorch reference and real-shape bench cases for TinyLlama's device RMSNorm.

The standalone launch loads BF16 hidden tiles H [M, 2048] and writes FP8 tiles of RMSNorm(H) * gamma / 2**s
(eps defaults to 1e-5), the q/k/v or gate/up FP8 input. Staging follows atlas-mlir tools/tiny_full/norm.py prepare_norm.
There is no independent Atlas-arithmetic golden; fidelity is against FP32 PyTorch.
"""
import numpy as np
import torch

from atlas_norm_postpass import standalone_source
from atlas_rmsnorm_postpass import MASK_BYTES
import kernel_bench as bench

CASES = [
    dict(name='tl-prefill-norm', model='TL', shape=[128, 2048], note='prefill input_layernorm, 128 rows'),
    dict(name='tl-decode-norm', model='TL', shape=[32, 2048], logical_rows=1, note='decode norm, one row padded to 32'),
]
CASES += [dict(name=f'width-{width}-rows-{m}', model='shape-coverage', shape=[m, width],
               note='standalone RMSNorm width expansion')
          for width in (32, 1024, 2304, 4096) for m in (32, 128)]


def reference(h, gamma, *, eps=1e-5):
    h = torch.from_numpy(h)
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * torch.from_numpy(gamma)).numpy()


def build(spec, rng):
    m, width = spec['shape']
    rows = spec.get('logical_rows', m)
    eps = spec.get('eps', 1e-5)
    h_words = np.zeros((m, width), dtype=np.uint16)
    h_words[:rows] = bench.bf16_words(rng.standard_normal((rows, width), dtype=np.float32))
    gamma_words = bench.bf16_words(1 + 0.1 * rng.standard_normal(width, dtype=np.float32))
    expected = reference(bench.bf16_values(h_words), bench.bf16_values(gamma_words), eps=eps)
    scale = bench.fp8_scale(expected)
    masks = np.zeros((32, 32, 32), dtype='<u2')
    for i in range(32):
        masks[i, :, i] = 0x3f80
    h = bench.pack_bf16_tiles(h_words)
    output_offset, output_bytes = 2 * m * width, m * width
    mask_offset = output_offset + output_bytes + 256
    gamma_offset = mask_offset + MASK_BYTES
    arena = (h + b'\xcd' * output_bytes + b'\x5a' * 256 + b''.join(bench.pack_bf16_tiles(tile) for tile in masks)
             + bench.pack_bf16_tiles(np.broadcast_to(gamma_words, (32, width)).copy()) + b'\xa5' * 256)
    source = bench.relocate(standalone_source(m, width, mask_offset, gamma_offset, scale, eps=eps))
    return bench.Case(source, arena, output_offset, output_bytes, expected[:m],
                      lambda raw: bench.fp8_values(bench.unpack_fp8_tiles(raw, m, width), scale),
                      meta=dict(output_scale=scale, eps=eps))
