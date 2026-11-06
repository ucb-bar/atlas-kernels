"""PyTorch reference and real-shape bench case for rotate-half RoPE fused into the q projection.

The post-pass rotates each 64-channel head of the BF16 projection P: out = RNE(RNE(x*cos) + RNE(r*sin)) with
r = -x[d+32] for d < 32 and x[d-32] otherwise (atlas_rope_postpass.py). Tables are TinyLlama's (theta 10000,
positions 0..M-1), packed as atlas-mlir tools/tiny_full/fusion.py rope_tables packs them. The FP32 reference
rotates A @ W on the unquantized operands; the golden applies that RNE formula to the ordered_matmul output.
"""
import numpy as np
import torch

from atlas_rope_postpass import ResidentRopeEpilogue, append_postpass
from atlas_vector_arithmetic import decode_bf16, ordered_matmul, rescale_bf16_array, round_bf16
import kernel_bench as bench

CASES = [dict(name='tl-prefill-q-proj', model='TL', shape=[128, 2048, 2048],
              note='prefill q_proj with the RoPE post-pass, positions 0..127 (PREFILL_ROPE)')]


def tables(rows):
    """BF16 cos and sin words [rows, 64] with equal halves, Llama rotary embedding (theta 10000)."""
    inverse = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    angles = torch.outer(torch.arange(rows, dtype=torch.float32), inverse)
    angles = torch.cat((angles, angles), dim=-1)
    return bench.bf16_words(angles.cos().numpy()), bench.bf16_words(angles.sin().numpy())


def rotate_half(x):
    x = x.reshape(x.shape[0], -1, 64)
    return np.concatenate((-x[..., 32:], x[..., :32]), axis=-1).reshape(x.shape[0], -1)


def reference(a, w, cosine, sine):
    q = (torch.from_numpy(a) @ torch.from_numpy(w)).numpy()
    heads = q.shape[1] // 64
    return q * np.tile(cosine, heads) + rotate_half(q) * np.tile(sine, heads)


def build(spec, rng):
    m, k, n = spec['shape']
    a, w, (a_codes, a_scale), (w_codes, w_scale) = bench.projection_operands(spec, rng)
    cos_words, sin_words = tables(m)
    blob = b''.join(bench.pack_bf16_tiles(value[:m, :32]) for value in (cos_words, sin_words, sin_words ^ 0x8000))
    output_offset = m * k + k * n + 256
    extra = output_offset + 2 * m * n + 256
    epilogue = ResidentRopeEpilogue(m, n, output_offset, scratch_registers=(10, 11), cos_offset=extra,
                                    sin_offset=extra + 64 * m, neg_sin_offset=extra + 128 * m)
    source, arena, offset, extent = bench.stage_projection(m, k, n, a_codes, w_codes, a_scale + w_scale, epilogue,
                                                           blob, postpass=append_postpass)
    x = decode_bf16(rescale_bf16_array(ordered_matmul(a_codes, w_codes), a_scale + w_scale))
    heads = n // 64
    cosine, sine = (np.tile(decode_bf16(words), heads) for words in (cos_words, sin_words))
    golden = round_bf16(decode_bf16(round_bf16(x * cosine)) + decode_bf16(round_bf16(rotate_half(x) * sine)))
    return bench.Case(source, arena, offset, extent,
                      reference(a, w, bench.bf16_values(cos_words), bench.bf16_values(sin_words)),
                      lambda raw: bench.bf16_values(bench.unpack_bf16_tiles(raw, m, n)),
                      bench.pack_bf16_tiles(golden), 'ordered_matmul + RoPE RNE formula',
                      dict(a_scale=a_scale, w_scale=w_scale))
