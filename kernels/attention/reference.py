"""PyTorch reference and real-shape bench cases for attention v2 (QK^T, scale, mask, softmax, PV).

Inputs are BF16 Q [Hq, S, D], K and V [Hkv, T, D], D in 64/128/256; the program quantizes them to FP8 with the given
power-of-two exponents, uses unnormalized FP8 exponentials at fixed scale 256, and writes BF16 (SmolVLA) or FP8
(TinyLlama) output in the o_proj input layout. There is no independent Atlas-arithmetic golden; fidelity is
against FP32 PyTorch softmax attention on the same BF16 inputs.
"""
import numpy as np
import torch

from atlas_attention_v2_program import AttentionV2Config, assembly_v2, check_program_v2, stage_v2, unpack_output_v2
from atlas_attention_wide_program import AttentionWideConfig, assembly_wide, stage_wide, unpack_output_wide
import kernel_bench as bench

CASES = [
    dict(name='tl-prefill-causal', model='TL', shape=[128, 128, 32, 4], mask='causal', output_fp8=True,
         note='prefill self-attention, 32 query and 4 KV heads, causal, FP8 output'),
    dict(name='sv-prefix-arbitrary', model='SV', shape=[177, 177, 15, 5], mask='arbitrary', output_fp8=False,
         note='VLM prefix self-attention, 15 query and 5 KV heads, prefix block mask (176 image and language tokens, then state), BF16 output'),
    dict(name='wide128-none-tail', model='shape-expansion', shape=[33, 35, 2, 1], head_dim=128,
         mask='none', output_fp8=False, note='wide-head GQA with query and key tails, BF16 output'),
    dict(name='wide128-causal-decode', model='shape-expansion', shape=[1, 65, 2, 1], head_dim=128,
         mask='causal', output_fp8=True, note='wide-head ordinary tiled GQA decode, FP8 output'),
    dict(name='wide256-causal-tail', model='shape-expansion', shape=[33, 65, 2, 1], head_dim=256,
         mask='causal', output_fp8=False, note='wide-head causal GQA with query and key tails, BF16 output'),
    dict(name='wide256-none-decode', model='shape-expansion', shape=[1, 65, 2, 1], head_dim=256,
         mask='none', output_fp8=True, note='wide-head ordinary tiled GQA decode, FP8 output'),
]


def allowed(spec):
    rows, keys = spec['shape'][:2]
    if spec['mask'] == 'causal':
        return np.tril(np.ones((rows, keys), dtype=bool), keys - rows)
    mask = np.ones((rows, keys), dtype=bool)
    if spec['mask'] == 'arbitrary':
        mask[:keys - 1, keys - 1] = False  # SmolVLA prefix: the state token starts a new attention block.
    return mask


def reference(q, k, v, mask, scale=0.125):
    q, k, v = (torch.from_numpy(x).float() for x in (q, k, v))
    group = q.shape[0] // k.shape[0]
    k, v = k.repeat_interleave(group, 0), v.repeat_interleave(group, 0)
    scores = (q @ k.transpose(1, 2)) * scale
    scores = scores.masked_fill(~torch.from_numpy(mask), float('-inf'))
    return (torch.softmax(scores, -1) @ v).numpy()


def mask_tiles(mask):
    rows, keys = mask.shape
    padded = np.zeros(((rows + 31) // 32 * 32, (keys + 31) // 32 * 32), dtype=bool)
    padded[:rows, :keys] = mask
    return [[padded[r:r + 32, c:c + 32] for c in range(0, padded.shape[1], 32)] for r in range(0, padded.shape[0], 32)]


def build(spec, rng):
    rows, keys, heads, kv_heads = spec['shape']
    dim = spec.get('head_dim', 64)
    words = [bench.bf16_words(rng.standard_normal(shape, dtype=np.float32))
             for shape in ((heads, rows, dim), (kv_heads, keys, dim), (kv_heads, keys, dim))]
    q, k, v = (bench.bf16_values(x) for x in words)
    mask = allowed(spec)
    expected = reference(q, k, v, mask, scale=dim ** -0.5)
    exponents = dict(q_exponent=bench.fp8_scale(q), k_exponent=bench.fp8_scale(k), v_exponent=bench.fp8_scale(v))
    o_exponent = bench.fp8_scale(expected) if spec['output_fp8'] else 0
    config_type, stage_attention, assembly_attention, unpack_attention = (
        (AttentionV2Config, stage_v2, assembly_v2, unpack_output_v2) if dim == 64 else
        (AttentionWideConfig, stage_wide, assembly_wide, unpack_output_wide))
    extra = {} if dim == 64 else {"head_dim": dim}
    config = config_type(rows, keys, heads, kv_heads, mask=spec['mask'], query_start=keys - rows if spec['mask'] == 'causal' else None,
                               output_fp8=spec['output_fp8'], o_exponent=o_exponent, **exponents, **extra)
    arena, offsets = stage_attention(config, *words, mask_tiles=mask_tiles(mask) if spec['mask'] == 'arbitrary' else None)
    source = check_program_v2(assembly_attention(config, offsets))

    def decode(raw):
        value = unpack_attention(config, raw)
        return bench.fp8_values(value, o_exponent) if spec['output_fp8'] else bench.bf16_values(value)
    return bench.Case(source, arena, offsets['output'], offsets['output_bytes'], expected, decode,
                      meta=dict(**exponents, o_exponent=o_exponent, head_dim=dim))
