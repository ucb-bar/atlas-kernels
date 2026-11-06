"""Independent finite FC1 -> bias -> GELU(tanh) -> FP8 arithmetic.

No execution-model imports or model lookup tables are used. MXU0 uses the
independently qualified ordered MAC oracle. VPU add truncates an FP32 sum,
multiply rounds directly to BF16, and square truncates its normalized integer
significand. Tanh reconstructs the RTL's 32 hardwired Q1.16 knots and Q3.16
interpolation, rather than using an exhaustive execution-model unary table.

Admitted inputs are finite, with finite intermediate BF16 multiply results.
The FP8 store clips to the consumer scale before nearest-even packing.
"""

from functools import lru_cache
from fractions import Fraction
import math

import numpy as np

from atlas_tile_arithmetic import bf16_round, pack_bf16
from atlas_vector_arithmetic import (decode_bf16, ordered_matmul,
                                     rescale_bf16_array, round_bf16)

_TANH_KNOTS = tuple(math.floor(math.tanh(i / 8) * 65536 + .5) for i in range(32))


def add_truncated(a, b):
    """BF16 inputs, FP32 nearest-even sum, upper sixteen result bits."""
    value = (decode_bf16(a).astype(np.float32) + decode_bf16(b).astype(np.float32))
    if not np.all(np.isfinite(value)):
        raise ValueError('nonfinite add outside admitted arithmetic')
    return (value.view(np.uint32) >> 16).astype(np.uint16)


def multiply(a, b):
    exact = decode_bf16(a) * decode_bf16(b)
    if np.any(np.abs(exact) > float(255 * 2**120)):
        raise ValueError('multiply overflows admitted finite BF16')
    result = round_bf16(exact)
    # IEEE multiplication preserves the XOR sign for an exact zero product.
    return np.where(exact == 0, (np.asarray(a) ^ np.asarray(b)) & 0x8000, result).astype(np.uint16)


@lru_cache(maxsize=65536)
def square(bits):
    exponent, fraction = (bits >> 7) & 255, bits & 127
    if exponent == 255:
        raise ValueError('nonfinite square input')
    if not exponent:
        return 0
    significand = (128 + fraction)**2
    lead = significand.bit_length() - 1
    output_exp = 2 * (exponent - 127) + (lead - 14) + 127
    if output_exp <= 0:
        return 0
    if output_exp >= 255:
        raise ValueError('square overflows admitted finite BF16')
    return (output_exp << 7) | ((significand >> (lead - 7)) & 127)


@lru_cache(maxsize=65536)
def tanh(bits):
    exponent, fraction = (bits >> 7) & 255, bits & 127
    sign = bits & 0x8000
    if exponent == 255:
        raise ValueError('nonfinite tanh outside admitted arithmetic')
    if exponent >= 129:
        return sign | 0x3f80
    true_exp = exponent - 127 if exponent else -126
    significand = (128 if exponent else 0) + fraction
    shift = true_exp + 9
    fixed = ((significand << shift) if shift >= 0 else (significand >> -shift)) & ((1 << 19) - 1)
    address, alpha = (fixed >> 13) & 31, fixed & 8191
    y0, y1 = _TANH_KNOTS[address], _TANH_KNOTS[min(31, address + 1)]
    magnitude = y0 + (((y1 - y0) * alpha) >> 13)
    return sign | bf16_round(Fraction(magnitude, 65536))


def _lookup(function, words):
    distinct, inverse = np.unique(words, return_inverse=True)
    values = np.array([function(int(word)) for word in distinct], dtype=np.uint16)
    return values[inverse].reshape(words.shape)


def gelu(words):
    """The emitted sequence's explicit BF16 rounding/truncation boundaries."""
    cubic = multiply(_lookup(square, words), words)
    cubic = multiply(cubic, np.array(15671, dtype=np.uint16))
    argument = multiply(add_truncated(words, cubic), np.array(16204, dtype=np.uint16))
    gate = add_truncated(_lookup(tanh, argument), np.array(16256, dtype=np.uint16))
    gate = multiply(gate, np.array(16128, dtype=np.uint16))
    return multiply(words, gate)


def golden_arena(initial, fusion):
    """Return independently computed full arena and finite/saturation evidence."""
    m, k, n = fusion['physical_mkn']
    if any(type(v) is not int or v <= 0 or v % 32 for v in (m, k, n)):
        raise ValueError('shape must contain positive multiples of 32')
    offset, size = fusion['output_offset'], m*n
    if fusion['output_dtype'] != 'fp8' or fusion['operations'] != ['bias', 'gelu_tanh']:
        raise ValueError('only FC1 bias/GELU FP8 is admitted')
    if offset != m*k + k*n + 256 or len(initial) != fusion['total_dram_bytes']:
        raise ValueError('arena geometry differs')
    a = np.frombuffer(initial[:m*k], dtype=np.uint8).reshape(m//32, k//32, 32, 32).transpose(0, 2, 1, 3).reshape(m, k)
    b = np.frombuffer(initial[m*k:m*k+k*n], dtype=np.uint8).reshape(n//32, k//32, 32, 32).transpose(1, 2, 0, 3).reshape(k, n)
    extra = fusion['extra_offset']
    if extra != offset+size+256 or len(initial) != extra+64*n+256:
        raise ValueError('bias arena geometry differs')
    if (initial[offset-256:offset] != b'\xa5'*256
            or initial[offset:offset+size] != b'\xcd'*size
            or initial[offset+size:extra] != b'Z'*256
            or initial[-256:] != b'Z'*256):
        raise ValueError('arena poison or guards differ')
    bias_tiles = np.frombuffer(initial[extra:extra+64*n], dtype='<u2').reshape(n//32, 2, 32, 16)
    bias = bias_tiles[:, :, 0, :].reshape(n)
    if not np.all(bias_tiles == bias_tiles[:, :, :1, :]):
        raise ValueError('channel bias tile rows differ')
    product = rescale_bf16_array(ordered_matmul(a, b), fusion['output_rescale_exponent'])
    output = gelu(add_truncated(product, bias))
    scale = fusion['output_scale_exponent']
    if type(scale) is not int or not -126 <= scale <= 119:
        raise ValueError('consumer scale outside admitted finite clipping domain')
    decoded = decode_bf16(output)
    limit = math.ldexp(448., scale)
    saturated = np.abs(decoded) > limit
    clipped = np.clip(decoded, -limit, limit)
    words = round_bf16(clipped)
    unique, inverse = np.unique(words, return_inverse=True)
    packed = np.array([pack_bf16(int(word), 127+scale) for word in unique], dtype=np.uint8)[inverse].reshape(m, n)
    raw = packed.reshape(m//32, 32, n//32, 32).transpose(0, 2, 1, 3).copy().tobytes()
    result = initial[:offset] + raw + initial[offset+size:]
    return result, dict(output_bytes=size, saturation_values=int(saturated.sum()), nonfinite_values=0,
                        preserved_bytes=len(initial)-size, arithmetic='independent ordered BF16 MAC, truncated add/square, RNE multiply, fixed-point tanh, clipped FP8')
