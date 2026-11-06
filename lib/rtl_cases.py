"""Deterministic, reduced physical programs for per-family RTL comparisons.

``build_cases()`` yields ``(family, spec, Case)``. Addresses in every Case are
relative to ``kernel_bench.DRAM_BASE``. Reduction metadata records the real
model geometry without implying that a reduced run validates that geometry.
"""
from __future__ import annotations

import numpy as np

import kernel_bench as bench

FAMILIES = ('projection_gemm', 'attention', 'rmsnorm', 'rope', 'epilogue',
            'rne_add', 'smol_fusion', 'resnet_pool', 'resnet_patch_gather')
SEED = 7


def _pool(rng):
    from atlas_resnet_pool import MaxPool

    # Two output rows avoid the generator's mandatory subsequent-row loop
    # starting at zero. One complete output tile exercises padded boundary taps.
    channels, height, width = 32, 4, 32
    codes = np.concatenate((np.array([0], dtype=np.uint8), np.arange(8, 127, dtype=np.uint8)))
    image = rng.choice(codes, (height, width, channels))
    source = image.tobytes()
    pool = MaxPool(0, len(source), len(source) + 4096 + 256, channels, height, width)
    arena = source + pool.constants() + b'\xa5' * 256 + b'\xcd' * pool.output_bytes + b'\x5a' * 256
    pool.validate(len(arena))
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)))
    golden = np.zeros((height // 2, width // 2, channels), dtype=np.uint8)
    for ky in range(3):
        for kx in range(3):
            golden = np.maximum(golden, padded[ky:ky + height:2, kx:kx + width:2])
    decode = lambda raw: bench.fp8_values(np.frombuffer(raw, dtype=np.uint8))
    case = bench.Case(pool.assembly(bench.DRAM_BASE), arena, pool.output_offset, pool.output_bytes,
                      decode(golden.tobytes()), decode, golden.tobytes(), 'nonnegative FP8 byte maximum',
                      dict(requested_dma=dict(load_bytes=pool.dma_load_bytes, store_bytes=pool.dma_store_bytes)))
    return [channels, height, width], case


def _patch_gather(rng):
    from atlas_resnet_input import DirectInput

    # Existing gather_block execution-test geometry: neighborhood boundaries,
    # padded output rows, and a VMEM bank jump, without the following GEMM.
    channels, height, width, row_tiles = 32, 6, 6, 2
    rows, k = 32 * row_tiles, 9 * channels
    source = rng.integers(1, 256, rows * channels, dtype=np.uint8).tobytes()
    direct = DirectInput(bench.DRAM_BASE + 64, channels, height, width,
                         zero_offset=bench.DRAM_BASE, kernel=3, padding=1)
    direct.validate(rows, k)
    output, extent = 64 + len(source), rows * k
    lines = direct.gather_block(0, row_tiles, 1024, 8192, 'probe')
    lines += ['li x1, 1024', f'li x2, {bench.DRAM_BASE + output}', 'li x3, 1024',
              f'li x4, {extent // 1024}', 'store:', 'dma.store x2, x1, x3, ch0', 'dma.wait ch0',
              'addi x1, x1, 256', 'add x2, x2, x3', 'li x5, 2048', 'bne x1, x5, same',
              'nop', 'li x1, 8192', 'same:', 'addi x4, x4, -1', 'bne x4, x0, store', 'nop', 'ecall']
    arena = bytes(64) + source + b'\xcd' * extent + b'\x5a' * 256
    image = np.frombuffer(source, dtype=np.uint8)[:height * width * channels].reshape(height, width, channels)
    matrix = np.zeros((rows, k), dtype=np.uint8)
    for row in range(height * width):
        y, x = divmod(row, width)
        for tap in range(9):
            iy, ix = y + tap // 3 - 1, x + tap % 3 - 1
            if 0 <= iy < height and 0 <= ix < width:
                matrix[row, tap * channels:(tap + 1) * channels] = image[iy, ix]
    golden = bench.pack_fp8_tiles(matrix)
    # This is an exact byte movement test, including arbitrary FP8 bit patterns.
    decode = lambda raw: np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    case = bench.Case('\n'.join(lines) + '\n', arena, output, extent, decode(golden), decode,
                      golden, 'host im2col in (ky,kx,channel) order',
                      dict(requested_dma=dict(load_bytes=extent, store_bytes=extent),
                           scope='3x3 gather_block only; excludes following projection GEMM'))
    return [channels, height, width], case


def load_case(family, reduced=True):
    """Return ``(spec, Case)``; spec contains shape label and reduction reason.

    ``reduced=False`` selects the first published family reference case. ResNet
    adapters expose reduced cases only, since they have no published bench API.
    """
    if family not in FAMILIES:
        raise ValueError(f'unknown RTL case family: {family}')
    rng = np.random.default_rng(SEED)
    actually_reduced = reduced
    if family in ('resnet_pool', 'resnet_patch_gather'):
        if not reduced:
            raise ValueError('ResNet adapters expose reduced execution-test cases only')
        shape, case = (_pool if family == 'resnet_pool' else _patch_gather)(rng)
        original = [64, 112, 112] if family == 'resnet_pool' else [64, 56, 56]
        reason = ('one full output tile with two output rows and boundary padding' if family == 'resnet_pool'
                  else 'existing 3x3 gather execution-test geometry, padded rows, and VMEM bank jump; excludes GEMM')
        spec = dict(name=f'rtl-{family}', model='ResNet50', shape=shape)
    else:
        module = bench.load_reference(family)
        spec = dict(module.CASES[0])
        original = list(spec['shape'])
        if reduced:
            if family == 'projection_gemm':
                spec.update(shape=[32, 32, 960], logical_rows=1, model='SV',
                            name='sv-state-proj', note='SmolVLA state_proj, logical 1x32x960')
                original = [32, 32, 960]
                actually_reduced = False
                reason = ('exact coverage/examples/smolvla/MODEL-BENCH.md state_proj shape; '
                          'physical 32x32x960, logical 1x32x960')
            elif family == 'attention':
                spec.update(shape=[1, 129, 32, 4], mask='causal', output_fp8=True,
                            query_start=128, name='tl-decode-causal',
                            note='TinyLlama packed decode attention, query position 128')
                original = [1, 129, 32, 4]
                actually_reduced = False
                reason = ('exact coverage/examples/tinyllama/MODEL-BENCH.md packed decode attention shape; '
                          'one query, 129 keys, 32 query heads, four KV heads, head64, query position 128')
            elif family == 'rmsnorm':
                spec.update(shape=[32, 2048], logical_rows=1, name='tl-decode-norm',
                            note='TinyLlama decode norm, one logical row padded to 32')
                original = [32, 2048]
                actually_reduced = False
                reason = ('exact coverage/examples/tinyllama/MODEL-BENCH.md decode norm shape; '
                          'physical 32x2048, one logical row')
            elif family == 'rope':
                spec.update(shape=[32, 2048, 256], logical_rows=1, name='tl-decode-k-proj',
                            note='TinyLlama decode K projection and RoPE, one logical row padded to 32')
                original = [32, 2048, 256]
                actually_reduced = False
                reason = ('exact coverage/examples/tinyllama/MODEL-BENCH.md decode K projection shape; '
                          'physical 32x2048x256, logical 1x2048x256; synthetic RoPE table positions 0..31')
            else:
                spec.update(shape=[32, 32, 960])
                reason = 'smallest qualified traffic-reuse contraction; retains the family fused operation'
        else:
            reason = 'published real-model family benchmark shape'
        case = module.build(spec, rng)
        if family == 'rope' and spec.get('logical_rows') == 1:
            case.meta.update(golden_logical_rows=1,
                             golden_padding_note='Independent golden comparison covers logical row 0; '
                             'padded rows can differ only by BF16 zero signs, as documented in '
                             'coverage/examples/tinyllama/MODEL-BENCH.md. RTL/model comparison retains all bytes.')
        if actually_reduced:
            spec.update(original_case_name=spec['name'], original_note=spec.get('note', ''),
                        name=f'rtl-{family}', note=reason)
    label = 'x'.join(map(str, spec['shape']))
    spec.update(seed=SEED, original_model_shape=original, shape_label=label,
                reduced=actually_reduced, reduction=reason,
                reduction_label='reduced' if actually_reduced else 'published', reduction_reason=reason)
    case.meta.update(seed=SEED, original_model_shape=original, shape_label=label,
                     reduced=actually_reduced, reduction=reason,
                     reduction_label=spec['reduction_label'], reduction_reason=reason)
    return spec, case


def build_cases(families=None, reduced=True):
    """Yield all nine families, or the explicitly selected family names."""
    for family in FAMILIES if families is None else families:
        spec, case = load_case(family, reduced=reduced)
        yield family, spec, case
