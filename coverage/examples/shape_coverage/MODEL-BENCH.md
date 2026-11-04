# shape_sweep kernel bench at model shapes

Regenerate: `python coverage/bench_shapes.py --model-root NPU_MODEL --atlas-mlir-bin BIN --out OUT`

Input: `shapes/coverage.json` (SHA-256 `c0d292860d0198282f6df1f8c68e9ac49b7e10e5b88b271fee09e0e7cb4e1dfe`), coverage status COMPLETE. The coverage derives 28 distinct launches; 28 executed on npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` with numerics="rtl" on seeded synthetic inputs at the model's shape (4 worker processes; 739 job-seconds in all, longest job 145 s).

Model cycles are npu_model Core cycle-model counts, not RTL cycles. Cycles x count multiplies one launch by the number of launches in the model; the sums below are device-only estimates that ignore host time, launch overhead, and data movement between launches. Bit-exact compares output bytes with the independent lib/golden reference where the family has one; cosine and rel-err compare the decoded output with an FP32 PyTorch reference on the unquantized inputs, so they include FP8 input quantization. Inputs are synthetic, so fidelity and data-dependent cycles are representative, not the model's.

A distinct launch groups the same shape and variant; a distinct program is the address-specific program text, as in the separate 1,589-program kit check.

Shares below are by launch kind: `epilogue`, `rne_add`, `rope`, and `smol_fusion` include their GEMM projection and fused stage, while `projection_gemm` covers unfused panels. They are not shares of non-GEMM time; the generator metadata does not provide separately measured cycles for those stages.

Compiler backend: machine IR, atlas-opt stream and timing verifiers, and atlas-emit, with minimum tier `stream-verified`. Successful jobs require emitted words to equal npu_model's assembler; per-job JSON records binary SHA-256 digests, diagnostics, and word counts.

| device launches | derived here | qualified kit |
| --- | ---: | ---: |
| attention | 16 | not declared |
| rmsnorm | 12 | not declared |

## Summary

Device cycle estimate (sum of model cycles x count over 28 of 28 launches): 12,052,277 npu_model model cycles. It ignores host time and launch overhead.

Compiler tier counts by distinct launch: 16 `stream-verified`, 12 `verified`.

| launch kind | launches | launches with cycles | model cycles x count | share |
| --- | ---: | ---: | ---: | ---: |
| `rmsnorm` | 12 | 12 | 10,138,881 | 84.12% |
| `attention` | 16 | 16 | 1,913,396 | 15.88% |

Top shapes by total cycles:

| family | shape | count | model cycles | cycles x count |
| --- | --- | ---: | ---: | ---: |
| `rmsnorm` | 128x7136 | 1 | 3,348,763 | 3,348,763 |
| `rmsnorm` | 128x4096 | 1 | 1,943,686 | 1,943,686 |
| `rmsnorm` | 32x7136 | 1 | 1,128,190 | 1,128,190 |
| `rmsnorm` | 128x2304 | 1 | 1,116,018 | 1,116,018 |
| `rmsnorm` | 32x4096 | 1 | 668,236 | 668,236 |
| `rmsnorm` | 128x1024 | 1 | 523,906 | 523,906 |
| `rmsnorm` | 128x960 | 1 | 494,073 | 494,073 |
| `rmsnorm` | 32x2304 | 1 | 397,521 | 397,521 |
| `rmsnorm` | 32x1024 | 1 | 203,923 | 203,923 |
| `rmsnorm` | 32x960 | 1 | 194,193 | 194,193 |

Unsupported or failed launches: 0.

## Launches

| family | shape | export dtype | variant | tier | words | count | model cycles | cycles x count | golden | bit-exact | timed equals functional | cosine | rel-err | modules |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | :-: | :-: | ---: | ---: | --- |
| `attention` | 32x64 2/1 d128 causal | bfloat16 | attention wide d128, 1 unique mask tiles | stream-verified | 667 | 1 | 72,023 | 72,023 | none | n/a | exact | 0.99860 | 5.300% | `d128_causal_False_32` |
| `attention` | 32x64 2/1 d128 causal FP8 out | bfloat16 | attention wide d128, 1 unique mask tiles | stream-verified | 716 | 1 | 69,951 | 69,951 | none | n/a | exact | 0.99852 | 5.437% | `d128_causal_True_32` |
| `attention` | 32x64 2/1 d128 none | bfloat16 | attention wide d128, 0 unique mask tiles | stream-verified | 659 | 1 | 69,925 | 69,925 | none | n/a | exact | 0.99876 | 5.000% | `d128_none_False_32` |
| `attention` | 32x64 2/1 d128 none FP8 out | bfloat16 | attention wide d128, 0 unique mask tiles | stream-verified | 708 | 1 | 67,853 | 67,853 | none | n/a | exact | 0.99849 | 5.503% | `d128_none_True_32` |
| `attention` | 1x65 2/1 d128 causal | bfloat16 | attention wide d128, 2 unique mask tiles | stream-verified | 666 | 1 | 95,133 | 95,133 | none | n/a | exact | 0.99818 | 6.334% | `d128_causal_False_1` |
| `attention` | 1x65 2/1 d128 causal FP8 out | bfloat16 | attention wide d128, 2 unique mask tiles | stream-verified | 715 | 1 | 93,061 | 93,061 | none | n/a | exact | 0.99877 | 5.168% | `d128_causal_True_1` |
| `attention` | 1x65 2/1 d128 none | bfloat16 | attention wide d128, 2 unique mask tiles | stream-verified | 666 | 1 | 95,133 | 95,133 | none | n/a | exact | 0.99871 | 5.093% | `d128_none_False_1` |
| `attention` | 1x65 2/1 d128 none FP8 out | bfloat16 | attention wide d128, 2 unique mask tiles | stream-verified | 715 | 1 | 93,061 | 93,061 | none | n/a | exact | 0.99774 | 6.785% | `d128_none_True_1` |
| `attention` | 32x64 2/1 d256 causal | bfloat16 | attention wide d256, 1 unique mask tiles | stream-verified | 1119 | 1 | 138,170 | 138,170 | none | n/a | exact | 0.99880 | 4.907% | `d256_causal_False_32` |
| `attention` | 32x64 2/1 d256 causal FP8 out | bfloat16 | attention wide d256, 1 unique mask tiles | stream-verified | 1215 | 1 | 134,022 | 134,022 | none | n/a | exact | 0.99848 | 5.544% | `d256_causal_True_32` |
| `attention` | 32x64 2/1 d256 none | bfloat16 | attention wide d256, 0 unique mask tiles | stream-verified | 1111 | 1 | 136,072 | 136,072 | none | n/a | exact | 0.99874 | 5.013% | `d256_none_False_32` |
| `attention` | 32x64 2/1 d256 none FP8 out | bfloat16 | attention wide d256, 0 unique mask tiles | stream-verified | 1207 | 1 | 131,924 | 131,924 | none | n/a | exact | 0.99848 | 5.529% | `d256_none_True_32` |
| `attention` | 1x65 2/1 d256 causal | bfloat16 | attention wide d256, 2 unique mask tiles | stream-verified | 1118 | 1 | 181,341 | 181,341 | none | n/a | exact | 0.99883 | 5.005% | `d256_causal_False_1` |
| `attention` | 1x65 2/1 d256 causal FP8 out | bfloat16 | attention wide d256, 2 unique mask tiles | stream-verified | 1214 | 1 | 177,193 | 177,193 | none | n/a | exact | 0.99851 | 5.503% | `d256_causal_True_1` |
| `attention` | 1x65 2/1 d256 none | bfloat16 | attention wide d256, 2 unique mask tiles | stream-verified | 1118 | 1 | 181,341 | 181,341 | none | n/a | exact | 0.99859 | 5.465% | `d256_none_False_1` |
| `attention` | 1x65 2/1 d256 none FP8 out | bfloat16 | attention wide d256, 2 unique mask tiles | stream-verified | 1214 | 1 | 177,193 | 177,193 | none | n/a | exact | 0.99828 | 6.110% | `d256_none_True_1` |
| `rmsnorm` | 128x1024 | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 156 | 1 | 523,906 | 523,906 | none | n/a | exact | 0.99963 | 2.723% | `norm_128_1024` |
| `rmsnorm` | 128x2304 | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 218 | 1 | 1,116,018 | 1,116,018 | none | n/a | exact | 0.99964 | 2.710% | `norm_128_2304` |
| `rmsnorm` | 128x32 | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 145 | 1 | 66,186 | 66,186 | none | n/a | exact | 0.99963 | 2.738% | `norm_128_32` |
| `rmsnorm` | 128x4096 | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 249 | 1 | 1,943,686 | 1,943,686 | none | n/a | exact | 0.99964 | 2.722% | `norm_128_4096` |
| `rmsnorm` | 128x7136 | bfloat16 | streamed tile-row RMSNorm, FP8 out | verified | 355 | 1 | 3,348,763 | 3,348,763 | none | n/a | exact | 0.99964 | 2.746% | `norm_128_7136` |
| `rmsnorm` | 128x960 | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 154 | 1 | 494,073 | 494,073 | none | n/a | exact | 0.99964 | 2.693% | `norm_128_960` |
| `rmsnorm` | 32x1024 | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 156 | 1 | 203,923 | 203,923 | none | n/a | exact | 0.99964 | 2.693% | `norm_32_1024` |
| `rmsnorm` | 32x2304 | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 218 | 1 | 397,521 | 397,521 | none | n/a | exact | 0.99963 | 2.738% | `norm_32_2304` |
| `rmsnorm` | 32x32 | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 148 | 1 | 54,186 | 54,186 | none | n/a | exact | 0.99966 | 2.613% | `norm_32_32` |
| `rmsnorm` | 32x4096 | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 249 | 1 | 668,236 | 668,236 | none | n/a | exact | 0.99964 | 2.720% | `norm_32_4096` |
| `rmsnorm` | 32x7136 | bfloat16 | streamed tile-row RMSNorm, FP8 out | verified | 358 | 1 | 1,128,190 | 1,128,190 | none | n/a | exact | 0.99964 | 2.751% | `norm_32_7136` |
| `rmsnorm` | 32x960 | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 154 | 1 | 194,193 | 194,193 | none | n/a | exact | 0.99964 | 2.673% | `norm_32_960` |

`words` is the executed program size in 32-bit instruction words (after the arena relocation the bench applies). Bit-exact covers the logical rows and columns the model consumes; a padding note counts output words in the zero-padded rows or columns that differ from the golden (`+0/-0`: only the sign of zero differs).

## Unmapped ops (0 rows, 0 instances, 0 FLOPs)

None: every exported op has a placement.

## Notes

New-shape acceptance: cosine >= 0.98, relative L2 error <= 0.20; both families lack an independent Atlas-arithmetic golden. All cycle counts are full launches.
