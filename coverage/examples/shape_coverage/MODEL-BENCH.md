# shape_sweep kernel bench at synthetic shapes

Regenerate: `python coverage/bench_shapes.py --model-root NPU_MODEL --atlas-mlir-bin BIN --out OUT`

Input: `../out/shapes-run2/coverage.json` (SHA-256 `8c70fa61a41cd9d2895e2b826a7ffebfa57b494b12ca8dce6aa0319460c76f5b`), coverage status COMPLETE. The coverage derives 30 distinct launches; 30 executed on npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` with numerics="rtl" on seeded synthetic inputs at synthetic shapes (12 worker processes; 856 job-seconds in all, longest job 145 s).

Model cycles are npu_model Core cycle-model counts, not RTL cycles. Cycles x count multiplies one launch by the number of launches in the model; the sums below are device-only estimates that ignore host time, launch overhead, and data movement between launches. Bit-exact compares output bytes with the independent lib/golden reference where the family has one; cosine and rel-err compare the decoded output with an FP32 PyTorch reference on the unquantized inputs, so they include FP8 input quantization. Inputs are synthetic, so fidelity and data-dependent cycles are representative, not the model's.

A distinct launch groups the same shape and variant; a distinct program is the address-specific program text, as in the separate 1,589-program kit check.

Shares below are by launch kind: `epilogue`, `rne_add`, `rope`, and `smol_fusion` include their GEMM projection and fused stage, while `projection_gemm` covers unfused panels. They are not shares of non-GEMM time; the generator metadata does not provide separately measured cycles for those stages.

Compiler backend: machine IR, atlas-opt stream and timing verifiers, and atlas-emit, with minimum tier `stream-verified`. Successful jobs require emitted words to equal npu_model's assembler; per-job JSON records binary SHA-256 digests, diagnostics, and word counts.

| device launches | derived here | qualified kit |
| --- | ---: | ---: |
| attention | 18 | not declared |
| rmsnorm | 12 | not declared |

## Summary

Device cycle estimate (sum of model cycles x count over 30 of 30 launches): 14,801,976 npu_model model cycles. It ignores host time and launch overhead.

Compiler tier counts by distinct launch: 18 `stream-verified`, 12 `verified`.

| launch kind | launches | launches with cycles | model cycles x count | share |
| --- | ---: | ---: | ---: | ---: |
| `rmsnorm` | 12 | 12 | 10,138,881 | 68.50% |
| `attention` | 18 | 18 | 4,663,095 | 31.50% |

Unsupported or failed launches: 0.

## Launches

| family | shape | export dtype | variant | tier | words | count | model cycles | cycles x count | golden | bit-exact | timed equals functional | cosine | rel-err | modules |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | :-: | :-: | ---: | ---: | --- |
| `rmsnorm` | 128x7136 | bfloat16 | streamed tile-row RMSNorm, FP8 out | verified | 355 | 1 | 3,348,763 | 3,348,763 | none | n/a | exact | 0.99964 | 2.746% | `norm_128_7136` |
| `rmsnorm` | 128x4096 | bfloat16 | streamed tile-row RMSNorm, FP8 out | verified | 249 | 1 | 1,943,686 | 1,943,686 | none | n/a | exact | 0.99964 | 2.722% | `norm_128_4096` |
| `attention` | 1x2048 2/1 d128 causal | bfloat16 | attention wide d128, 1 unique mask tiles | stream-verified | 666 | 1 | 1,377,667 | 1,377,667 | none | n/a | exact | 0.99843 | 5.745% | `d128_decode_capacity` |
| `attention` | 1x1024 2/1 d256 causal | bfloat16 | attention wide d256, 1 unique mask tiles | stream-verified | 1118 | 1 | 1,372,048 | 1,372,048 | none | n/a | exact | 0.99888 | 4.744% | `d256_decode_capacity` |
| `rmsnorm` | 32x7136 | bfloat16 | streamed tile-row RMSNorm, FP8 out | verified | 358 | 1 | 1,128,190 | 1,128,190 | none | n/a | exact | 0.99964 | 2.751% | `norm_32_7136` |
| `rmsnorm` | 128x2304 | bfloat16 | streamed tile-row RMSNorm, FP8 out | verified | 218 | 1 | 1,116,018 | 1,116,018 | none | n/a | exact | 0.99964 | 2.710% | `norm_128_2304` |
| `rmsnorm` | 32x4096 | bfloat16 | streamed tile-row RMSNorm, FP8 out | verified | 249 | 1 | 668,236 | 668,236 | none | n/a | exact | 0.99964 | 2.720% | `norm_32_4096` |
| `rmsnorm` | 128x1024 | bfloat16 | streamed tile-row RMSNorm, FP8 out | verified | 156 | 1 | 523,906 | 523,906 | none | n/a | exact | 0.99963 | 2.723% | `norm_128_1024` |
| `rmsnorm` | 128x960 | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 154 | 1 | 494,073 | 494,073 | none | n/a | exact | 0.99964 | 2.693% | `norm_128_960` |
| `rmsnorm` | 32x2304 | bfloat16 | streamed tile-row RMSNorm, FP8 out | verified | 218 | 1 | 397,521 | 397,521 | none | n/a | exact | 0.99963 | 2.738% | `norm_32_2304` |
| `rmsnorm` | 32x1024 | bfloat16 | streamed tile-row RMSNorm, FP8 out | verified | 156 | 1 | 203,923 | 203,923 | none | n/a | exact | 0.99964 | 2.693% | `norm_32_1024` |
| `rmsnorm` | 32x960 | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 154 | 1 | 194,193 | 194,193 | none | n/a | exact | 0.99964 | 2.673% | `norm_32_960` |
| `attention` | 1x65 2/1 d256 causal | bfloat16 | attention wide d256, 2 unique mask tiles | stream-verified | 1117 | 1 | 181,340 | 181,340 | none | n/a | exact | 0.99883 | 5.005% | `d256_causal_False_1` |
| `attention` | 1x65 2/1 d256 none | bfloat16 | attention wide d256, 2 unique mask tiles | stream-verified | 1117 | 1 | 181,340 | 181,340 | none | n/a | exact | 0.99859 | 5.465% | `d256_none_False_1` |
| `attention` | 1x65 2/1 d256 causal FP8 out | bfloat16 | attention wide d256, 2 unique mask tiles | stream-verified | 1213 | 1 | 177,192 | 177,192 | none | n/a | exact | 0.99851 | 5.503% | `d256_causal_True_1` |
| `attention` | 1x65 2/1 d256 none FP8 out | bfloat16 | attention wide d256, 2 unique mask tiles | stream-verified | 1213 | 1 | 177,192 | 177,192 | none | n/a | exact | 0.99828 | 6.110% | `d256_none_True_1` |
| `attention` | 32x64 2/1 d256 causal | bfloat16 | attention wide d256, 1 unique mask tiles | stream-verified | 1118 | 1 | 138,169 | 138,169 | none | n/a | exact | 0.99880 | 4.907% | `d256_causal_False_32` |
| `attention` | 32x64 2/1 d256 none | bfloat16 | attention wide d256, 0 unique mask tiles | stream-verified | 1110 | 1 | 136,071 | 136,071 | none | n/a | exact | 0.99874 | 5.013% | `d256_none_False_32` |
| `attention` | 32x64 2/1 d256 causal FP8 out | bfloat16 | attention wide d256, 1 unique mask tiles | stream-verified | 1214 | 1 | 134,021 | 134,021 | none | n/a | exact | 0.99848 | 5.544% | `d256_causal_True_32` |
| `attention` | 32x64 2/1 d256 none FP8 out | bfloat16 | attention wide d256, 0 unique mask tiles | stream-verified | 1206 | 1 | 131,923 | 131,923 | none | n/a | exact | 0.99848 | 5.529% | `d256_none_True_32` |
| `attention` | 1x65 2/1 d128 causal | bfloat16 | attention wide d128, 2 unique mask tiles | stream-verified | 665 | 1 | 95,132 | 95,132 | none | n/a | exact | 0.99818 | 6.334% | `d128_causal_False_1` |
| `attention` | 1x65 2/1 d128 none | bfloat16 | attention wide d128, 2 unique mask tiles | stream-verified | 665 | 1 | 95,132 | 95,132 | none | n/a | exact | 0.99871 | 5.093% | `d128_none_False_1` |
| `attention` | 1x65 2/1 d128 causal FP8 out | bfloat16 | attention wide d128, 2 unique mask tiles | stream-verified | 714 | 1 | 93,060 | 93,060 | none | n/a | exact | 0.99877 | 5.168% | `d128_causal_True_1` |
| `attention` | 1x65 2/1 d128 none FP8 out | bfloat16 | attention wide d128, 2 unique mask tiles | stream-verified | 714 | 1 | 93,060 | 93,060 | none | n/a | exact | 0.99774 | 6.785% | `d128_none_True_1` |
| `attention` | 32x64 2/1 d128 causal | bfloat16 | attention wide d128, 1 unique mask tiles | stream-verified | 666 | 1 | 72,022 | 72,022 | none | n/a | exact | 0.99860 | 5.300% | `d128_causal_False_32` |
| `attention` | 32x64 2/1 d128 causal FP8 out | bfloat16 | attention wide d128, 1 unique mask tiles | stream-verified | 715 | 1 | 69,950 | 69,950 | none | n/a | exact | 0.99852 | 5.437% | `d128_causal_True_32` |
| `attention` | 32x64 2/1 d128 none | bfloat16 | attention wide d128, 0 unique mask tiles | stream-verified | 658 | 1 | 69,924 | 69,924 | none | n/a | exact | 0.99876 | 5.000% | `d128_none_False_32` |
| `attention` | 32x64 2/1 d128 none FP8 out | bfloat16 | attention wide d128, 0 unique mask tiles | stream-verified | 707 | 1 | 67,852 | 67,852 | none | n/a | exact | 0.99849 | 5.503% | `d128_none_True_32` |
| `rmsnorm` | 128x32 | bfloat16 | streamed tile-row RMSNorm, FP8 out | verified | 145 | 1 | 66,186 | 66,186 | none | n/a | exact | 0.99963 | 2.738% | `norm_128_32` |
| `rmsnorm` | 32x32 | bfloat16 | streamed tile-row RMSNorm, FP8 out | verified | 148 | 1 | 54,186 | 54,186 | none | n/a | exact | 0.99966 | 2.613% | `norm_32_32` |

`words` is the executed program size in 32-bit instruction words (after the arena relocation the bench applies). Bit-exact covers the logical rows and columns the model consumes; a padding note counts output words in the zero-padded rows or columns that differ from the golden (`+0/-0`: only the sign of zero differs).

## Unmapped ops (0 rows, 0 instances, 0 FLOPs)

None: every exported op has a placement.

## Notes

New-shape acceptance: cosine >= 0.98, relative L2 error <= 0.20; both families lack an independent Atlas-arithmetic golden. All cycle counts are full launches.
