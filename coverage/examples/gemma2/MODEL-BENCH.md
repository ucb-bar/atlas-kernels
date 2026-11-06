# gemma2 kernel bench at model shapes

Reproduce with the export, mapping, and benchmark commands in [STARTER.md](STARTER.md). This historical report uses random model weights and synthetic per-kernel operands; it is not a model bringup.

Historical generated coverage input (not retained here): `coverage.json` (SHA-256 `d5c4f4727107f8b38b3345b04b57b05c5981d1e85ffaa97f112ce0f8c40f8175`), coverage status INCOMPLETE. The coverage derives 10 distinct launches; 10 executed on npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` with numerics="rtl" on seeded synthetic inputs at the model's shape (10 worker processes; 16,916 job-seconds in all, longest job 5,516 s).

Model cycles are npu_model Core cycle-model counts, not RTL cycles. Cycles x count multiplies one launch by the number of launches in the model; the sums below are device-only estimates that ignore host time, launch overhead, and data movement between launches. Bit-exact compares output bytes with the independent lib/golden reference where the family has one; cosine and rel-err compare the decoded output with an FP32 PyTorch reference on the unquantized inputs, so they include FP8 input quantization. Inputs are synthetic, so fidelity and data-dependent cycles are representative, not the model's.

A distinct launch groups the same shape and variant; a distinct program is the address-specific program text.

Shares below are by launch kind: `epilogue`, `rne_add`, `rope`, and `smol_fusion` include their GEMM projection and fused stage, while `projection_gemm` covers unfused panels. They are not shares of non-GEMM time; the generator metadata does not provide separately measured cycles for those stages.

Compiler backend: machine IR, atlas-opt stream and timing verifiers, and atlas-emit, with minimum tier `stream-verified`. Successful jobs require emitted words to equal npu_model's assembler; new benchmark runs write binary SHA-256 digests, diagnostics, and word counts under the selected output directory.

| device launches | derived here | qualified kit |
| --- | ---: | ---: |
| projection_panels | 1,904 | not declared |

## Summary

Device cycle estimate (sum of model cycles x count over 1,904 of 1,904 launches): 22,795,072,704 npu_model model cycles. It ignores host time and launch overhead.

Compiler tier counts by distinct launch: 4 `stream-verified`, 6 `verified`.

| launch kind | launches | launches with cycles | model cycles x count | share |
| --- | ---: | ---: | ---: | ---: |
| `projection_gemm` | 1,904 | 1,904 | 22,795,072,704 | 100.00% |

Top shapes by total cycles:

| family | shape | count | model cycles | cycles x count |
| --- | --- | ---: | ---: | ---: |
| `projection_gemm` | 128x2304x1024 | 822 | 17,079,242 | 14,039,136,924 |
| `projection_gemm` | 128x9216x1024 | 52 | 67,675,082 | 3,519,104,264 |
| `projection_gemm` | 32x2304x1024 (logical 1x2304x1024) | 822 | 4,269,848 | 3,509,815,056 |
| `projection_gemm` | 32x9216x1024 (logical 1x9216x1024) | 52 | 16,918,808 | 879,778,016 |
| `projection_gemm` | 128x9216x256 | 26 | 16,918,826 | 439,889,476 |
| `projection_gemm` | 128x2048x2048 | 26 | 6,635,373 | 172,519,698 |
| `projection_gemm` | 32x9216x256 (logical 1x9216x256) | 26 | 4,229,744 | 109,973,344 |
| `projection_gemm` | 32x2048x2048 (logical 1x2048x2048) | 26 | 3,337,065 | 86,763,690 |
| `projection_gemm` | 128x2048x256 | 26 | 1,005,301 | 26,137,826 |
| `projection_gemm` | 32x2048x256 (logical 1x2048x256) | 26 | 459,785 | 11,954,410 |

Generator or functional failures: 0. Timing remains PENDING for any launch without cycles or timed/functional equality.

Shapes that took more than 20 minutes on npu_model:
- `projection_gemm` 128x2048x2048: functional 1028.8 s, timed 1493.0 s; completed; cycles: full launch
- `projection_gemm` 128x2304x1024: functional 607.8 s, timed 1415.9 s; completed; cycles: full launch
- `projection_gemm` 128x9216x1024: functional 2488.9 s, timed 5516.4 s; completed; cycles: full launch
- `projection_gemm` 128x9216x256: functional 518.4 s, timed 1238.7 s; completed; cycles: full launch

## Launches

| family | shape | export dtype | variant | tier | words | count | model cycles | cycles x count | golden | bit-exact | timed equals functional | cosine | rel-err | modules |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | :-: | :-: | ---: | ---: | --- |
| `projection_gemm` | 128x2048x2048 | bfloat16 | tiny recipe | stream-verified | 2048 | 26 | 6,635,373 | 172,519,698 | ordered_matmul | exact | exact | 0.99795 | 6.407% | `m.model.layers.*.self_attn.o_proj` |
| `projection_gemm` | 32x2048x2048 (logical 1x2048x2048) | bfloat16 | tiny_breadth recipe | stream-verified | 116 | 26 | 3,337,065 | 86,763,690 | ordered_matmul | exact | exact | 0.99802 | 6.296% | `m.model.layers.*.self_attn.o_proj` |
| `projection_gemm` | 128x2048x256 | bfloat16 | tiny recipe | stream-verified | 2048 | 26 | 1,005,301 | 26,137,826 | ordered_matmul | exact | exact | 0.99794 | 6.411% | `m.model.layers.*.self_attn.o_proj` |
| `projection_gemm` | 32x2048x256 (logical 1x2048x256) | bfloat16 | tiny_breadth recipe | stream-verified | 116 | 26 | 459,785 | 11,954,410 | ordered_matmul | exact | exact | 0.99804 | 6.251% | `m.model.layers.*.self_attn.o_proj` |
| `projection_gemm` | 128x2304x1024 | bfloat16 | default schedule (no traffic-reuse recipe for this shape) | verified | 1494 | 822 | 17,079,242 | 14,039,136,924 | ordered_matmul | exact | exact | 0.99779 | 6.649% | `m.model.layers.*.self_attn.q_proj`, `m.model.layers.*.self_attn.k_proj`, `m.model.layers.*.self_attn.v_proj` ... |
| `projection_gemm` | 32x2304x1024 (logical 1x2304x1024) | bfloat16 | default schedule (no traffic-reuse recipe for this shape) | verified | 1494 | 822 | 4,269,848 | 3,509,815,056 | ordered_matmul | exact | exact | 0.99770 | 6.776% | `m.model.layers.*.self_attn.q_proj`, `m.model.layers.*.self_attn.k_proj`, `m.model.layers.*.self_attn.v_proj` ... |
| `projection_gemm` | 128x9216x1024 | bfloat16 | default schedule (no traffic-reuse recipe for this shape) | verified | 5814 | 52 | 67,675,082 | 3,519,104,264 | ordered_matmul | exact | exact | 0.99377 | 11.148% | `m.model.layers.*.mlp.down_proj` |
| `projection_gemm` | 32x9216x1024 (logical 1x9216x1024) | bfloat16 | default schedule (no traffic-reuse recipe for this shape) | verified | 5814 | 52 | 16,918,808 | 879,778,016 | ordered_matmul | exact | exact | 0.99290 | 11.930% | `m.model.layers.*.mlp.down_proj` |
| `projection_gemm` | 128x9216x256 | bfloat16 | default schedule (no traffic-reuse recipe for this shape) | verified | 5814 | 26 | 16,918,826 | 439,889,476 | ordered_matmul | exact | exact | 0.99374 | 11.171% | `m.model.layers.*.mlp.down_proj` |
| `projection_gemm` | 32x9216x256 (logical 1x9216x256) | bfloat16 | default schedule (no traffic-reuse recipe for this shape) | verified | 5814 | 26 | 4,229,744 | 109,973,344 | ordered_matmul | exact | exact | 0.99463 | 10.389% | `m.model.layers.*.mlp.down_proj` |

`words` is the executed program size in 32-bit instruction words (after the arena relocation the bench applies). Bit-exact covers the logical rows and columns the model consumes; a padding note counts output words in the zero-padded rows or columns that differ from the golden (`+0/-0`: only the sign of zero differs).

Unmapped operations are retained once in [COVERAGE.md](COVERAGE.md).
