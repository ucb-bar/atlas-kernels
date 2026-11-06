# pi0 kernel bench at model shapes

Reproduce with the export, mapping, and benchmark commands in [STARTER.md](STARTER.md). This historical report uses random model weights and synthetic per-kernel operands; it is not a model bringup.

Historical generated coverage input (not retained here): `coverage.json` (SHA-256 `d305bf9f80dd88a6e657070bdb3ba8c82ce726bea4db21c5d904db47f365e2b3`), coverage status INCOMPLETE. The coverage derives 21 distinct launches; 21 executed on npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` with numerics="rtl" on seeded synthetic inputs at the model's shape (8 worker processes; 42,324 job-seconds in all, longest job 14,400 s).

Model cycles are npu_model Core cycle-model counts, not RTL cycles. Cycles x count multiplies one launch by the number of launches in the model; the sums below are device-only estimates that ignore host time, launch overhead, and data movement between launches. Bit-exact compares output bytes with the independent lib/golden reference where the family has one; cosine and rel-err compare the decoded output with an FP32 PyTorch reference on the unquantized inputs, so they include FP8 input quantization. Inputs are synthetic, so fidelity and data-dependent cycles are representative, not the model's.

A distinct launch groups the same shape and variant; a distinct program is the address-specific program text.

Shares below are by launch kind: `epilogue`, `rne_add`, `rope`, and `smol_fusion` include their GEMM projection and fused stage, while `projection_gemm` covers unfused panels. They are not shares of non-GEMM time; the generator metadata does not provide separately measured cycles for those stages.

Compiler backend: machine IR, atlas-opt stream and timing verifiers, and atlas-emit, with minimum tier `stream-verified`. Successful jobs require emitted words to equal npu_model's assembler; new benchmark runs write binary SHA-256 digests, diagnostics, and word counts under the selected output directory.

| device launches | derived here | qualified kit |
| --- | ---: | ---: |
| projection_panels | 5,912 | not declared |

## Summary

Device cycle estimate (sum of model cycles x count over 5,696 of 5,912 launches): 45,300,319,472 npu_model model cycles. It ignores host time and launch overhead.

Compiler tier counts by distinct launch: 3 `stream-verified`, 18 `verified`.

| launch kind | launches | launches with cycles | model cycles x count | share |
| --- | ---: | ---: | ---: | ---: |
| `projection_gemm` | 5,912 | 5,696 | 45,300,319,472 | 100.00% |

Top shapes by total cycles:

| family | shape | count | model cycles | cycles x count |
| --- | --- | ---: | ---: | ---: |
| `projection_gemm` | 128x2048x2048 | 1,944 | 6,635,373 | 12,899,165,112 |
| `projection_gemm` | 128x1152x1152 | 648 | 9,727,418 | 6,303,366,864 |
| `projection_gemm` | 128x1152x2048 | 330 | 17,293,130 | 5,706,732,900 |
| `projection_gemm` | 128x4320x1024 (logical 128x4304x1024) | 162 | 31,836,363 | 5,157,490,806 |
| `projection_gemm` | 64x2048x2048 (logical 48x2048x2048) | 324 | 15,205,310 | 4,926,520,440 |
| `projection_gemm` | 64x4096x1024 (logical 51x4096x1024) | 180 | 15,098,366 | 2,717,705,880 |
| `projection_gemm` | 64x1024x2048 (logical 51x1024x2048) | 900 | 2,582,193 | 2,323,973,700 |
| `projection_gemm` | 64x16384x1024 (logical 48x16384x1024) | 36 | 60,072,446 | 2,162,608,056 |
| `projection_gemm` | 64x2048x1024 (logical 51x2048x1024) | 180 | 7,602,686 | 1,368,483,480 |
| `projection_gemm` | 128x4320x128 (logical 128x4304x128) | 162 | 3,979,611 | 644,696,982 |

Generator or functional failures: 0. Timing remains PENDING for any launch without cycles or timed/functional equality.

Shapes that took more than 20 minutes on npu_model:
- `projection_gemm` 128x1152x2048: functional 645.1 s, timed 1396.9 s; completed; cycles: full launch
- `projection_gemm` 128x16384x1024: functional 4345.9 s, timed 14400.2 s; timed run exceeded 14400 s; cycles: none
- `projection_gemm` 64x16384x1024 (logical 48x16384x1024): functional 1693.1 s, timed 4552.1 s; completed; cycles: full launch
- `projection_gemm` 128x2048x2048: functional 1000.6 s, timed 1448.9 s; completed; cycles: full launch
- `projection_gemm` 64x4096x1024 (logical 51x4096x1024): functional 599.2 s, timed 1212.2 s; completed; cycles: full launch
- `projection_gemm` 128x4320x1024 (logical 128x4304x1024): functional 888.1 s, timed 2620.8 s; completed; cycles: full launch

## Launches

| family | shape | export dtype | variant | tier | words | count | model cycles | cycles x count | golden | bit-exact | timed equals functional | cosine | rel-err | modules |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | :-: | :-: | ---: | ---: | --- |
| `projection_gemm` | 64x1024x1024 (logical 50x1024x1024) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 694 | 10 | 3,854,846 | 38,548,460 | ordered_matmul | exact | exact | 0.99862 | 5.256% | `model.action_time_mlp_out` |
| `projection_gemm` | 64x1024x2048 (logical 51x1024x2048) | float32 | blocked recipe | stream-verified | 143 | 900 | 2,582,193 | 2,323,973,700 | ordered_matmul | exact | exact | 0.99861 | 5.264% | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn.q_proj`, `model.paligemma_with_expert.gemma_expert.model.layers.*.mlp.gate_proj`, `model.paligemma_with_expert.gemma_expert.model.layers.*.mlp.up_proj` |
| `projection_gemm` | 64x1024x256 (logical 51x1024x256) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 694 | 360 | 963,758 | 346,952,880 | ordered_matmul | exact | exact | 0.99862 | 5.247% | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn.k_proj`, `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn.v_proj` |
| `projection_gemm` | 64x1024x32 (logical 50x1024x32) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 694 | 10 | 120,524 | 1,205,240 | ordered_matmul | exact | exact | 0.99871 | 5.088% | `model.action_out_proj` |
| `projection_gemm` | 128x1152x1152 | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 774 | 648 | 9,727,418 | 6,303,366,864 | ordered_matmul | exact | exact | 0.99853 | 5.425% | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn.q_proj`, `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn.k_proj`, `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn.v_proj` ... |
| `projection_gemm` | 128x1152x2048 | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 774 | 330 | 17,293,130 | 5,706,732,900 | ordered_matmul | exact | exact | 0.99853 | 5.422% | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.mlp.fc1`, `model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear` |
| `projection_gemm` | 128x1152x224 (logical 128x1152x208) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 774 | 162 | 1,891,502 | 306,423,324 | ordered_matmul | exact | exact | 0.99856 | 5.363% | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.mlp.fc1` |
| `projection_gemm` | 128x16384x1024 | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 10294 | 216 | PENDING |  | ordered_matmul | exact | not compared | 0.99001 | 14.099% | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.mlp.down_proj` |
| `projection_gemm` | 64x16384x1024 (logical 48x16384x1024) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 10294 | 36 | 60,072,446 | 2,162,608,056 | ordered_matmul | exact | exact | 0.98976 | 14.273% | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.mlp.down_proj` |
| `projection_gemm` | 64x2048x1024 (logical 50x2048x1024) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 1334 | 10 | 7,602,686 | 76,026,860 | ordered_matmul | exact | exact | 0.99794 | 6.419% | `model.action_time_mlp_in` |
| `projection_gemm` | 64x2048x1024 (logical 51x2048x1024) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 1334 | 180 | 7,602,686 | 1,368,483,480 | ordered_matmul | exact | exact | 0.99797 | 6.373% | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn.o_proj` |
| `projection_gemm` | 128x2048x2048 | float32 | tiny recipe | stream-verified | 2048 | 1,944 | 6,635,373 | 12,899,165,112 | ordered_matmul | exact | exact | 0.99795 | 6.407% | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn.q_proj`, `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn.o_proj`, `model.paligemma_with_expert.paligemma.model.language_model.layers.*.mlp.gate_proj` ... |
| `projection_gemm` | 64x2048x2048 (logical 48x2048x2048) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 1334 | 324 | 15,205,310 | 4,926,520,440 | ordered_matmul | exact | exact | 0.99795 | 6.394% | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn.q_proj`, `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn.o_proj`, `model.paligemma_with_expert.paligemma.model.language_model.layers.*.mlp.gate_proj` ... |
| `projection_gemm` | 128x2048x256 | float32 | tiny recipe | stream-verified | 2048 | 216 | 1,005,301 | 217,145,016 | ordered_matmul | exact | exact | 0.99794 | 6.411% | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn.k_proj`, `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn.v_proj` |
| `projection_gemm` | 64x2048x256 (logical 48x2048x256) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 1334 | 36 | 1,900,718 | 68,425,848 | ordered_matmul | exact | exact | 0.99799 | 6.335% | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn.k_proj`, `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn.v_proj` |
| `projection_gemm` | 32x32x1024 (logical 1x32x1024) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 74 | 10 | 112,088 | 1,120,880 | ordered_matmul | exact | exact | 0.99941 | 3.570% | `model.state_proj` |
| `projection_gemm` | 64x32x1024 (logical 50x32x1024) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 75 | 10 | 224,127 | 2,241,270 | ordered_matmul | exact | exact | 0.99931 | 3.727% | `model.action_in_proj` |
| `projection_gemm` | 64x4096x1024 (logical 51x4096x1024) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 2614 | 180 | 15,098,366 | 2,717,705,880 | ordered_matmul | exact | exact | 0.99670 | 8.121% | `model.paligemma_with_expert.gemma_expert.model.layers.*.mlp.down_proj` |
| `projection_gemm` | 128x4320x1024 (logical 128x4304x1024) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 2755 | 162 | 31,836,363 | 5,157,490,806 | ordered_matmul | exact | exact | 0.99655 | 8.301% | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.mlp.fc2` |
| `projection_gemm` | 128x4320x128 (logical 128x4304x128) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 2755 | 162 | 3,979,611 | 644,696,982 | ordered_matmul | exact | exact | 0.99655 | 8.301% | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.mlp.fc2` |
| `projection_gemm` | 128x608x1152 (logical 128x588x1152) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 435 | 6 | 5,247,579 | 31,485,474 | ordered_matmul | exact | exact | 0.99890 | 4.683% | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding` |

`words` is the executed program size in 32-bit instruction words (after the arena relocation the bench applies). Bit-exact covers the logical rows and columns the model consumes; a padding note counts output words in the zero-padded rows or columns that differ from the golden (`+0/-0`: only the sign of zero differs).

Unmapped operations are retained once in [COVERAGE.md](COVERAGE.md).
