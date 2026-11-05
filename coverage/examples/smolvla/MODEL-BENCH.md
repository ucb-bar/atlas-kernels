# smolvla kernel bench at model shapes

Historical report from `6123bef`, retained below with its original synthetic-input receipts and row-subset estimate. Later full captured-input measurements qualify all 24 vision launches: original v2 takes 26,092,660 cycles, and opt-in vision-fast takes 19,752,187 cycles. Replacing that estimate gives 5,160,679,740 original device cycles, or 5,008,508,388 with the opt-in schedule and all other published counts unchanged. See [the measured correction and qualification](../../../kernels/attention/VISION-FAST.md).

Regenerate: `PYTHONPATH="$NPU_MODEL" "$MODEL_PYTHON" coverage/bench_model.py --coverage coverage/coverage.json --model smolvla --model-root "$NPU_MODEL" --out "$OUT/bench" --workers 32 --atlas-mlir-bin "$ATLAS_MLIR_BIN" --min-tier stream-verified`

Input: `coverage/coverage.json` (SHA-256 `4abb075f1683e18d7e1591b408feaab8ac55652db0dc3a81282c3de56b6c17a5`), coverage status COMPLETE. The coverage derives 28 distinct launches; 28 executed on npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` with numerics="rtl" on seeded synthetic inputs at the model's shape (16 worker processes; 12,678 job-seconds in all, longest job 1,659 s).

Model cycles are npu_model Core cycle-model counts, not RTL cycles. Cycles x count multiplies one launch by the number of launches in the model; the sums below are device-only estimates that ignore host time, launch overhead, and data movement between launches. Bit-exact compares output bytes with the independent lib/golden reference where the family has one; cosine and rel-err compare the decoded output with an FP32 PyTorch reference on the unquantized inputs, so they include FP8 input quantization. Inputs are synthetic, so fidelity and data-dependent cycles are representative, not the model's.

A distinct launch groups the same shape and variant; a distinct program is the address-specific program text, as in the separate 1,589-program kit check.

Shares below are by launch kind: `epilogue`, `rne_add`, `rope`, and `smol_fusion` include their GEMM projection and fused stage, while `projection_gemm` covers unfused panels. They are not shares of non-GEMM time; the generator metadata does not provide separately measured cycles for those stages.

Compiler backend: machine IR, atlas-opt stream and timing verifiers, and atlas-emit, with minimum tier `stream-verified`. Successful jobs require emitted words to equal npu_model's assembler; per-job JSON records binary SHA-256 digests, diagnostics, and word counts.

| device launches | derived here | qualified kit |
| --- | ---: | ---: |
| projection_panels | 2,715 | 2,715 |
| attention | 200 | 200 |

## Summary

Device cycle estimate (sum of model cycles x count over 2,915 of 2,915 launches): 5,560,633,884 (includes 24 estimated launches) npu_model model cycles. It ignores host time and launch overhead. 24 of those launches use a row-subset estimate (`attention` 1024x1024 12/12 d64 none).

These totals include historically reused functional and timed results; zero functional or timed bench jobs were rerun for this retained report. Reconstructed staging, program-word checks, and per-job result hashes are recorded in `reuse-proof.json` beside the generated JSON.

Compiler tier counts by distinct launch: 27 `stream-verified`, 1 `verified`.

| launch kind | launches | launches with cycles | model cycles x count | share |
| --- | ---: | ---: | ---: | ---: |
| `projection_gemm` | 2,523 | 2,523 | 3,424,717,132 | 61.59% |
| `attention` | 200 | 200 | 1,184,213,840 | 21.30% |
| `smol_fusion` | 192 | 192 | 951,702,912 | 17.12% |

Unsupported or failed launches: 0.

Shapes that took more than 20 minutes on npu_model:
- `attention` 1024x1024 12/12 d64 none: functional 1037.7 s, timed 1200.1 s; timed run exceeded 1200 s; cycles: estimate: 128-row subset (5,344,677 cycles) x 8; timeout provenance: timeout marker reconstructed from bench-smolvla.log after the merge crash (retained log: 1,200.1 s)
- `projection_gemm` 64x12288x960: functional 1053.9 s, timed 1658.9 s; completed; cycles: full launch

## Launches

| family | shape | export dtype | variant | tier | words | count | model cycles | cycles x count | golden | bit-exact | timed equals functional | cosine | rel-err | modules |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | :-: | :-: | ---: | ---: | --- |
| `attention` | 1024x1024 12/12 d64 none | bfloat16 | attention v2, 0 unique mask tiles | stream-verified | 420 | 24 | 42,757,416 (estimate) | 1,026,177,984 | none | n/a | not compared | 0.99846 | 5.572% | `vlm_with_expert.vlm.model.vision_model.encoder.layers.*.self_attn` |
| `smol_fusion` | 128x768x3072 bias+gelu_tanh FP8 out | bfloat16 | blocked recipe + fc1 bias and GELU (tanh), FP8 out (smol_fusion_fp8.stage_fc1_fp8) | stream-verified | 565 | 192 | 4,956,786 | 951,702,912 | none | n/a | exact | 0.99824 | 5.961% | `vlm_with_expert.vlm.model.vision_model.encoder.layers.*.mlp.fc1` |
| `projection_gemm` | 128x768x768 | bfloat16 | resident recipe | stream-verified | 479 | 784 | 1,077,912 | 845,083,008 | ordered_matmul | exact | exact | 0.99878 | 4.943% | `vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding`, `vlm_with_expert.vlm.model.vision_model.encoder.layers.*.self_attn.q_proj`, `vlm_with_expert.vlm.model.vision_model.encoder.layers.*.self_attn.k_proj` ... |
| `projection_gemm` | 128x3072x768 | bfloat16 | stream recipe | stream-verified | 3373 | 192 | 3,848,658 | 738,942,336 | ordered_matmul | exact | exact | 0.99734 | 7.289% | `vlm_with_expert.vlm.model.vision_model.encoder.layers.*.mlp.fc2` |
| `projection_gemm` | 64x736x2048 (logical 50x720x2048) | bfloat16 | blocked recipe | stream-verified | 146 | 320 | 1,916,097 | 613,151,040 | ordered_matmul | exact | exact | 0.99880 | 4.904% | `vlm_with_expert.lm_expert.layers.*.mlp.gate_proj`, `vlm_with_expert.lm_expert.layers.*.mlp.up_proj` |
| `projection_gemm` | 64x2048x736 (logical 50x2048x720) | bfloat16 | blocked recipe | stream-verified | 144 | 160 | 1,843,626 | 294,980,160 | ordered_matmul | exact | exact | 0.99795 | 6.396% | `vlm_with_expert.lm_expert.layers.*.mlp.down_proj` |
| `projection_gemm` | 64x736x960 (logical 50x720x960) | bfloat16 | blocked recipe | stream-verified | 146 | 160 | 917,415 | 146,786,400 | ordered_matmul | exact | exact | 0.99879 | 4.916% | `vlm_with_expert.lm_expert.layers.*.self_attn.q_proj` |
| `projection_gemm` | 64x960x736 (logical 50x960x720) | bfloat16 | blocked recipe | stream-verified | 146 | 160 | 905,046 | 144,807,360 | ordered_matmul | exact | exact | 0.99865 | 5.190% | `vlm_with_expert.lm_expert.layers.*.self_attn.o_proj` |
| `projection_gemm` | 128x960x2560 | bfloat16 | blocked recipe | stream-verified | 218 | 32 | 4,163,274 | 133,224,768 | ordered_matmul | exact | exact | 0.99865 | 5.196% | `vlm_with_expert.vlm.model.text_model.layers.*.mlp.gate_proj`, `vlm_with_expert.vlm.model.text_model.layers.*.mlp.up_proj` |
| `projection_gemm` | 64x960x2560 (logical 49x960x2560) | bfloat16 | blocked recipe | stream-verified | 145 | 32 | 3,030,974 | 96,991,168 | ordered_matmul | exact | exact | 0.99865 | 5.187% | `vlm_with_expert.vlm.model.text_model.layers.*.mlp.gate_proj`, `vlm_with_expert.vlm.model.text_model.layers.*.mlp.up_proj` |
| `attention` | 50x227 15/5 d64 arbitrary (all_visible) | float32 | attention v2, 3 unique mask tiles | stream-verified | 429 | 80 | 907,744 | 72,619,520 | none | n/a | exact | 0.99854 | 5.415% | `(top)` |
| `projection_gemm` | 128x2560x960 | bfloat16 | blocked recipe | stream-verified | 254 | 16 | 3,994,175 | 63,906,800 | ordered_matmul | exact | exact | 0.99761 | 6.903% | `vlm_with_expert.vlm.model.text_model.layers.*.mlp.down_proj` |
| `attention` | 50x177 15/5 d64 arbitrary (all_visible) | float32 | attention v2, 3 unique mask tiles | stream-verified | 429 | 80 | 731,850 | 58,548,000 | none | n/a | exact | 0.99857 | 5.363% | `(top)` |
| `projection_gemm` | 64x736x320 (logical 50x720x320) | bfloat16 | blocked recipe | stream-verified | 146 | 160 | 329,955 | 52,792,800 | ordered_matmul | exact | exact | 0.99883 | 4.829% | `vlm_with_expert.lm_expert.layers.*.self_attn.k_proj`, `vlm_with_expert.lm_expert.layers.*.self_attn.v_proj` |
| `projection_gemm` | 128x960x960 | bfloat16 | blocked recipe | stream-verified | 218 | 32 | 1,620,174 | 51,845,568 | ordered_matmul | exact | exact | 0.99864 | 5.212% | `vlm_with_expert.vlm.model.text_model.layers.*.self_attn.q_proj`, `vlm_with_expert.vlm.model.text_model.layers.*.self_attn.o_proj` |
| `projection_gemm` | 64x2560x960 (logical 49x2560x960) | bfloat16 | blocked recipe | stream-verified | 143 | 16 | 2,942,593 | 47,081,488 | ordered_matmul | exact | exact | 0.99767 | 6.829% | `vlm_with_expert.vlm.model.text_model.layers.*.mlp.down_proj` |
| `projection_gemm` | 128x320x320 | float32 | blocked recipe | stream-verified | 218 | 160 | 244,534 | 39,125,440 | ordered_matmul | exact | exact | 0.99909 | 4.265% | `vlm_with_expert.lm_expert.layers.*.self_attn.k_proj`, `vlm_with_expert.lm_expert.layers.*.self_attn.v_proj` |
| `projection_gemm` | 64x12288x960 | bfloat16 | smol_connector recipe | stream-verified | 311 | 2 | 18,824,758 | 37,649,516 | ordered_matmul | exact | exact | 0.99209 | 12.556% | `vlm_with_expert.vlm.model.connector.modality_projection.proj` |
| `projection_gemm` | 64x960x960 (logical 49x960x960) | bfloat16 | blocked recipe | stream-verified | 145 | 32 | 1,166,124 | 37,315,968 | ordered_matmul | exact | exact | 0.99867 | 5.154% | `vlm_with_expert.vlm.model.text_model.layers.*.self_attn.q_proj`, `vlm_with_expert.vlm.model.text_model.layers.*.self_attn.o_proj` |
| `attention` | 177x177 15/5 d64 arbitrary (prefix_state_block) | float32 | attention v2, 3 unique mask tiles | stream-verified | 429 | 16 | 1,679,271 | 26,868,336 | none | n/a | exact | 0.99862 | 5.264% | `(top)` |
| `projection_gemm` | 64x320x320 (logical 49x320x320) | float32 | blocked recipe | stream-verified | 145 | 160 | 162,384 | 25,981,440 | ordered_matmul | exact | exact | 0.99906 | 4.326% | `vlm_with_expert.lm_expert.layers.*.self_attn.k_proj`, `vlm_with_expert.lm_expert.layers.*.self_attn.v_proj` |
| `projection_gemm` | 128x960x320 | bfloat16 | blocked recipe | stream-verified | 218 | 32 | 602,934 | 19,293,888 | ordered_matmul | exact | exact | 0.99866 | 5.181% | `vlm_with_expert.vlm.model.text_model.layers.*.self_attn.k_proj`, `vlm_with_expert.vlm.model.text_model.layers.*.self_attn.v_proj` |
| `projection_gemm` | 64x960x320 (logical 49x960x320) | bfloat16 | blocked recipe | stream-verified | 145 | 32 | 420,184 | 13,445,888 | ordered_matmul | exact | exact | 0.99869 | 5.122% | `vlm_with_expert.vlm.model.text_model.layers.*.self_attn.k_proj`, `vlm_with_expert.vlm.model.text_model.layers.*.self_attn.v_proj` |
| `projection_gemm` | 64x1440x736 (logical 50x1440x720) | float32 | blocked recipe | stream-verified | 147 | 10 | 1,319,137 | 13,191,370 | ordered_matmul | exact | exact | 0.99833 | 5.773% | `action_time_mlp_in` |
| `projection_gemm` | 64x736x736 (logical 50x720x720) | float32 | blocked recipe | stream-verified | 147 | 10 | 711,805 | 7,118,050 | ordered_matmul | exact | exact | 0.99881 | 4.887% | `action_time_mlp_out` |
| `projection_gemm` | 64x32x736 (logical 50x32x720) | float32 | blocked recipe | stream-verified | 118 | 10 | 104,378 | 1,043,780 | ordered_matmul | exact | exact | 0.99924 | 3.890% | `action_in_proj` |
| `projection_gemm` | 64x736x32 (logical 50x720x32) | float32 | default schedule (no traffic-reuse recipe for this shape; not used by a qualified kit) | verified | 516 | 10 | 87,586 | 875,860 | ordered_matmul | exact | exact | 0.99880 | 4.908% | `action_out_proj` |
| `projection_gemm` | 32x32x960 (logical 1x32x960) | float32 | blocked recipe | stream-verified | 96 | 1 | 83,036 | 83,036 | ordered_matmul | exact | exact | 0.99938 | 3.766% | `state_proj` |

`words` is the executed program size in 32-bit instruction words (after the arena relocation the bench applies). Bit-exact covers the logical rows and columns the model consumes; a padding note counts output words in the zero-padded rows or columns that differ from the golden (`+0/-0`: only the sign of zero differs).

## Unmapped ops (0 rows, 0 instances, 0 FLOPs)

None: every exported op has a placement.

## Notes

Launches come from `coverage/coverage.json` and the `launch` hints in `coverage/mapping.yaml`: projection calls split at 128 rows and padded to multiples of 32 (`smol_full_plan.panels()`), which reproduces the kit's 2,715 panels; the 192 vision fc1 panels bench as the fused fc1 bias and GELU program staged by `smol_fusion_fp8.stage_fc1_fp8`.

The 1024x1024 vision attention timed run exceeded 20 minutes (stopped at 1,200 s), so its cycles are an estimate: the 128-query-row subset of the same launch (mask none, so every 128-row block does the same work) took 5,344,677 model cycles, multiplied by 8. The estimate counts once-per-launch work 8 times. Its functional run covered all 1,024 rows. The 64x12288x960 connector (timed 1,659 s) also took more than 20 minutes and is a full-launch count.

Attention masks are synthetic stand-ins: the 177x177 prefix mask hides the state key from earlier rows (`prefix_state_block`), and the expert masks show every key (`all_visible`). The model's masks also hide padded language keys, which changes the mask-descriptor table and may change cycles.

The 4 SmolVLA shapes that `lib/kernel_bench.py` also benches (`coverage/RESULTS.md`) give identical model cycles here, including the fused fc1, which that bench reconstructs from the epilogue generator and this bench stages through `stage_fc1_fp8`.

Functional and timed results are reused from Track G only after regenerating each job with the pinned Track G staging code and this bench, proving equal program-word and staged-arena SHA-256 digests, and passing the compiler word check; the per-job proof is in `reuse-proof.json` beside the generated JSON. No functional or timed job changed, including the row-subset estimate.
