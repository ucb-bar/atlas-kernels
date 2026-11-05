# tinyllama kernel bench at model shapes

Regenerate: `PYTHONPATH="$NPU_MODEL" "$MODEL_PYTHON" coverage/bench_model.py --coverage coverage/coverage.json --model tinyllama --model-root "$NPU_MODEL" --out "$OUT/bench" --workers 32 --atlas-mlir-bin "$ATLAS_MLIR_BIN" --min-tier stream-verified`

Input: `coverage/coverage.json` (SHA-256 `4abb075f1683e18d7e1591b408feaab8ac55652db0dc3a81282c3de56b6c17a5`), coverage status COMPLETE. The coverage derives 26 distinct launches; 26 executed on npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` with numerics="rtl" on seeded synthetic inputs at the model's shape (16 worker processes; 17,318 job-seconds in all, longest job 1,396 s).

Model cycles are npu_model Core cycle-model counts, not RTL cycles. Cycles x count multiplies one launch by the number of launches in the model; the sums below are device-only estimates that ignore host time, launch overhead, and data movement between launches. Bit-exact compares output bytes with the independent lib/golden reference where the family has one; cosine and rel-err compare the decoded output with an FP32 PyTorch reference on the unquantized inputs, so they include FP8 input quantization. Inputs are synthetic, so fidelity and data-dependent cycles are representative, not the model's.

A distinct launch groups the same shape and variant; a distinct program is the address-specific program text, as in the separate 1,589-program kit check.

Shares below are by launch kind: `epilogue`, `rne_add`, `rope`, and `smol_fusion` include their GEMM projection and fused stage, while `projection_gemm` covers unfused panels. They are not shares of non-GEMM time; the generator metadata does not provide separately measured cycles for those stages.

Compiler backend: machine IR, atlas-opt stream and timing verifiers, and atlas-emit, with minimum tier `stream-verified`. Successful jobs require emitted words to equal npu_model's assembler; per-job JSON records binary SHA-256 digests, diagnostics, and word counts.

| device launches | derived here | qualified kit |
| --- | ---: | ---: |
| projection_panels | 840 | 840 |
| attention | 66 | 66 |
| rmsnorm | 132 | 132 |

## Summary

Device cycle estimate (sum of model cycles x count over 1,038 of 1,038 launches): 3,537,507,930 npu_model model cycles. It ignores host time and launch overhead.

These totals include historically reused functional and timed results; zero functional or timed bench jobs were rerun for this retained report. Reconstructed staging, program-word checks, and per-job result hashes are recorded in `reuse-proof.json` beside the generated JSON.

Compiler tier counts by distinct launch: 24 `stream-verified`, 2 `verified`.

| launch kind | launches | launches with cycles | model cycles x count | share |
| --- | ---: | ---: | ---: | ---: |
| `epilogue` | 396 | 396 | 1,658,659,772 | 46.89% |
| `projection_gemm` | 246 | 246 | 793,412,598 | 22.43% |
| `rne_add` | 66 | 66 | 610,241,654 | 17.25% |
| `rope` | 132 | 132 | 361,121,200 | 10.21% |
| `rmsnorm` | 132 | 132 | 75,488,820 | 2.13% |
| `attention` | 66 | 66 | 38,583,886 | 1.09% |

Unsupported or failed launches: 0.

Shapes that took more than 20 minutes on npu_model:
- `rne_add` 128x5632x1024: functional 1108.4 s, timed 1395.7 s; completed; cycles: full launch

## Launches

| family | shape | export dtype | variant | tier | words | count | model cycles | cycles x count | golden | bit-exact | timed equals functional | cosine | rel-err | modules |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | :-: | :-: | ---: | ---: | --- |
| `rne_add` | 128x5632x1024 | bfloat16 | tiny_breadth recipe + add_rne epilogue (kernels/epilogue/gen.py --steps add_rne; kernels/rne_add/gen.py emits only the add sequence) | stream-verified | 354 | 44 | 9,979,346 | 439,091,224 | ordered_matmul + exact sum rounded to BF16 | exact | exact | 0.99708 | 7.634% | `model.layers.*.mlp.down_proj` |
| `projection_gemm` | 32x5632x1024 (logical 1x5632x1024) | bfloat16 | tiny_breadth recipe | stream-verified | 116 | 88 | 4,501,837 | 396,161,656 | ordered_matmul | exact | exact | 0.99541 | 9.586% | `model.layers.*.mlp.down_proj` |
| `epilogue` | 128x2048x2048 mul FP8 out | bfloat16 | tiny recipe + multiply epilogue, FP8 out | stream-verified | 2316 | 44 | 6,977,261 | 306,999,484 | none | n/a | exact | 0.99764 | 6.863% | `model.layers.*.mlp.up_proj` |
| `epilogue` | 32x2048x2048 (logical 1x2048x2048) mul FP8 out | bfloat16 | tiny_breadth recipe + multiply epilogue, FP8 out | stream-verified | 153 | 88 | 3,420,329 | 300,988,952 | none | n/a | exact | 0.99764 | 6.896% | `model.layers.*.mlp.up_proj` |
| `epilogue` | 128x2048x2048 silu | bfloat16 | tiny recipe + SiLU epilogue | stream-verified | 2120 | 44 | 6,792,045 | 298,849,980 | none | n/a | exact | 0.99750 | 7.060% | `model.layers.*.mlp.gate_proj` |
| `epilogue` | 32x2048x2048 (logical 1x2048x2048) silu | bfloat16 | tiny_breadth recipe + SiLU epilogue | stream-verified | 134 | 88 | 3,376,233 | 297,108,504 | none | n/a | exact | 0.99746 | 7.136% | `model.layers.*.mlp.gate_proj` |
| `projection_gemm` | 32x2048x2048 (logical 1x2048x2048) | bfloat16 | tiny_breadth recipe | stream-verified | 116 | 74 | 3,337,065 | 246,942,810 | ordered_matmul | exact | exact | 0.99802 | 6.296% | `model.layers.*.self_attn.o_proj`, `lm_head` |
| `rne_add` | 128x2048x2048 | bfloat16 | tiny recipe + add_rne epilogue (kernels/epilogue/gen.py --steps add_rne; kernels/rne_add/gen.py emits only the add sequence) | stream-verified | 2448 | 22 | 7,779,565 | 171,150,430 | ordered_matmul + exact sum rounded to BF16 | exact | exact | 0.99908 | 4.297% | `model.layers.*.self_attn.o_proj` |
| `rope` | 128x2048x2048 | bfloat16 | tiny recipe + RoPE post-pass (positions 0..127) | stream-verified | 2337 | 22 | 7,318,670 | 161,010,740 | ordered_matmul + RoPE RNE formula | exact | exact | 0.99796 | 6.385% | `model.layers.*.self_attn.q_proj` |
| `rope` | 32x2048x2048 (logical 1x2048x2048) | bfloat16 | tiny_breadth recipe + RoPE post-pass (positions 0..31) | stream-verified | 374 | 44 | 3,508,117 | 154,357,148 | ordered_matmul + RoPE RNE formula | exact (padding: 5,248 words +0/-0) | exact | 0.99803 | 6.285% | `model.layers.*.self_attn.q_proj` |
| `epilogue` | 128x2048x1536 mul FP8 out | bfloat16 | tiny recipe + multiply epilogue, FP8 out | stream-verified | 2268 | 22 | 5,280,893 | 116,179,646 | none | n/a | exact | 0.99760 | 6.925% | `model.layers.*.mlp.up_proj` |
| `epilogue` | 32x2048x1536 (logical 1x2048x1536) mul FP8 out | bfloat16 | tiny_breadth recipe + multiply epilogue, FP8 out | stream-verified | 153 | 44 | 2,577,433 | 113,407,052 | none | n/a | exact | 0.99779 | 6.655% | `model.layers.*.mlp.up_proj` |
| `epilogue` | 128x2048x1536 silu | bfloat16 | tiny recipe + SiLU epilogue | stream-verified | 2120 | 22 | 5,144,285 | 113,174,270 | none | n/a | exact | 0.99748 | 7.095% | `model.layers.*.mlp.gate_proj` |
| `epilogue` | 32x2048x1536 (logical 1x2048x1536) silu | bfloat16 | tiny_breadth recipe + SiLU epilogue | stream-verified | 134 | 44 | 2,544,361 | 111,951,884 | none | n/a | exact | 0.99787 | 6.525% | `model.layers.*.mlp.gate_proj` |
| `projection_gemm` | 128x2048x2048 | bfloat16 | tiny recipe | stream-verified | 2048 | 15 | 6,635,373 | 99,530,595 | ordered_matmul | exact | exact | 0.99795 | 6.407% | `lm_head` |
| `rmsnorm` | 128x2048 | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 200 | 44 | 997,891 | 43,907,204 | none | n/a | exact | 0.99964 | 2.710% | `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm` |
| `rmsnorm` | 32x2048 (1 rows) | bfloat16 | standalone norm launch (DMA BF16 rows, RMS post-pass, FP8 out) | verified | 200 | 88 | 358,882 | 31,581,616 | none | n/a | exact | 0.99963 | 2.716% | `model.layers.*.input_layernorm`, `model.layers.*.post_attention_layernorm` |
| `attention` | 128x128 32/4 d64 causal FP8 out | float32 | attention v2, 1 unique mask tiles | stream-verified | 452 | 22 | 1,358,729 | 29,892,038 | none | n/a | exact | 0.99869 | 5.130% | `model.layers.*.self_attn` |
| `rope` | 128x2048x256 | bfloat16 | tiny recipe + RoPE post-pass (positions 0..127) | stream-verified | 2337 | 22 | 1,107,934 | 24,374,548 | ordered_matmul + RoPE RNE formula | exact | exact | 0.99797 | 6.375% | `model.layers.*.self_attn.k_proj` |
| `projection_gemm` | 128x2048x256 | bfloat16 | tiny recipe | stream-verified | 2048 | 22 | 1,005,301 | 22,116,622 | ordered_matmul | exact | exact | 0.99794 | 6.411% | `model.layers.*.self_attn.v_proj` |
| `rope` | 32x2048x256 (logical 1x2048x256) | bfloat16 | tiny_breadth recipe + RoPE post-pass (positions 0..31) | stream-verified | 374 | 44 | 485,881 | 21,378,764 | ordered_matmul + RoPE RNE formula | exact (padding: 656 words +0/-0) | exact | 0.99773 | 6.752% | `model.layers.*.self_attn.k_proj` |
| `projection_gemm` | 32x2048x256 (logical 1x2048x256) | bfloat16 | tiny_breadth recipe | stream-verified | 116 | 44 | 459,785 | 20,230,540 | ordered_matmul | exact | exact | 0.99804 | 6.251% | `model.layers.*.self_attn.v_proj` |
| `attention` | 1x129 32/4 d64 causal FP8 out | float32 | attention v2 packed decode, 2 unique mask tiles | stream-verified | 468 | 22 | 197,542 | 4,345,924 | none | n/a | exact | 0.99823 | 5.956% | `model.layers.*.self_attn` |
| `attention` | 1x130 32/4 d64 causal FP8 out | float32 | attention v2 packed decode, 2 unique mask tiles | stream-verified | 468 | 22 | 197,542 | 4,345,924 | none | n/a | exact | 0.99803 | 6.270% | `model.layers.*.self_attn` |
| `projection_gemm` | 128x2048x1280 | bfloat16 | tiny recipe | stream-verified | 2048 | 1 | 4,222,485 | 4,222,485 | ordered_matmul | exact | exact | 0.99796 | 6.382% | `lm_head` |
| `projection_gemm` | 32x2048x1280 (logical 1x2048x1280) | bfloat16 | tiny_breadth recipe | stream-verified | 116 | 2 | 2,103,945 | 4,207,890 | ordered_matmul | exact | exact | 0.99809 | 6.181% | `lm_head` |

`words` is the executed program size in 32-bit instruction words (after the arena relocation the bench applies). Bit-exact covers the logical rows and columns the model consumes; a padding note counts output words in the zero-padded rows or columns that differ from the golden (`+0/-0`: only the sign of zero differs).

## Unmapped ops (0 rows, 0 instances, 0 FLOPs)

None: every exported op has a placement.

## Notes

Launches come from `coverage/coverage.json` and the `launch` hints in `coverage/mapping.yaml`: projection calls split into physical panels by `bench.tinyllama` (128 rows, N in 2,048-column chunks, 1,024 when K is 5,632), which reproduces the kit's 840 panels; gate and up panels bench as their fused SiLU and multiply epilogues, q and k panels with the RoPE post-pass, and prefill o and down panels with the exact residual add, as the qualified build launches them.

The 9 TinyLlama shapes that `lib/kernel_bench.py` also benches (`coverage/RESULTS.md`) give identical model cycles here.

Decode RoPE launches (32 physical rows, 1 logical row) match the golden on the logical row; in the 31 zero-padded rows the device writes -0 where the golden RNE formula gives +0 (656 and 5,248 words). The kit discards those rows. Bit-exact is therefore reported over the logical extent, with the padding words counted.

RoPE tables use positions 0 to M-1 for every launch (decode uses positions 128 and 129 in the model); attention uses the generator's causal mask with the model's query start. Table values are arena data; matching launch geometry alone does not establish literal word equality with address-specific kit programs.

Functional and timed results are reused from Track G only after regenerating each job with the pinned Track G staging code and this bench, proving equal program-word and staged-arena SHA-256 digests, and passing the compiler word check; the per-job proof is in `reuse-proof.json` beside the generated JSON. No functional or timed job changed, including the row-subset estimate.

This bench's 3,537,507,930 npu_model model cycles and the 3,528,847,056 recorded by the baseline F3-h0 kit (`F/f3h0/results.json`, cited by atlas-mlir `tools/tiny_full/REPRODUCING.md`) are canonical for different programs and timing conventions. The baseline performs decode RoPE on the host; the current kit overlay and this bench fuse it into 88 decode q/k launches. In the historical timing convention, that adds 44 x 171,052 + 44 x 26,096 = 8,674,512 cycles.

The historical cycle runner replaces the final ECALL with a cycle-counter sequence, adding 5 cycles per launch. This bench times the unmodified ECALL, which also cancels the remaining 128 cycles of the terminal `delay 128` in all 66 attention launches on npu_model 47b8599. The exact reconciliation is 3,537,507,930 - 3,528,847,056 = 8,674,512 - 972 x 5 - 66 x 133 = 8,660,874 cycles. The current overlay in the historical timing convention would therefore be 3,537,521,568 cycles, a derived value, not an independently timed full-overlay result.

All 66 attention masks, descriptors, key counts, and layout offsets match the kit controls byte for byte. RoPE positions and synthetic values differ, but the variant and terminal-timing changes account for the complete difference, leaving no unexplained data-dependent cycle term. The reconciliation uses per-shape historical receipts, normalized schedule joins for 1,038 launches, and four scalar timing probes; it does not claim literal word equality between these address-specific kit and bench programs.
