# Opt-in TinyLlama fused streaming

`atlas_program_tiny_fused_fast.optimize(source, m, k, n)` selects a new streamed producer for the captured TinyLlama fused launches. `gen_tiny_fast.py` exposes the same strict source transform. Existing generators, pinned modules, and kit outputs are unchanged.

The admitted physical shapes are 128x2048x2048 with multiply/FP8, SiLU, exact residual add, or RoPE; and 128x5632x1024 with exact residual add. This variant does not admit the 32-row decode shapes, 1536-wide epilogue shards, or 256-wide K projection.

KGEMM overlaps weight pushes, operand loads, and MXU work, but its Tiny 128-row core finishes B-column loading and transposition before computing the first row pair. The original 32-row core prefetches four-tile B blocks with conservative transpose and compute spacing. KOPT2 streams B through the first row pair for other shapes and plain projection outputs. Extending that implementation requires full-A down-projection residency, exact fused-hook ownership, sharded FP8 consumer placement, and RoPE's retained output.

The new core prefetches the next B tile while the first row pair transposes and computes the current tile. It caches transposed B only when later row pairs need it. The 128-row down projection retains all A in banks 1 through 3 and loads every B tile once. Its original 12,517,376 load bytes become 6,750,208, including the residual operand, matching KGEMM's traffic reduction. Other target load and store traffic is unchanged.

Increasing-K products, one BF16 accumulator pop, rescaling, exact residual rounding, SiLU tables, FP8 clipping and packing, and RoPE arithmetic are preserved. Hooks run after the final compute drains. No core value in m6 survives across a hook. Each incoming DMA targets a disjoint B tile and is waited before its first read; the final tile has no speculative successor.

Selection compares the entire original source against a reconstructed family template, including immediates, delays, scales, operands, tables, and output layout. Multiply accepts the captured 5632-wide consumer stride of 180,224 bytes, shard columns 0 and 64, and the captured output scale. The established 0x90000000 relocation is supported. Modified sources, custom hooks, unsupported shapes, plain projections, and repeated transforms are rejected.

The largest candidate is 7,993 words, within the 32,768-word IMEM. The largest VMEM end address, including the preserved RoPE staging area, is 1,314,816 bytes within 1,572,864 bytes. The larger unrolled program and changed residency require explicit selection and regenerated words, loading metadata, compiler receipts, and execution receipts.

```sh
python kernels/projection_gemm/gen_tiny_fast.py \
  --program captured-original.S --m 128 --k 2048 --n 2048 \
  --out candidate.S
```

`describe()` reports the admitted family, scales, shard placement, output extent, and actual DMA bytes. `layout()` reports the core residency and BF16 scratch geometry. Full captured arenas, including FP8 stride holes and guards, remain part of numerical qualification.

The candidate still transfers every B byte once through the configured memory curve. A preload, fused hook loads, output store waits, later row-pair computation, and the separate RoPE post-pass retain serialization. No timing-ledger percentage is assigned to those remaining costs. The earlier roofline bounds describe original work; the down-projection traffic reduction changes that work.

The experimental 32-row path was rejected. One-KiB streaming regressed multiply from 3,420,329 to 3,448,418 cycles. A legal 4-KiB block-prefetch probe measured 3,392,418 cycles, saving 27,911 cycles, or 0.816%, below the 5% retention threshold. Larger DMA probes are invalid because the pinned model limits one command to 4,096 bytes. The measured 8-KiB command failed immediately; the 16-KiB candidate was not run. The 32-row defaults remain selected, and their attempted sources, numerical results, and timing receipts are retained outside the repository.

All three legal 4-KiB 32-row probes passed full-arena numerical and engine-drain checks. Their measured gains remain below 1%:

| Physical shape | Hook | Original | Legal 4-KiB probe | Saved | Gain |
| --- | --- | ---: | ---: | ---: | ---: |
| 32x2048x2048 | multiply, FP8 out | 3,420,329 | 3,392,418 | 27,911 | 0.816% |
| 32x2048x2048 | SiLU | 3,376,233 | 3,348,322 | 27,911 | 0.827% |
| 32x2048x2048 | RoPE | 3,508,117 | 3,480,206 | 27,911 | 0.796% |

## Captured TinyLlama results

| Family and hook | Physical MxKxN | Programs | Count | Original cycles | KGEMM or original | Selected cycles | Saved cycles x count | Incremental saved cycles x count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epilogue mul:fp8 (rejected) | 32x2048x2048 | 39 | 88 | 3,420,329 | 3,420,329 | 3,420,329 | 0 | 0 |
| epilogue silu:bf16 (rejected) | 32x2048x2048 | 5 | 88 | 3,376,233 | 3,376,233 | 3,376,233 | 0 | 0 |
| epilogue mul:fp8 (retained) | 128x2048x2048 | 26 | 44 | 6,977,261 | 5,404,397 | 4,788,338 | 96,312,612 | 27,106,596 |
| epilogue silu:bf16 (retained) | 128x2048x2048 | 5 | 44 | 6,792,045 | 5,219,181 | 4,603,122 | 96,312,612 | 27,106,596 |
| rne_add add_rne:bf16 (retained) | 128x2048x2048 | 6 | 22 | 7,779,565 | 6,206,701 | 5,590,642 | 48,156,306 | 13,553,298 |
| rne_add add_rne:bf16 (retained) | 128x5632x1024 | 9 | 44 | 9,979,346 | 7,420,776 | 6,575,973 | 149,748,412 | 37,171,332 |
| rope rope (rejected) | 32x2048x2048 | 3 | 44 | 3,508,117 | 3,508,117 | 3,508,117 | 0 | 0 |
| rope rope (retained) | 128x2048x2048 | 3 | 22 | 7,318,670 | 5,745,806 | 5,129,747 | 48,156,306 | 13,553,298 |

Across the eight requested groups, original cycles x count are 2,129,556,462, KGEMM-or-original cycles x count are 1,809,361,334, and selected cycles x count are 1,690,870,214. Savings are 438,686,248 against the original and 118,491,120 incremental to KGEMM. Subtracting the original savings from the cited 3,537,507,930-cycle TinyLlama device baseline derives 3,098,821,682 cycles. These totals multiply measured launch cycles by captured counts and do not claim a whole-model, host, or program-loading run.

The five retained groups cover all 49 distinct address-specific kit programs and 176 captured launch instances. The public selector rejects all 47 distinct 32-row programs representing 220 launches. Rejected 32-row groups retain original cycles. Every retained program passes bit-exact original/candidate comparison on its captured input with unmodified npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`, `numerics="rtl"`, and native acceleration disabled. Complete arenas include padding, guards, and FP8 stride holes. The 47 rejected experimental candidates also pass, giving 192 authenticated original/candidate numerical receipts over all 96 programs.

All 96 original and 96 attempted programs pass `lib/atlas_backend.py` tier `verified`, both Atlas verifiers, and emitted-word equality to the assembler. The compiler uses the INTEG2 Release binaries identified by the receipt digests.

Pinned RTL `0079c0541111197741a231c002e3843fa6f545b2` confirms the complete retained 128x2048x2048 SiLU candidate at 4,603,121 execution cycles with matched memory. All 524,288 output bytes and the full 4,981,248-byte arena match the captured reference. The complete memory request/response trace is validated against the unchanged model backend. The optional original replay was cancelled, so original RTL confirmation remains PENDING; before/after performance uses the authenticated model timing receipts. See [RTL results](../../coverage/RTL-RESULTS.md#opt-in-tinyllama-fused-prefill) for provenance and scope. Other retained shapes remain PENDING for RTL.

Resumed `make test` passes 271 invocations across 10 suites with 0 skips under the Track H recipe and INTEG2 Release tools. The explicit citation check also passes. The prior final run passed 271 invocations with 0 skips. Independent review found no blocking source defects.

The [compact result manifest](../../coverage/tiny-fused-results.json) joins measured facts and receipt digests. Raw assembly, arenas, binary programs, logs, and detailed joins remain in the KOPT3 work-area `out/` directory. Default generators and pinned source modules remain byte-identical to local base `2568421`.
