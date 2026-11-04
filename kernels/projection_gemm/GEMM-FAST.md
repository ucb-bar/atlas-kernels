# Opt-in paired-row GEMM core

`atlas_program_gemm_fast.assembly()` selects an overlapping operand schedule for the seven physical shapes in `SHAPES`. The 128x5632x1024 case also keeps A in VMEM banks 1 through 3 and loads each B column once through `atlas_program_gemm_fast_breadth`. The default dispatcher, original generator modules, and shipped kit assets are unchanged. Other shapes, including logical one-row decode launches, are rejected by this entry point and continue using their existing schedules.

For a plain projection or an output hook, replace the explicit call to `atlas_program.assembly(..., traffic_reuse=True, epilogue=hook)` with `atlas_program_gemm_fast.assembly(..., epilogue=hook)`. Generate RoPE through the same hook and append its original post-pass. For an already generated family program, including a relocated SmolVLA fusion program, call `atlas_program_gemm_fast.optimize(source, m, k, n)`. The standalone selector is:

```sh
python kernels/projection_gemm/gen_fast.py \
  --program original.S --out fast.S --m 128 --k 768 --n 768
```

The shared path starts in `atlas_program.assembly()`, which selects the `resident`, `tiny`, `stream`, `tiny_breadth`, or `blocked` recipe from the compact projection catalog. The `epilogue` family uses an `Epilogue` output hook, and the fused `rne_add` launches use `epilogue/gen.py --steps add_rne` (the standalone `rne_add/gen.py` emits only the addition); `rope` retains output and appends `atlas_rope_postpass`; `smol_fusion` stages its hook through `smol_fusion_fp8`. This variant reuses those original layouts and hook generators.

The variant preserves input values, accumulator operations, rescaling, output conversion, output stores, and increasing-K reduction order. Six shapes preserve their original DMA and transpose sequences; the full-A down projection removes its duplicate B loads and transposes. A weight push overlaps the following first-row operand load. The second row loads into `m6` while the first compute still reads `m0`, then computes after the original load spacing. Hooks may reuse `m6` only after a full restoring write; admission resets that proof at every compute and control-flow edge. CFG boundaries retain the original drain delays. All output hooks execute after the operand streams have drained.

The timing argument uses the pinned RTL-model row schedules: a load retains `delay 34`, the two computes are separated by at least 36 cycles, and the next same-slot weight push occurs beyond 62 cycles after the preceding compute. The first accumulator pop occurs after its row-zero writeback; the second pop follows completion of the first output hook and store. These are software scheduling constraints, independent of the memory curve.

The full-shape 128x768x768 diagnostic trace executes in 1,077,143 original cycles and 855,959 fast cycles, saving 221,184 cycles. Both runs produce identical full 885,248-byte arenas. Original and fast ledgers contain 579,028 DMA.WAIT holds; DELAY holds fall from 468,930 to 250,050, and issued instructions fall from 29,183 to 26,879. Each has 2 startup/halt cycles. This diagnostic uses the original 4-GiB DMA base and an exponent of 0, so its totals differ from the relocated canonical bench; the measured schedule delta is the same.

The trace loads 688,128 bytes and stores 196,608 bytes, using channels 0 and 1, with at most one DMA command active and no DMA/MXU overlap. A and B each load once for this shape. The 96 output stores remain serialized. The curve pacing sum for its 27,648 beats is 638,012.3314750849 cycles, an idealized bound excluding command and VMEM arbitration costs; the original run is about 1.69 times that bound. DMA occupies 656,308 cycles and incurs 12,144 denied VMEM writes. Extra channel capacity alone does not remove the tensor serialization measured here. The compute-only down candidate measured 9,954,514 cycles against the cited 9,979,346 baseline, saving only 24,832 cycles (0.25% rounded), so it is excluded from the selected variant. The original 128x5632x1024 recipe streamed weights twice, once per row pair. Its full-A variant loads each weight column once, removing 5,767,168 load bytes per launch. Its residual-add launch requests 6,750,208 load bytes and 262,144 store bytes. Resident output-hook subclasses are excluded because they could overlap the enlarged A region.

The established memory and issue relationship is documented in the K2 GAP report: npu_model and AtlasCore issue timing agree under matched memory, and the default model memory is the curve backend. This variant measures software scheduling gains under that memory service, rather than a change to the memory model.

Pinned AtlasCore RTL at `0079c0541111197741a231c002e3843fa6f545b2` confirms the full captured 128x768x768 launch: 1,077,911 original cycles and 856,727 fast cycles, saving 221,184 cycles. All 196,608 output bytes match the captured reference. Both runs validate all 27,648 memory requests and responses against the unchanged model backend, with peak pending depth 63. They reuse the pinned standalone harness from `followup/rtl-results` at `4af4ce0`, with matched memory; no fresh RTL build is claimed. Raw-ECALL Core counts include one additional startup/halt cycle in both runs. Functional and RTL instruction counts include ECALL and exceed Core's completed-instruction count by one; the receipts join these conventions explicitly.

All per-launch counts below are measured on captured inputs with unmodified npu_model `47b8599`, `numerics="rtl"`, default curve memory, and raw ECALL. Every original representative reproduces its cited `6123bef` baseline, and every selected fast representative matches the captured full arena and finishes with all engines and DMA flags drained. Counts and canonical totals come from the frozen [TinyLlama bench](../../coverage/examples/tinyllama/MODEL-BENCH.md) and [SmolVLA bench](../../coverage/examples/smolvla/MODEL-BENCH.md).

| Model | Family / hook | MxKxN | Count | Before | After | Saved x count | Gain |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| tinyllama | epilogue mul:fp8 | 128x2048x2048 | 44 | 6,977,261 | 5,404,397 | 69,206,016 | 22.54% |
| tinyllama | epilogue silu:bf16 | 128x2048x2048 | 44 | 6,792,045 | 5,219,181 | 69,206,016 | 23.16% |
| tinyllama | rne_add add_rne:bf16 | 128x5632x1024 | 44 | 9,979,346 | 7,420,776 | 112,577,080 | 25.64% |
| tinyllama | rne_add add_rne:bf16 | 128x2048x2048 | 22 | 7,779,565 | 6,206,701 | 34,603,008 | 20.22% |
| tinyllama | rope rope | 128x2048x2048 | 22 | 7,318,670 | 5,745,806 | 34,603,008 | 21.49% |
| smolvla | projection_gemm plain | 128x768x768 | 784 | 1,077,912 | 856,728 | 173,408,256 | 20.52% |
| smolvla | projection_gemm plain | 128x3072x768 | 192 | 3,848,658 | 2,963,922 | 169,869,312 | 22.99% |
| smolvla | projection_gemm plain | 64x736x2048 | 320 | 1,916,097 | 1,724,737 | 61,235,200 | 9.99% |
| smolvla | projection_gemm plain | 64x2048x736 | 160 | 1,843,626 | 1,652,266 | 30,617,600 | 10.38% |
| smolvla | smol_fusion bias+gelu_tanh:fp8 | 128x768x3072 | 192 | 4,956,786 | 4,357,746 | 115,015,680 | 12.09% |

Applying these measured savings to the requested launch counts saves 320,195,128 TinyLlama cycles and 550,146,048 SmolVLA cycles. The derived device totals are 3,217,312,802 versus 3,537,507,930 for TinyLlama (9.05% reduction), and 5,010,487,836 versus 5,560,633,884 for SmolVLA (9.89% reduction). SmolVLA retains the baseline's 24 estimated vision-attention launches. These totals exclude host time, program loading, and transfers between launches; no whole-model execution is claimed.

The same 128x2048x2048 core also applies to 15 captured TinyLlama `lm_head` panels outside the requested aggregate. Its measured representative is 6,635,373 to 5,062,509 cycles, saving another 23,592,960 cycles across those panels. Including them gives a derived TinyLlama total of 3,193,719,842 cycles. The remaining 1280-column panel and logical one-row decode shapes retain their existing implementations.

Full-A residency increases the captured residual-add down program from 354 to 5,139 words and uses A bytes [262144,983040), B bytes [0,180224), and output scratch [188416,190464). Kit adoption must reserve this workspace and regenerate program-loading metadata. The diagnostic rejected compute-only down result is retained as analysis, and the selected public entry point always uses full-A residency for this shape.

The original `5ca81ab` follow-up qualification passed 170 unit-test invocations across 7 suites with 0 skips. All 76 selected programs have verified compiler receipts and atlas-emit/assembler word equality through the exact `abb67c1` timing-verifier build. Numerical replay passes all 152 executions: 76 original programs and 76 selected fast programs, with full DRAM arenas identical to the captured references, including output padding and input guards. Every program uses unmodified npu_model `47b8599` with `numerics="rtl"` and native acceleration disabled. The final joins authenticate source, program, input, output, and compiler hashes; all 22 selected Core executions also join their functional results.

Coverage comprises 74 distinct requested programs and 2 additional LM-head programs, representing 1,824 requested launches and 15 additional panels. Each program runs on one captured input. Replays do not cover every represented launch's input. The preserved arithmetic and reduction order, supported by these bit-identical checks, retain the existing accuracy results; no new accuracy evaluation is claimed.

The 16 legacy ELF rebuilds, global 1,589-program word gate, and full compiler test suite are PENDING for this branch. In the original `5ca81ab` follow-up, all 123 baseline repository files at `6123bef` were byte-identical, including the default generators and shipped kit metadata. The exact timing-verifier tools were built for this job, and their focused CLI and lit checks pass.

Kit adoption requires explicit variant selection and regenerated program words, source pins, program hashes, compiler receipts, and execution receipts. The original kit metadata does not authenticate the new variant. No host graph or model accuracy result was regenerated by selecting this module.
