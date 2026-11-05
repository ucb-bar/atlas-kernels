# Opt-in streamed SmolVLA projection

`atlas_program_smol_projection_fast.assembly()` emits a plain projection for the eight physical shapes in `SHAPES`. `optimize(source, m, k, n)` accepts the exact original traffic-reuse assembly or its established SmolVLA low-word relocation. It rejects hooks, post-passes, changed instructions, and already transformed programs. Default generators, the earlier GEMM variant, and shipped kit assets retain their existing behavior.

```sh
python kernels/projection_gemm/gen_smol_fast.py \
  --program original.S --out streamed.S --m 128 --k 768 --n 768
```

The first row pair starts the next raw B tile DMA before transposing and computing the current tile. Each request targets a disjoint VMEM tile and is waited before that tile is read. The last tile issues no speculative request. A 128-row launch caches transposed B tiles for its second pair. Both rows share a weight push, use separate operand registers, and preserve increasing-K accumulation, the sole BF16 pop, and the original power-of-two rescale. Every column ends with all fixed-latency work and DMA drained.

A occupies bank 1, then spills into bank 0 after the B column and 2,048-byte output scratch. The largest admitted shape, 128x3072x768, uses B bytes [0,98304), output scratch [98304,100352), A spill [100352,231424), and A bank 1 [262144,524288). Other admitted shapes need no A spill. The workspace fits the same two physical banks as the original schedules. Kit adoption requires explicit selection and regeneration of program-loading metadata and compiler receipts.

The earlier paired-row variant admits 128x768x768, 128x3072x768, 64x736x2048, and 64x2048x736. It does not admit 64x736x960, 64x960x736, 128x960x2560, or 64x960x2560. Its weight-column DMA and transpose phase finishes before matrix computation begins. The new schedule overlaps these phases while retaining the same total DRAM traffic.

All 33 distinct captured programs pass numerical execution on unmodified npu_model `47b8599`, `numerics="rtl"`, with native acceleration disabled. Each original and candidate runs on one captured input, giving 66 selected executions with full-arena identity, including padding and guards. This represents 1,840 launches, rather than every represented launch's input. All 33 candidates are compiler tier `verified` through the supplied INTEG2 Release tools, with emitted words equal to the assembler. The complete unit gate passes 263 invocations across 10 suites with 0 skips, and the separate citation check passes.

| Physical MxKxN | Programs | Count | Original | KGEMM or original | New | Saved x count | Incremental gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x736x960 | 1 | 160 | 917,415 | 917,415 | 680,175 | 37,958,400 | 25.86% |
| 64x736x2048 | 1 | 320 | 1,916,097 | 1,724,737 | 1,409,985 | 161,955,840 | 18.25% |
| 64x960x736 | 4 | 160 | 905,046 | 905,046 | 666,651 | 38,143,200 | 26.34% |
| 64x960x2560 | 6 | 32 | 3,030,974 | 3,030,974 | 2,201,774 | 26,534,400 | 27.36% |
| 64x2048x736 | 2 | 160 | 1,843,626 | 1,652,266 | 1,330,726 | 82,064,000 | 19.46% |
| 128x768x768 | 6 | 784 | 1,077,912 | 856,728 | 774,450 | 237,914,208 | 9.60% |
| 128x960x2560 | 6 | 32 | 4,163,274 | 4,163,274 | 2,871,994 | 41,320,960 | 31.02% |
| 128x3072x768 | 7 | 192 | 3,848,658 | 2,963,922 | 2,612,347 | 237,371,712 | 11.86% |

Cycle counts are exact; displayed percentages are rounded to two decimal places.

The new variant saves 863,262,720 cycles against the original target launches (28.64%). Against KGEMM where admitted and the original elsewhere, it additionally saves 428,132,352 cycles (16.60%). Every shape exceeds 5% incremental gain, so the variant is retained. The original projection family total of 3,424,717,132 becomes a derived 2,561,454,412. Applying only this variant to the fresh SmolVLA baseline of 5,160,679,740 gives 4,297,417,020 device cycles. These are measured representative launch deltas multiplied by published counts. No whole-model execution, host time, program-loading time, or new accuracy evaluation is claimed. The fresh baseline measures the full vision launch and does not use the earlier vision estimate.

Pinned matched-memory RTL confirms 128x768x768: 1,077,911 original cycles and 774,449 new cycles, saving 303,462. All 196,608 output bytes match the captured reference. The new live replay and exact authenticated earlier original receipt use the same pinned simulator and memory configuration. Both complete memory request/response traces validate against the unchanged model backend. Core counts are one cycle higher under its startup/halt convention.

The 128x768x768 candidate trace measures 92,160 cycles with DMA and MXU0 simultaneously active. Its DRAM traffic is unchanged at 688,128 load bytes and 196,608 store bytes. In addition to overlap, the first pair avoids reloading its transposed B tile from VMEM. Consequently, the published original SOL provides context, rather than a newly measured candidate bound. Timing traces check output bytes and drained fixed-latency engines; functional executions check complete arenas. Model package bytes are authenticated against the read-only pinned checkout, with later traces additionally recording in-process model bookends.

The [machine-readable results](../../coverage/smol-projection-results.json) retain all per-shape values, compiler joins, tool identities, and receipt hashes. Local full receipts are [/scratch/reednicolas/atlas-tmp/jobs/K2/KOPT2/out/RESULTS.json](/scratch/reednicolas/atlas-tmp/jobs/K2/KOPT2/out/RESULTS.json), [/scratch/reednicolas/atlas-tmp/jobs/K2/KOPT2/out/qualification/manifest.json](/scratch/reednicolas/atlas-tmp/jobs/K2/KOPT2/out/qualification/manifest.json), and [/scratch/reednicolas/atlas-tmp/jobs/K2/KOPT2/out/rtl-final/new/result.json](/scratch/reednicolas/atlas-tmp/jobs/K2/KOPT2/out/rtl-final/new/result.json). Exact earlier timing and original RTL reuse are explicitly recorded in `out/reused-timing.json` and `out/rtl-final/reused-old.json`.

All default executable sources and kit assets remain byte-identical to `local/bench-fresh` at `de45370`. Only the RTL result documentation is extended among baseline files. The 16 legacy ELF rebuilds, global 1,589-program word gate, and full compiler test suite are PENDING. Explicit adoption requires regenerated programs, loading metadata, source pins, and execution receipts. The largest candidate has 4,173 words after relocation, within the pinned RTL's 32,768-word instruction memory.
