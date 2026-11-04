# Opt-in vision attention schedule

Select `--variant vision-fast` in `gen.py`, or call `atlas_attention_vision_fast.assembly_v2` with the original `AttentionV2Config`, `stage_v2` offsets, and staged arena. This schedule accepts unmasked whole tiles, equal query and KV head counts, and BF16 output. It rejects padding, masks, grouped heads, packed decode, and FP8 output. The original v2 module, default selection, and shipped kits are unchanged.

The wrapper preserves every arithmetic operation and its order, including the BF16 score scaling, row maximum, BF16 exponential, FP8 probability conversion, MXU denominator accumulation, reciprocal, PV accumulation, and final BF16 conversion. It shortens MXU1 compute waits from `delay 96` to `delay 34`, overlaps each first QK/PV compute with the next independent weight load, and hoists the immutable -128 and 256 constants into unused logical pairs. The fixed key count and zero mask descriptors expose the supported domain to the timing verifier. Explicit DMA waits remain at completion; the dead terminal delay is removed.

## Why the original is far above its MXU bound

The target has 384 query tiles and 12,288 query/key tile pairs. Its QK, quantized denominator, and PV execute 24,576, 12,288, and 24,576 matmuls, respectively. The denominator is part of the exact arithmetic. With 32-cycle compute feed occupancy from pinned npu_model `47b8599` (`npu_model/hardware/mxu.py:19`), the 61,440 matmuls give a 1,966,080-cycle compute-feed bound. Preserving the 13,440 BF16 accumulator clears strengthens the MXU1 read-port-0 bound to 2,396,160 cycles. Neither bound includes vector work, conversions, memory transfers, or scalar overhead.

MXU1 writes its last compute row at age 34, while MXU0 does so at age 94. Original v2 nevertheless uses a 96-cycle delay for MXU1, giving a 98-cycle issue gap. The new 34-cycle delay gives a 36-cycle gap. The next independent weight load hides the first compute of each QK/PV pair without changing weight contents during a compute.

The reduced 32x1024, 1/1-head trace measures 260,590 original cycles: 114,888 explicit delay holds, 139,058 DMA.WAIT holds, 6,642 issue cycles, one startup cycle, and one halt cycle. Vision-fast measures 244,213 cycles: 98,700 delay holds, the same 139,058 DMA.WAIT holds, 6,453 issue cycles, one startup cycle, and one halt cycle. The outputs match byte for byte. Reduced-case KV staging dominates and is amortized over more query tiles in the full launch. The retained matched-memory scalar issue analysis proves scalar issue agreement and identifies the default curve memory backend; this work uses those facts rather than requalifying them.

## Qualification

`make test` passes 166 test invocations across eight suites with zero skips, including rejection checks and bit-exact functional checks with zero and nonzero FP8 scale exponents. Two default CLI configurations retain byte-identical assembly, binary, and JSON outputs against `6123bef`.

Full-shape captured-input qualification passes for all five distinct kit programs, representing 24 launches. Both schedules ran to ECALL on each program's original captured input with unmodified npu_model `47b8599`, RTL numerics, seed 42, the default curve memory, and a 14,400-second timeout. Every old/new full arena is byte-identical, matches the captured output, preserves inputs and guards, transfers exactly 4,726,912 load bytes and 1,572,864 store bytes, and drains all engines and DMA flags. No subset estimate is used.

| Configuration | Distinct programs | Launches | Original cycles | Vision-fast cycles | Saved cycles per launch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024x1024, 12/12 heads, d64, none, BF16 output | 5 | 24 | 26,092,660 | 19,752,187 | 6,340,473 |

All five programs have identical timing within each schedule. The reduction is 24.2998337463 percent. Original/new completed instruction counts are 1,443,842 and 1,370,117; program sizes are 420 and 417 words. The 24-launch subtotal falls from 626,223,840 to 474,052,488 cycles, saving 152,171,352 cycles. These are raw-ECALL model cycles, rather than the kits' completion-marker timing convention.

Replacing only the 24 estimated launches in the frozen model bench gives a corrected original SmolVLA device total of 5,160,679,740 cycles. Selecting vision-fast for those launches projects 5,008,508,388 cycles with the other published counts unchanged. These totals are arithmetic replacements, not a new whole-model execution. The estimate correction reduces the reported total by 399,954,144 cycles before the separate measured optimization saving of 152,171,352 cycles.

All five new full-shape programs pass `lib/atlas_backend.py` with atlas-emit words equal to the assembler and timing tier `verified`, using atlas-mlir `abb67c1`. Each full program has 417 words. The pinned RTL runner from atlas-kernels `4af4ce0` confirms the explicitly reduced 32x1024, 1/1-head shape with matched model memory: 260,589 original cycles and 244,212 new cycles, identical 4,096 output bytes, and 8,457 memory requests and responses validated for each. Each RTL count is exactly one startup cycle below the corresponding model count. Full 1024x1024, 12/12-head RTL qualification is PENDING.

An additional 64x1024, 1/1-head matched-memory RTL test covers query prefetch and buffer swapping: 322,478 original cycles and 289,589 new cycles, with all 8,192 output bytes identical and 8,721 memory requests and responses validated for each. The corresponding model counts are 322,479 and 289,590. All 128 second-query read beats and responses complete before the first output store in both runs. This remains a reduction from the full target shape.

The frozen [SmolVLA model bench](../../coverage/examples/smolvla/MODEL-BENCH.md) at `6123bef` reports 42,757,416 estimated cycles per vision launch and 5,560,633,884 device cycles in total, including 24 estimated launches. Its estimate scales a 128-query measurement by eight, repeating the once-per-launch KV staging cost. Correcting that estimate and measuring the new schedule are separate results.

Adoption in model kits requires explicit selection, new authenticated program assets, updated source pins, and host/runtime requalification. The five distinct-program checks use one captured input per program, not every input from the 24 launches. Existing kit assets and ELFs are untouched. A fresh 16-ELF rebuild and the kit-wide 1,589-program gate are PENDING for this follow-up.
