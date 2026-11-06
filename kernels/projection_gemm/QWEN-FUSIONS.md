# Opt-in Qwen exact projection additions

`atlas_program_qwen_fused.assembly(role, m, k, n, rescale_exponent)` adds exact BF16 round-to-nearest-even bias for Q/K/V or residual addition for o/down. The explicit roles `q`, `k`, `v`, and `o` admit only physical 32x1536x128; `down` admits only 32x8960x128. The logits role is excluded. Logical decode rows remain padded to 32 physical rows. This API adds an isolated candidate and does not connect a Qwen host graph.

The selected K1536 producer retains the existing `resident_a=True` schedule; K8960 retains the existing base schedule. Increasing-K FP8 products, ordered BF16 accumulation, the single accumulator pop, and the explicit power-of-two BF16 rescale precede the exact addition. The result matches the Atlas projection followed by host FP32 addition and BF16 RNE. This is an exact addition claim relative to the Atlas projection, rather than a claim that FP8 projection equals an unquantized PyTorch projection.

Q/K/V use one BF16 channel broadcast tile per 32 output columns. `stage()` accepts a uint16 vector of length 128 and expands it to the four broadcast tiles. o/down accept a uint16 residual panel with all 32x128 physical entries. Activations and weights are uint8 FP8 matrices. Callers initialize activation padding explicitly; bias also applies to padded rows, and residual padding is supplied explicitly. The entire physical output, inputs, and guards are checked in qualification.

The arena contains A, B, a 256-byte guard, an 8192-byte BF16 output, a 256-byte guard, an 8192-byte staged addend, and a final 256-byte guard. `layout()` returns every offset, the arena extent, the selected residency, and exact requested DMA bytes. The fused hook loads 8192 additional bytes and retains the producer's 8192 output store bytes. It uses the producer's BF16 output scratch and changes no weight or activation layout.

`ChannelRNE` validates its channel extent through the existing epilogue contract and its exact-add requirements through a tiled surrogate. It retains an `add_rne` step, so the unchanged `checked_assembly` wrapper checks the actual generated hook's matrix-register footprint. The channel address and exact-add instructions come from the existing epilogue implementation. Scalar scratch x27/x28 remains disjoint from the original producer, and x31 remains available for arena relocation. Default channel `add` remains truncating and is unchanged.

```sh
python kernels/projection_gemm/gen_qwen_fused.py \
  --role q --m 32 --k 1536 --n 128 --rescale-exponent -14 \
  --out candidate.S

python kernels/projection_gemm/qualify_qwen_fusions.py \
  --model-root /path/to/npu-model --atlas-bin /path/to/compiler/bin \
  --out /path/to/qualification --timed
```

Qualification covers one shared Q/K/V bias kernel plus o and down residual kernels, each with 1 and 32 logical rows. Every case uses its full physical K1536 or K8960 dimension and nonzero rescale. The independent golden uses ordered FP8/BF16 matrix multiplication and rescaling, then separately embeds BF16 addends exactly into FP32, adds, and performs integer BF16 RNE with canonical NaNs. It imports no device exact-add arithmetic.

Each selected plain projection, separate full-panel RNE add, and fused projection must pass complete-arena comparison on unmodified npu_model `47b8599`, `numerics="rtl"`, compiler tier `verified`, and emitted-word equality to the model assembler. `--timed` additionally executes all three complete launches on the Core cycle model and compares complete timed arenas with their functional results. The summary records plain cycles, fused incremental cycles, and the fused saving against the sum of plain plus separate-add cycles. Raw assembly, binaries, model identities, compiler records, and individual results remain under the explicit output directory.

The separate-add baseline processes all 4096 physical BF16 elements, including decode padding. A compact 128-element decode add is a distinct alternative that requires extracting the logical row; extraction and transport remain unmeasured. Neither timing baseline includes host staging, program loading, checking, whole-model accuracy, or connected-session latency. No net host savings are claimed. All existing default generators, pinned source modules, kit assets, and census outputs remain unchanged.

All 6 cases pass independent numerical, compiler, and timing qualification. Each mode checks 49152 output bytes across the 6 fixtures in functional execution and again in timed execution. All 18 functional and 18 timed executions check their complete input and guard arenas, including padding. Fused outputs cover 24576 physical BF16 words, with 12672 logical words across the 6 fixtures. The fixtures differ from truncating addition in 32 and 1000 words for decode and prefill Q/K/V, 32 and 1088 for o, and 24 and 1158 for down. All scales are nonzero. All programs reach compiler tier `verified`, and all emitted words equal the model assembler.

| Role | Physical MxKxN | Logical rows | Plain cycles | Separate full-panel add | Plain plus add | Fused cycles | Fused increment over plain | Saved versus padded split |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q/K/V bias | 32x1536x128 | 1 and 32 | 246405 | 28617 | 275022 | 264277 | 17872 | 10745 |
| o residual | 32x1536x128 | 1 and 32 | 246405 | 28617 | 275022 | 264265 | 17860 | 10757 |
| down residual | 32x8960x128 | 1 and 32 | 2056340 | 28617 | 2084957 | 2074200 | 17860 | 10757 |

The table reports measured complete Core launches for both logical-row cases. The largest fused program has 5754 words, within the 32768-word IMEM. K1536 fused DMA is 253952 load bytes and 8192 store bytes, compared with 245760 and 8192 for plain. K8960 fused DMA is 2301952 and 8192, compared with 2293760 and 8192 for plain. The separate full-panel add transfers 16384 load bytes and 8192 store bytes. All requested byte counts equal observed counts.

A separately qualified contiguous 128-element decode add, padded to 1024 elements, costs 7594 cycles. That is 10278 fewer cycles than the Q/K/V fused increment and 10266 fewer than the o/down increment. Its independent synthetic input does not include projection-row extraction or transport. Prefill fusion avoids the separate add launch and output reread in the isolated comparison. Decode fusion remains an explicit candidate with undecided net benefit against compact addition once extraction and transport are charged. No net host saving is established.

Receipt identifiers are relative to the ONDEV2 evidence job root; `out/` denotes its archived evidence directory. The completed qualification summary `out/qwen-fusions-timed/summary.json` has SHA-256 `95597cf29663583fc97bd4bfc27a3ba063b53c58d97ea99670a26272d3fc377d`. It contains all 6 cases, 18 receipt digests, the frozen generator and qualification source digests, compiler tool identities, and the compact decode receipt `out/qwen-fusions-compact128/result.json`, SHA-256 `b8b7e4f54fec9d622f46a2d72159890ebca30d189e6db801d31ebec2c603af32`. The compiler uses the INTEG2 binaries identified in those receipts. Actual RTL simulation, connected Qwen execution, pretrained accuracy, and net host timing are PENDING.
