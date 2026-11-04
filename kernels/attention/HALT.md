# Attention v2 halt variant

Select `gen.py --variant v2-no-halt-delay` or import `atlas_attention_v2_halt.assembly_v2` with the original arguments. The wrapper removes exactly the terminal `delay 128`, preserves `dma.wait ch0` and `dma.wait ch1`, and rejects an unexpected tail. Original v2 remains the default; shipped kit assets are unchanged.

## Completion argument

ECALL is not a fence. RTL `atlas/scalar/ScalarCore.scala:237,248,323` detects ECALL before stall gating and clears the delay counter. `diplomatic/top/AtlasCore.scala:78,92,96` stops fetch and reports scalar halt without resetting the engines; `diplomatic/memory/CSRFile.scala:77` exposes halt without checking them. In-flight work can continue after halt. These citations are relative to the RTL Scala source root and apply to the inspected default configuration.

These programs are drained before ECALL: `atlas_attention_v2_program.py:575` serializes the final MXU pop, VPU multiply/clipping, and stores. Pop delay 34 and VPU delay 66 cover the default 32-row streams (`atlas/mxu/ipt/InnerProductTreesSequencer.scala:560`, `atlas/mxu/sa/SystolicArraySequencer.scala:522`, `atlas/vector/VectorFSM.scala:375`). Both output formats retain store delay 34 for LSU writeback (`atlas/lsu/LSU.scala:286,313,327`). Packed and unpacked output DMA stores wait on ch0 (`atlas_attention_v2_program.py:594,598`); only scalar loop-exit instructions and the final ch0/ch1 waits follow. DMA busy clears after full dispatch and all responses, including store acknowledgements (`diplomatic/memory/DMA.scala:291,309`); `diplomatic/top/AtlasCore.scala:185` covers launch-to-busy latency. No output DMA remains unacknowledged, so no replacement fence is needed. This is a static hardware argument, not an RTL execution pass.

Pinned npu_model `47b8599` agrees: `npu_model/hardware/core.py:190,216,241` handles ECALL before delay and waits for channel retirement; `npu_model/hardware/dma.py:380` retires DMA after dispatch and responses. The removed delay cannot drain work at this position.

## Qualification and limits

Archived qualification covers 45 distinct programs (25 TinyLlama, 20 SmolVLA), 1 captured input each, and 90 independent old/new Core executions with `numerics="rtl"`, seed 42, and native acceleration disabled. All 45 full final arenas are bit-identical; each program saves exactly 1 model cycle, 1 word, and 1 completed instruction. Outputs match captured expectations, guards remain intact, and every engine and all 8 DMA flags are drained at halt. All 266 launch configurations (66 TinyLlama, 200 SmolVLA) regenerate correctly; execution does not cover all 266 input sets.

All 45 new programs are stream-verified with assembler/compiler word agreement. Continued diagnostic checks reduce halt-after-DELAY violations from 45 to 0 with all other diagnostic multisets unchanged. Full timing verification still rejects the existing DMA-loop diagnostics. PENDING: independent full 90-execution rerun, RTL/SoC execution, host cache/ordering validation, and the 16-ELF identity rebuild. Qualification receipts and the job-specific driver are retained with the job evidence, outside this repository.

## Kit adoption

Advance atlas-mlir's atlas-kernels gitlink to a reviewed commit, explicitly select and record the variant in generation, execution, and admission, and authenticate both the wrapper and original generator dependency. TinyLlama requires coordinated worker/export admission and migrated library source closure; SmolVLA requires coordinated capture worker, fast-attention, asset selector, and source bookends. Preserve historical admission or introduce distinct authenticated admission for the variant.

Regenerate all 66 TinyLlama and 200 SmolVLA attention assets, program hashes, word counts, halt PCs, instruction counts, source pins, checksum manifests, contracts, and replay/qualification receipts. Rebuild affected default and fast host assets and ELFs, then repeat kit admission, backend, launch exactness, and host/runtime qualification. Do not relabel old receipts.

Raw-ECALL savings do not predict completion-marker timing. Historical kit runners replace ECALL with a completion-marker sequence that lets the terminal delay take effect; remeasure that runtime after adoption.
