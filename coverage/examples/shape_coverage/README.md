# Attention and RMSNorm shape coverage

Run from the repository root with the Python interpreter that supports npu_model:

```sh
PYTHONPATH="$NPU_MODEL" ATLAS_NPU_MODEL_ROOT="$NPU_MODEL" "$MODEL_PYTHON" coverage/bench_shapes.py --model-root "$NPU_MODEL" --atlas-mlir-bin "$ATLAS_MLIR_BIN" --out "$OUT" --workers 12
```

[MODEL-BENCH.md](MODEL-BENCH.md) records 28 configurations, all with full cycle-model counts: 16 attention configurations and 12 RMSNorm configurations. Each program is cross-checked through `atlas_backend` against the assembler, executed with npu_model `47b8599` RTL numerics, compared with the family's FP32 reference, and executed on the Core cycle model. Timed and functional output bytes must match. The driver additionally requires cosine at least 0.98 and relative L2 error at most 0.20. Neither family has a general independent Atlas-arithmetic golden; the unit tests also check analytically known constant or single-key outputs exactly.

Attention covers head dimensions 128 and 256, causal and no mask, BF16 and FP8 output, GQA 2/1, and query/key shapes 32x64 and 1x65. RMSNorm covers widths 32, 960, 1024, 2304, 4096, and 7136 at 32 and 128 rows. This matrix checks representative configurations and capacity boundaries, not every admitted combination of sequence lengths, head counts, scales, and widths.

Hardware geometry comes from npu_model `npu_model/configs/hardware/default.py` at `47b8599`, plus atlas-mlir `docs/dialect-reference.md` and `docs/hardware-status.md`: 32x32 FP8 MXU tiles, 64 tensor registers of 1024 bytes, even register pairs for BF16 tiles, 2 accumulators per MXU, 1572864 bytes of VMEM, 1024-byte-aligned vector accesses contained within a bank, and DMA commands of 32-byte multiples through 4096 bytes. The generators split larger transfers and wait before reuse.

Dimension 128 uses 4 MXU K tiles; dimension 256 uses 8. Two accumulators emit successive 64-column PV panels. Resident K and V each reserve 262144 bytes, so `ceil(keys/32) * 32 * head_dim` must fit that reservation. This permits 2048 keys at dimension 128 and 1024 at dimension 256. Descriptor storage separately requires `query_tiles * key_tiles <= 2048`; query tiles must be at most 128. These are resident-layout limits, not evidence that longer attention is impossible with streaming.

RMSNorm streams one 32-row H block. Its resident broadcast gamma requires `64 * width` bytes after byte 1114112 and before output staging at byte 1570816, limiting the current layout to width 7136. Width 7168 would overlap staging by 2048 bytes. Streaming gamma could admit wider widths; this change rejects that overlap explicitly. All requested widths fit the hardware.

The dimension-64 attention path and the 2048-wide RMSNorm path retain their original generator modules. Frozen-model compatibility is checked by rebuilding every one of the 26 TinyLlama and 28 SmolVLA launch configurations and comparing source text, assembled words, staged arenas, and generator metadata against `6123bef`; direct family CLI artifacts are compared separately.
