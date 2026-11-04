# rmsnorm

Row normalization as a serialized post-pass appended to a projection launch. The producer projection retains its post-residual BF16 output `h` in VMEM banks 2 and 3 ([epilogue](../epilogue/README.md) `retain_output`) and stores it as usual; the post-pass then computes per-row statistics with two register pairs and emits the normalized, gamma-scaled result as natural FP8 tiles immediately after `h` in the same output extent, ready to be the next projection's A operand.

Stage: a post-pass. TinyLlama runs it as a standalone launch that first DMAs the BF16 rows into VMEM banks 2 and 3 (atlas-mlir `tools/tiny_full/norm.py`).

## Entry points

- `atlas_rmsnorm_tiled.py`: `standalone_source(rows, width, mask_offset, weight_offset, scale=-8)` returns unrelocated assembly for widths rejected by the pinned RMS post-pass. It streams one 32-row BF16 tile row at a time, accumulates statistics in groups of at most 32 column tiles, and writes natural FP8 tiles at byte offset `2 * rows * width`. `gen.py` selects it for widths other than 960 and 2048.
- `atlas_rmsnorm_postpass.py`: `ResidentRMSNormEpilogue`, an `Epilogue` with `norm_weight_offset` (BF16 gamma), `norm_mask_offset`, and `norm_scale_exponent` (default -8, the FP8 output scale); `postpass(epilogue)` returns the post-pass instructions and `append_postpass(source, epilogue)` appends them to a producer program. This is the original 128x2048 RMSNorm prototype.
- `atlas_norm_postpass.py`: `ResidentNormEpilogue` extends it with `norm_kind` (`"rms"` or `"layer"`), `real_n` (logical width; padded columns of `h` and gamma must be zero), and `norm_bias_offset` (LayerNorm beta), with the original 2048-wide RMS path unchanged. Its `postpass` and `append_postpass` have the same contract. Only m10/m11 and m12/m13 hold statistics.

## Where the models use it

TinyLlama RMSNorm, width 2048, after the attention and MLP residuals: 128 rows in prefill and 1 row padded to 32 in decode (atlas-mlir `tools/tiny_full/norm.py` builds `ResidentNormEpilogue(..., norm_kind="rms", real_n=2048)`; `tools/tiny_full/norm_export.py` exports the launches). SmolVLA normalizations run on the host in Merlin C today.

## Generate

```sh
make gen-rmsnorm NPU_MODEL=/path/to/npu-model ARGS="--rows 128"
make gen-rmsnorm NPU_MODEL=/path/to/npu-model ARGS="--rows 32"
make gen-rmsnorm NPU_MODEL=/path/to/npu-model ARGS="--rows 128 --width 4096"
```

`gen.py` admits rows 32 or 128 and positive widths divisible by 32 through 7136. Widths 960 and 2048 retain the pinned resident path, including byte-identical 2048-wide programs. Other widths use the separate standalone path. The standalone generator requires logical width to equal physical width and fixes eps at 1e-5. The DRAM arena remains BF16 H, FP8 output, a 256-byte guard, 65536 mask bytes, and `64 * width` broadcast gamma bytes.

The streamed layout uses the default 1572864-byte VMEM: H starts at byte 524288 and occupies `64 * width` bytes, masks start at byte 1048576, gamma starts at byte 1114112, and the 2048-byte output staging region starts at byte 1570816. Gamma must end by staging, giving width 7136. This is the current generator layout capacity; wider rows could use another streaming layout.

`gen.py` writes the program text (`.S`), the 32-bit words npu_model's assembler produces (`.bin`), and a `.json` record into `build/rmsnorm/`. `kernel.yaml` is the machine-readable manifest.

## Reference and bench results

`reference.py` defines the bench cases and their FP32 PyTorch references for `lib/kernel_bench.py`: 10 cases (TinyLlama prefill 128 rows and decode 1 row, plus widths 32, 1024, 2304, and 4096 at 32 and 128 physical rows); no Atlas-arithmetic golden yet, so they report fidelity and cycles only. [`RESULTS.md`](RESULTS.md) retains the TinyLlama results. The [shape-coverage bench](../../coverage/examples/shape_coverage/MODEL-BENCH.md) checks widths 32, 960, 1024, 2304, 4096, and 7136 at both 32 and 128 rows: 12 full cycle-model launches, all `verified`, with timed output identical to functional output. Minimum cosine is 0.9996254347070453, and maximum relative error is 0.02750694796445788. These are npu_model model cycles, not RTL cycles.

## Tests and checks

`tests/test_norm_postpass.py` checks register hygiene (reserved scratch pairs and the default-sentinel refusal). `tests/test_rmsnorm_tiled.py` checks admission, arena bounds, the short DMA transfer for width 32, and exact signed unit rows at the expanded widths on npu_model when `ATLAS_NPU_MODEL_ROOT` is set. The random-input bench still has no independent Atlas-arithmetic golden. Legacy exactness is established by executing every TinyLlama norm launch on npu_model `47b8599` with `numerics="rtl"` in the export flow (atlas-mlir `tools/tiny_full/export_capture.py --qualify-launches`).
