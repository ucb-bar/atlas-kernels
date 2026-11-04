# rmsnorm

Row normalization as a serialized post-pass appended to a projection launch. The producer projection retains its post-residual BF16 output `h` in VMEM banks 2 and 3 ([epilogue](../epilogue/README.md) `retain_output`) and stores it as usual; the post-pass then computes per-row statistics with two register pairs and emits the normalized, gamma-scaled result as natural FP8 tiles immediately after `h` in the same output extent, ready to be the next projection's A operand.

Stage: a post-pass. TinyLlama runs it as a standalone launch that first DMAs the BF16 rows into VMEM banks 2 and 3 (atlas-mlir `tools/tiny_full/norm.py`).

## Entry points

- `atlas_rmsnorm_postpass.py`: `ResidentRMSNormEpilogue`, an `Epilogue` with `norm_weight_offset` (BF16 gamma), `norm_mask_offset`, and `norm_scale_exponent` (default -8, the FP8 output scale); `postpass(epilogue)` returns the post-pass instructions and `append_postpass(source, epilogue)` appends them to a producer program. This is the original 128x2048 RMSNorm prototype.
- `atlas_norm_postpass.py`: `ResidentNormEpilogue` extends it with `norm_kind` (`"rms"` or `"layer"`), `real_n` (logical width; padded columns of `h` and gamma must be zero), and `norm_bias_offset` (LayerNorm beta), with the original 2048-wide RMS path unchanged. Its `postpass` and `append_postpass` have the same contract. Only m10/m11 and m12/m13 hold statistics.

## Where the models use it

TinyLlama RMSNorm, width 2048, after the attention and MLP residuals: 128 rows in prefill and 1 row padded to 32 in decode (atlas-mlir `tools/tiny_full/norm.py` builds `ResidentNormEpilogue(..., norm_kind="rms", real_n=2048)`; `tools/tiny_full/norm_export.py` exports the launches). SmolVLA normalizations run on the host in Merlin C today.

## Generate

```sh
make gen-rmsnorm NPU_MODEL=/path/to/npu-model ARGS="--rows 128"
make gen-rmsnorm NPU_MODEL=/path/to/npu-model ARGS="--rows 32"
```

`gen.py` writes the program text (`.S`), the 32-bit words npu_model's assembler produces (`.bin`), and a `.json` record into `build/rmsnorm/`. `kernel.yaml` is the machine-readable manifest.

## Tests and checks

`tests/test_norm_postpass.py` checks register hygiene (reserved scratch pairs and the default-sentinel refusal). There is no standalone numeric test yet; exactness is established by executing every TinyLlama norm launch on npu_model `47b8599` with `numerics="rtl"` in the export flow (atlas-mlir `tools/tiny_full/export_capture.py --qualify-launches`).
