# epilogue

Ordered BF16 tile epilogues fused into a projection's output store. An `Epilogue` runs after each output tile is popped from MXU0 and before it is stored, so element-wise work that follows a GEMM costs no extra launch or DRAM round trip.

Stage: fused into a projection_gemm launch (no launch of its own).

## Entry points

`atlas_epilogue.py`:

- `Epilogue(m, n, output_offset, steps=(), output_dtype="bf16", output_scale_exponent=0, output_layout="tiles", scratch_registers=(30, 31), output_row_stride=None, output_column_tile=0)` describes the fused work for one `m x n` output. Pass it as `epilogue=` to any [projection_gemm](../projection_gemm/README.md) entry point.
- `Step(op, operand=None)` is one ordered operation. `op` is one of `add`, `add_rne` (exact BF16 add, [rne_add](../rne_add/README.md)), `mul`, `relu`, `exp`, `recip`, `silu`, `gelu_tanh`, or `connector_scale`.
- `TileInput(offset, layout="tiles", dtype="bf16", scale_exponent=0)` is a staged operand: a full tile tensor, or with `layout="channel"` one BF16 broadcast tile per column (a per-channel bias).
- `output_dtype="fp8"` packs the result to FP8 E4M3 with `output_scale_exponent`, producing the next projection's input directly; `output_layout` selects compact tiles or channels-last rows.

`atlas_resident_output.py` `retain_output(epilogue, output_reg)` keeps the BF16 output pair in VMEM banks 2 and 3 for a following post-pass ([rmsnorm](../rmsnorm/README.md), [rope](../rope/README.md)).

## Where the models use it

- TinyLlama (atlas-mlir `tools/tiny_full/fusion.py` `prepare_projection`): residual add after the o and down projections (`add_rne`), SiLU on the gate projection, and the up times gate product with FP8 output for the down projection.
- SmolVLA ([smol_fusion](../smol_fusion/README.md)): vision fc1 bias plus tanh GELU with optional FP8 output, and the connector scale.
- ResNet50 (atlas-mlir `tools/resnet_sims/fast.py` and Merlin `examples/atlas/comparisons/refresh/capture_resnet_device.py`): the captured per-convolution `mul`, `add`, and `relu` hooks, emitted through `legacy_epilogue_assembly`.

## Generate

```sh
make gen-epilogue NPU_MODEL=/path/to/npu-model ARGS="--shape 128x2048x1536 --steps silu"
make gen-epilogue NPU_MODEL=/path/to/npu-model ARGS="--shape 128x2048x2048 --steps add_rne"
```

`gen.py` writes the program text (`.S`), the 32-bit words npu_model's assembler produces (`.bin`), and a `.json` record into `build/epilogue/`. `kernel.yaml` is the machine-readable manifest.

## Reference and bench results

`reference.py` defines the bench cases and their FP32 PyTorch references for `lib/kernel_bench.py`: 2 cases (TinyLlama gate SiLU and up multiply with FP8 output); no Atlas-arithmetic golden yet, so they report fidelity and cycles only. [`RESULTS.md`](RESULTS.md) lists bit-exactness, cosine similarity and relative error against the FP32 reference, and npu_model model cycles (not RTL cycles) for each case, run on npu_model 47b8599 with `numerics="rtl"`.

## Tests and checks

`checked_assembly` (`lib/atlas_program_checks.py`) admits a hook only when the recipe names none of the registers it writes; `kernels/projection_gemm/tests/test_program_checks.py` and `test_program_legacy_epilogue.py` cover every recipe with hooks. In atlas-mlir, `test/test_tiny_fusion.py` and the Smol fusion tests check staged layouts, and the capture flows execute every fused launch on npu_model `47b8599`.
