# epilogue

Ordered BF16 tile epilogues fused into a projection_gemm output store. An `Epilogue` runs between each MXU0 output-tile pop and store, avoiding an extra launch or DRAM round trip for element-wise work. The opt-in contiguous SiLU times up stream below launches separately.

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

## Reference and bench results

`reference.py` provides FP32 PyTorch references for the [shared bench](../../lib/README.md): 2 cases (TinyLlama gate SiLU and up multiply with FP8 output); no Atlas-arithmetic golden yet, so they report fidelity and cycles only. [`RESULTS.md`](RESULTS.md) reports exactness, FP32 fidelity, and model cycles under the [shared bench conventions](../../README.md#model-coverage).

## Tests and checks

From the repository root, run `make test-lib test-projection_gemm test-coverage NPU_MODEL=/path/to/npu-model`; see [shared coverage](../../README.md#running-the-tests).

`checked_assembly` (`lib/atlas_program_checks.py`) admits a hook only when the recipe names none of the registers it writes; `kernels/projection_gemm/tests/test_program_checks.py` and `test_program_legacy_epilogue.py` cover every recipe with hooks. In atlas-mlir, `test/test_tiny_fusion.py` and the Smol fusion tests check staged layouts, and the capture flows execute every fused launch on npu_model `47b8599`.

## Opt-in contiguous SiLU times up stream

`atlas_swiglu_stream.assembly(elements, dram_base=0x90000000)` is a separate standalone launch. It reuses `Step("silu")` and `Step("mul", TileInput(...))` without changing their arithmetic. `layout(elements)` returns the padded element count, input and output byte offsets, four 256-byte guards, and arena extent. Both gate and up must be finite BF16 buffers in the same contiguous lane order. Because every arithmetic operation is elementwise, their common lane permutation cancels at output, and no 2D layout conversion is required. Zero pad only the last 1024-element chunk of both inputs and discard the output tail. VMEM is fixed at 4096 bytes: 2048 bytes for gate and 2048 shared by up and output.

```sh
make gen-epilogue NPU_MODEL=/path/to/npu-model ARGS="--swiglu-elements 453120"
```

This mode writes BF16 output and rejects projection steps, FP8 output, projection rescaling, and output-layout options. `--shape` is mutually exclusive with `--swiglu-elements`; projection mode still requires `--shape` and `--steps` and keeps its existing program words. No default placement or selection rule enables the stream.

The native SiLU policy uses the existing approximate RTL EXP and reciprocal lookup tables, a truncating denominator addition, and BF16 RNE multiplications. It is not exact host SiLU and does not establish a model-level action verdict. A model must pass its independent full-action rule before enabling this primitive. `tests/test_swiglu_stream.py` checks original47 instruction execution against an independent NumPy table and rounding oracle, including tails, complete arenas, guards, DMA counts, and address admission. Run `make test-epilogue NPU_MODEL=/path/to/npu-model`.
