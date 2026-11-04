# smol_fusion

SmolVLA projection panels with fused element-wise work: a [projection_gemm](../projection_gemm/README.md) launch whose [epilogue](../epilogue/README.md) adds a per-channel BF16 bias, applies tanh GELU, or applies the connector scale, relocated to the SmolVLA arena at `0x90000000`. The FP8 variant writes vision fc1's output as natural FP8 tiles that fc2 reads directly.

Stage: a projection launch with a fused epilogue (one launch per panel).

## Entry points

- `smol_fusion_program.py` `stage_panel(row, initial, *, operations=(), bias_bf16=None, weight_source=None)` returns a `FusionProgram(source, arena, metadata)` for one captured panel. `row` is the panel record from the SmolVLA plan (module, physical and logical MxKxN, output offset, rescale exponent, recipe), `initial` the staged arena bytes, and `operations` one of `()`, `("bias",)`, `("bias", "gelu_tanh")`, or `("connector_scale",)`. `channel_tiles(bias_bf16, n)` lays out the bias.
- `smol_fusion_fp8.py` `stage_fc1_fp8(row, initial, *, bias_bf16, output_scale_exponent, weight_source=None)` is the fc1 bias plus GELU producer with FP8 output.
- With `weight_source`, programs read FP8 weights in place and relocate through `lib/smol_weight_relocate.py`; otherwise through `chipyard.arena.relocate_assembly` (atlas-mlir's, or the identical stand-in in `lib/fallback/` when atlas-mlir is not importable).

## Where the model uses it

| Variant | Shape (MxKxN) | SmolVLA modules | Notes |
| --- | --- | --- | --- |
| `stage_fc1_fp8` | 128x768x3072 | `vision_model.encoder.layers.*.mlp.fc1` | 192 panels in the `fc1-bias-gelu-fp8` fusion set of the kit build |
| `stage_panel` with `bias`, `bias, gelu_tanh`, or `connector_scale` | the projection_gemm reuse shapes | vision and connector projections | available; the default SmolVLA builds run bias, GELU, and the connector scale on the host in Merlin C |

## Generate

```sh
make gen-smol_fusion NPU_MODEL=/path/to/npu-model ARGS="--shape 128x768x3072 --fp8"
make gen-smol_fusion NPU_MODEL=/path/to/npu-model ARGS="--shape 128x768x768 --operations bias"
```

`gen.py` stages a deterministic panel and writes `.S`, `.bin` (npu_model-assembled words), and `.json` into `build/smol_fusion/`. `kernel.yaml` is the machine-readable manifest.

## Tests and checks

There are no tests in this directory yet. In atlas-mlir, `test/test_smol_fusion_program.py`, `test/test_smol_fp8_producer.py`, and `test/test_smol_fusion_assets.py` check staging, layouts, and relocation, and the SmolVLA fast-build ELF identity check regenerates all 2,715 fast launches (629 distinct programs), fused panels included, through these generators and requires byte-identical ELFs.
