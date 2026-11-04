# causal_decoder kernel bench at model shapes

Regenerate: `MERLIN_ROOT=/path/to/merlin TORCH_PYTHON=/path/to/venv/bin/python NPU_MODEL=/path/to/npu-model python3 -B coverage/examples/causal_decoder/bench.py --out "$OUT"`

Input: `table/coverage.json` (SHA-256 `a9c5d9b7ec6e14718e171f6cb7c18201329b698bc60fa7adf762aa0b4a57371e`), coverage status INCOMPLETE. Every distinct device launch the coverage places on Atlas ran on npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` with numerics="rtl" on seeded synthetic inputs at the model's shape (32 worker processes; 14 job-seconds in all, longest job 4 s).

Model cycles are npu_model Core cycle-model counts, not RTL cycles. Cycles x count multiplies one launch by the number of launches in the model; the sums below are device-only estimates that ignore host time, launch overhead, and data movement between launches. Bit-exact compares output bytes with the independent lib/golden reference where the family has one; cosine and rel-err compare the decoded output with an FP32 PyTorch reference on the unquantized inputs, so they include FP8 input quantization. Inputs are synthetic, so fidelity and data-dependent cycles are representative, not the model's.

| device launches | derived here | qualified kit |
| --- | ---: | ---: |
| projection_panels | 6 | not declared |
| rmsnorm | 2 | not declared |

## Summary

Device cycle estimate (sum of model cycles x count over 2 of 8 launches): 17,617 npu_model model cycles. It ignores host time and launch overhead.

| family | launches | launches with cycles | model cycles x count | share |
| --- | ---: | ---: | ---: | ---: |
| `projection_gemm` | 2 | 2 | 17,617 | 100.00% |
| `epilogue` | 2 | 0 | 0 | 0.00% |
| `rmsnorm` | 2 | 0 | 0 | 0.00% |
| `rne_add` | 2 | 0 | 0 | 0.00% |

Top shapes by total cycles:

| family | shape | count | model cycles | cycles x count |
| --- | --- | ---: | ---: | ---: |
| `projection_gemm` | 32x32x96 (logical 8x32x96) | 1 | 10,559 | 10,559 |
| `projection_gemm` | 32x32x64 (logical 8x32x64) | 1 | 7,058 | 7,058 |

Unsupported or failed launches: 5.
- `epilogue` 32x32x64 (logical 8x32x64) mul FP8 out x1: no-variant: kernels/epilogue/gen.py --shape 32x32x64 --steps mul --fp8 --rescale-exponent -19 --output-scale-exponent -9 --output-row-stride 2048: ValueError: traffic reuse is supported only for the qualified shapes
- `epilogue` 32x32x64 (logical 8x32x64) silu x1: no-variant: kernels/epilogue/gen.py --shape 32x32x64 --steps silu --rescale-exponent -19: ValueError: traffic reuse is supported only for the qualified shapes
- `rmsnorm` 32x32 (8 rows) x2: no-variant: kernels/rmsnorm/gen.py --rows 32 --width 32 --norm-scale-exponent -7: gen.py: error: argument --width: invalid choice: '32' (choose from '2048')
- `rne_add` 32x32x32 (logical 8x32x32) x1: no-variant: kernels/epilogue/gen.py --shape 32x32x32 --steps add_rne --rescale-exponent -19: ValueError: traffic reuse is supported only for the qualified shapes
- `rne_add` 32x64x32 (logical 8x64x32) x1: no-variant: kernels/epilogue/gen.py --shape 32x64x32 --steps add_rne --rescale-exponent -19: ValueError: traffic reuse is supported only for the qualified shapes

## Launches

| family | shape | export dtype | variant | words | count | model cycles | cycles x count | golden | bit-exact | timed equals functional | cosine | rel-err | modules |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | :-: | :-: | ---: | ---: | --- |
| `epilogue` | 32x32x64 (logical 8x32x64) mul FP8 out | float32 | no-variant | | 1 | | | | | | | | `up` |
| `epilogue` | 32x32x64 (logical 8x32x64) silu | float32 | no-variant | | 1 | | | | | | | | `gate` |
| `projection_gemm` | 32x32x64 (logical 8x32x64) | float32 | default schedule (no traffic-reuse recipe for this shape) | 74 | 1 | 7,058 | 7,058 | ordered_matmul | exact | exact | 0.99924 | 3.912% | `head` |
| `projection_gemm` | 32x32x96 (logical 8x32x96) | float32 | default schedule (no traffic-reuse recipe for this shape) | 74 | 1 | 10,559 | 10,559 | ordered_matmul | exact | exact | 0.99935 | 3.608% | `qkv` |
| `rmsnorm` | 32x32 (8 rows) | float32 | no-variant | | 2 | | | | | | | | `norm` |
| `rne_add` | 32x32x32 (logical 8x32x32) | float32 | no-variant | | 1 | | | | | | | | `projection` |
| `rne_add` | 32x64x32 (logical 8x64x32) | float32 | no-variant | | 1 | | | | | | | | `down` |

`words` is the executed program size in 32-bit instruction words (after the arena relocation the bench applies). Bit-exact covers the logical rows and columns the model consumes; a padding note counts output words in the zero-padded rows or columns that differ from the golden (`+0/-0`: only the sign of zero differs).

## Unmapped ops (16 rows, 16 instances, 10,240 FLOPs)

| phase | module | op | shape | count | FLOPs |
| --- | --- | --- | --- | ---: | ---: |
| inference | `embedding` | `aten.embedding.default` | 64x32, 1x8 -> 1x8x32 float32 | 1 | 0 |
| inference | `(top)` | `aten.chunk.default` | 1x8x96 -> 1x8x32 float32 | 1 | 256 |
| inference | `(top)` | `aten.reshape.default` | 1x8x32 -> 1x8x4x8 float32 | 1 | 0 |
| inference | `(top)` | `aten.transpose.int` | 1x8x4x8 -> 1x4x8x8 float32 | 1 | 0 |
| inference | `(top)` | `aten.reshape.default` | 1x8x32 -> 1x8x4x8 float32 | 1 | 0 |
| inference | `(top)` | `aten.transpose.int` | 1x8x4x8 -> 1x4x8x8 float32 | 1 | 0 |
| inference | `(top)` | `aten.reshape.default` | 1x8x32 -> 1x8x4x8 float32 | 1 | 0 |
| inference | `(top)` | `aten.transpose.int` | 1x8x4x8 -> 1x4x8x8 float32 | 1 | 0 |
| inference | `(top)` | `aten.transpose.int` | 1x4x8x8 -> 1x4x8x8 float32 | 1 | 0 |
| inference | `(top)` | `aten.matmul.default` | 1x4x8x8, 1x4x8x8 -> 1x4x8x8 float32 | 1 | 4,096 |
| inference | `(top)` | `aten.div.Tensor` | 1x4x8x8 -> 1x4x8x8 float32 | 1 | 256 |
| inference | `(top)` | `aten.add.Tensor` | 1x4x8x8, 8x8 -> 1x4x8x8 float32 | 1 | 256 |
| inference | `(top)` | `aten.softmax.int` | 1x4x8x8 -> 1x4x8x8 float32 | 1 | 1,280 |
| inference | `(top)` | `aten.matmul.default` | 1x4x8x8, 1x4x8x8 -> 1x4x8x8 float32 | 1 | 4,096 |
| inference | `(top)` | `aten.transpose.int` | 1x4x8x8 -> 1x8x4x8 float32 | 1 | 0 |
| inference | `(top)` | `aten.reshape.default` | 1x8x4x8 -> 1x8x32 float32 | 1 | 0 |

## Notes

Placement comes from `coverage/examples/causal_decoder/mapping.yaml`, which maps only ops an existing family computes and cites why; the attention core stays unmapped because attention v2 requires head dimension 64 and this decoder uses 8. The export is `export_ops.py causal_decoder --loader merlin:examples/workloads/causal_decoder/loader.py`, the same command as `coverage/examples/causal_decoder/REPORT.md`, which lists the decoder with no placement rules.
