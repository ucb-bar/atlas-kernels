# causal_decoder kernel bench at model shapes

Regenerate: `MERLIN_ROOT=/path/to/merlin EXPORT_PYTHON=/path/to/export/bin/python MODEL_PYTHON=/path/to/model/bin/python NPU_MODEL=/path/to/npu-model python3 -B coverage/examples/causal_decoder/bench.py --atlas-mlir-bin /path/to/atlas-mlir/build/bin --min-tier stream-verified --out "$OUT"`

Input: `../out/causal_decoder-final/table/coverage.json` (SHA-256 `a9c5d9b7ec6e14718e171f6cb7c18201329b698bc60fa7adf762aa0b4a57371e`), coverage status INCOMPLETE. The coverage derives 7 distinct launches; 3 executed on npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` with numerics="rtl" on seeded synthetic inputs at model shapes (12 worker processes; 18 job-seconds in all, longest job 4 s).

Model cycles are npu_model Core cycle-model counts, not RTL cycles. Cycles x count multiplies one launch by the number of launches in the model; the sums below are device-only estimates that ignore host time, launch overhead, and data movement between launches. Bit-exact compares output bytes with the independent lib/golden reference where the family has one; cosine and rel-err compare the decoded output with an FP32 PyTorch reference on the unquantized inputs, so they include FP8 input quantization. Inputs are synthetic, so fidelity and data-dependent cycles are representative, not the model's.

A distinct launch groups the same shape and variant; a distinct program is the address-specific program text, as in the separate 1,589-program kit check.

Shares below are by launch kind: `epilogue`, `rne_add`, `rope`, and `smol_fusion` include their GEMM projection and fused stage, while `projection_gemm` covers unfused panels. They are not shares of non-GEMM time; the generator metadata does not provide separately measured cycles for those stages.

Compiler backend: machine IR, atlas-opt stream and timing verifiers, and atlas-emit, with minimum tier `stream-verified`. Successful jobs require emitted words to equal npu_model's assembler; per-job JSON records binary SHA-256 digests, diagnostics, and word counts.

| device launches | derived here | qualified kit |
| --- | ---: | ---: |
| projection_panels | 6 | not declared |
| rmsnorm | 2 | not declared |

## Summary

Device cycle estimate (sum of model cycles x count over 4 of 8 launches): 125,989 npu_model model cycles. It ignores host time and launch overhead.

These totals include historically reused functional and timed results; zero functional or timed bench jobs were rerun for this retained report. Reconstructed staging, program-word checks, and per-job result hashes are recorded in `reuse-receipt.json` beside the generated JSON.

Compiler tier counts by distinct launch: 3 `verified`.

| launch kind | launches | launches with cycles | model cycles x count | share |
| --- | ---: | ---: | ---: | ---: |
| `rmsnorm` | 2 | 2 | 108,372 | 86.02% |
| `projection_gemm` | 2 | 2 | 17,617 | 13.98% |
| `epilogue` | 2 | 0 | 0 | 0.00% |
| `rne_add` | 2 | 0 | 0 | 0.00% |

Top shapes by total cycles:

| family | shape | count | model cycles | cycles x count |
| --- | --- | ---: | ---: | ---: |
| `rmsnorm` | 32x32 (8 rows) | 2 | 54,186 | 108,372 |
| `projection_gemm` | 32x32x96 (logical 8x32x96) | 1 | 10,559 | 10,559 |
| `projection_gemm` | 32x32x64 (logical 8x32x64) | 1 | 7,058 | 7,058 |

Unsupported or failed launches: 4.
- `epilogue` 32x32x64 (logical 8x32x64) mul FP8 out x1: no-variant: kernels/epilogue/gen.py --shape 32x32x64 --steps mul --fp8 --rescale-exponent -19 --output-scale-exponent -9 --output-row-stride 2048: ValueError: traffic reuse is supported only for the qualified shapes
- `epilogue` 32x32x64 (logical 8x32x64) silu x1: no-variant: kernels/epilogue/gen.py --shape 32x32x64 --steps silu --rescale-exponent -19: ValueError: traffic reuse is supported only for the qualified shapes
- `rne_add` 32x32x32 (logical 8x32x32) x1: no-variant: kernels/epilogue/gen.py --shape 32x32x32 --steps add_rne --rescale-exponent -19: ValueError: traffic reuse is supported only for the qualified shapes
- `rne_add` 32x64x32 (logical 8x64x32) x1: no-variant: kernels/epilogue/gen.py --shape 32x64x32 --steps add_rne --rescale-exponent -19: ValueError: traffic reuse is supported only for the qualified shapes

## Launches

| family | shape | export dtype | variant | tier | words | count | model cycles | cycles x count | golden | bit-exact | timed equals functional | cosine | rel-err | modules |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | :-: | :-: | ---: | ---: | --- |
| `epilogue` | 32x32x64 (logical 8x32x64) mul FP8 out | float32 | no-variant | n/a | | 1 | | | | | | | | `up` |
| `epilogue` | 32x32x64 (logical 8x32x64) silu | float32 | no-variant | n/a | | 1 | | | | | | | | `gate` |
| `projection_gemm` | 32x32x64 (logical 8x32x64) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 74 | 1 | 7,058 | 7,058 | ordered_matmul | exact | exact | 0.99924 | 3.912% | `head` |
| `projection_gemm` | 32x32x96 (logical 8x32x96) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 74 | 1 | 10,559 | 10,559 | ordered_matmul | exact | exact | 0.99935 | 3.608% | `qkv` |
| `rmsnorm` | 32x32 (8 rows) | float32 | streamed tile-row RMSNorm, FP8 out | verified | 148 | 2 | 54,186 | 108,372 | none | n/a | exact | 0.99968 | 2.517% | `norm` |
| `rne_add` | 32x32x32 (logical 8x32x32) | float32 | no-variant | n/a | | 1 | | | | | | | | `projection` |
| `rne_add` | 32x64x32 (logical 8x64x32) | float32 | no-variant | n/a | | 1 | | | | | | | | `down` |

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

Placement comes from `coverage/examples/causal_decoder/mapping.yaml`, which maps only ops an existing family computes and cites why; the attention core stays unmapped because this decoder uses head dimension 8, while the family supports 64, 128, and 256. The export is `export_ops.py causal_decoder --loader merlin:examples/workloads/causal_decoder/loader.py`, the same command as `coverage/examples/causal_decoder/REPORT.md`, which lists the decoder with no placement rules.
