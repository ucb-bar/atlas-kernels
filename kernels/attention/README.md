# attention

Head-dimension-64, 128, and 256 FP8 attention in one launch per attention call: QK^T on the MXU, a masked softmax on the VPU (exponentials with `vexp`, normalization with `vrecip`), and PV. Integral grouped-query attention is supported (query heads a multiple of KV heads).

Stage: standalone launch, one per attention call (all heads).

## Entry points

`atlas_attention_v2_program.py` (current; byte-identical to the source atlas-mlir pins by SHA-256 in `tools/tiny_full/attention_v2_worker.py`):

- `AttentionV2Config(query_rows, keys, query_heads, kv_heads, head_dim=64, mask="none", query_start=None, scale=0.125, q_exponent=0, k_exponent=0, v_exponent=0, p_exponent=-8, output_fp8=False, o_exponent=0, packed_decode=True)`. `mask` selects no mask or causal; `query_start` places decode rows; unnormalized FP8 exponentials use the fixed P scale 256 (`p_exponent=-8`); `output_fp8` writes FP8 output in the projection A layout with `o_exponent`, otherwise BF16. At most 64 key tiles and 128 query tiles.
- `stage_v2(config, q, k, v, mask_tiles=None)` returns the initial arena bytes and the offsets dictionary; `assembly_v2(config, offsets, *, dram_base=0x90000000)` returns the checked program; `check_program_v2(source)` is the static check; `pack_output_v2` and `unpack_output_v2` convert outputs.

`atlas_attention_wide_program.py` is the separate path for previously unsupported head dimensions 128 and 256. `AttentionWideConfig` has the v2 configuration fields, defaults to `head_dim=128`, and accepts `mask="none"` or `mask="causal"`; `stage_wide`, `assembly_wide`, `pack_output_wide`, and `unpack_output_wide` provide the corresponding arena, program, and output codecs. Integral GQA, BF16 output, FP8 projection-layout output, tail tiles, and `query_start` are supported. Decode uses ordinary padded head tiles; `packed_decode` must be `False`.

Wide attention reserves one 256 KiB VMEM bank each for resident K and V, so dimension 128 supports at most 2048 keys, and dimension 256 supports at most 1024 keys. Query tiles are limited to 128, and the product of query tiles and key tiles is limited to 2048 by the 16 KiB descriptor region. Deduplicated masks must fit 192512 bytes. The score and FP8 probability buffers occupy separate regions of bank 1, and two MXU accumulators produce successive 64-column output panels. Every DMA command transfers at most 4096 bytes, including wide input and output head tiles. These are explicit resident-kernel limits; larger key sequences and arbitrary wide-head masks are rejected.

The wide default scale is `head_dim**-0.5`, rounded to nearest-even BF16: 0.08837890625 for dimension 128, and 0.0625 for dimension 256. An explicit positive finite scale receives the same rounding. Other arithmetic follows v2, including FP8 input quantization, BF16 score and reciprocal operations, and fixed P scale 256.

The [shape-coverage bench](../../coverage/examples/shape_coverage/MODEL-BENCH.md) checks 18 wide configurations: both dimensions, both masks, both output formats, and 32x64 or 1x65 query/key shapes with 2/1 heads, plus causal BF16 decode at the resident key-capacity limits d128 1x2048 and d256 1x1024. All reach `stream-verified`, all record full npu_model cycle counts, and every timed output equals its functional output byte for byte. Minimum cosine is 0.9977356825901555, and maximum relative error is 0.06785118773625355 against `reference.py` with the ideal FP32 scale.

`atlas_attention_program.py` is the first version (`AttentionConfig`, `stage`, `assembly`, `pack_tiles`, `unpack_tiles`), kept for the legacy TinyLlama attention worker.

`atlas_attention_v2_halt.py` is the explicit `v2-no-halt-delay` variant. Its `assembly_v2` accepts the same arguments and removes only the dead terminal `delay 128`, retaining `dma.wait ch0` and `dma.wait ch1` before `ecall`. It delegates all other generation to the unchanged, SHA-pinned v2 module and rejects an unexpected tail. Existing imports and the CLI default still select the original v2 programs. See [HALT.md](HALT.md) for the source justification, qualification, and kit adoption pins.

## Shapes the models use

| Model | Calls | Configuration |
| --- | --- | --- |
| TinyLlama | prefill (128 query rows, 128 keys), decode (1 row, 129 and 130 keys) | 32 query heads, 4 KV heads, causal, FP8 output |
| SmolVLA | vision (1024 rows, 1024 keys, 12/12 heads), prefix (177 rows, 177 keys, 15/5 heads), expert (50 rows, 227 or 177 keys, 15/5 heads) | BF16 output |

## Generate

```sh
make gen-attention NPU_MODEL=/path/to/npu-model ARGS="--rows 128 --keys 128 --heads 32/4 --causal --fp8"
make gen-attention NPU_MODEL=/path/to/npu-model ARGS="--rows 177 --keys 177 --heads 15/5"
make gen-attention NPU_MODEL=/path/to/npu-model ARGS="--rows 33 --keys 65 --heads 2/1 --head-dim 256 --causal"
make gen-attention NPU_MODEL=/path/to/npu-model ARGS="--rows 1 --keys 129 --heads 32/4 --causal --query-start 128 --fp8 --variant v2-no-halt-delay"
```

`gen.py` keeps the existing dimension-64 generator and artifact names by default; `--head-dim 128` or `--head-dim 256` selects the wide module and adds a dimension suffix. It writes the program text (`.S`), the 32-bit words npu_model's assembler produces (`.bin`), and a `.json` record into `build/attention/`. The dimension-64-only halt variant requires `--variant v2-no-halt-delay`; combining it with a wide head is rejected. `kernel.yaml` is the machine-readable manifest.

## Reference and bench results

`reference.py` defines the bench cases and their FP32 PyTorch references for `lib/kernel_bench.py`: 6 cases (TinyLlama prefill causal 128x128 with 32/4 heads and FP8 output, SmolVLA prefix 177x177 with 15/5 heads, and 4 synthetic wide-head cases covering dimensions 128 and 256, both mask kinds, tail tiles, decode, and both output formats); no Atlas-arithmetic golden yet, so they report fidelity and cycles only. [`RESULTS.md`](RESULTS.md) lists bit-exactness, cosine similarity and relative error against the FP32 reference, and npu_model model cycles (not RTL cycles) for each case, run on npu_model 47b8599 with `numerics="rtl"`.

## Tests and checks

Run `make test-attention test-lib test-coverage NPU_MODEL=/path/to/npu-model` from the repository root. See [shared test coverage](../../README.md#running-the-tests).

The program check rejects malformed register use at generation time. In atlas-mlir, `test/test_tiny_attention_v2_export.py`, `test/test_smol_fast_attention.py`, and `test/test_atlas_execution.py` (version 1, with `ATLAS_EXECUTION_MODEL_ROOT`) cover staging and execution, and the TinyLlama and SmolVLA capture flows execute every attention launch on npu_model `47b8599` with `numerics="rtl"` (`tools/tiny_full/attention_v2_worker.py`, `tools/capture_smolvla_attention.py --execute`). The wide module adds 7 standalone tests in `tests/test_attention_wide.py`: admission and capacity checks, padded output layout round trips, BF16 scale rounding, 8 exact single-key attention variants with Q=K=0, a seeded causal tail case with 2 KV heads and explicit query position, a small seeded random multi-key GQA case with fidelity and exact timed-versus-functional output checks, and a dimension-256 timed npu_model regression that checks segmented DMA and exact output against the functional model. Execution tests require `ATLAS_NPU_MODEL_ROOT`; the Make target sets it from `NPU_MODEL`. The exact single-key cases use Q=K=0 and varying V values, 4 query heads, and 2 KV heads, so the expected output is independently V repeated across the GQA groups. General random-input attention still has no independent Atlas-arithmetic golden.

`make test-attention NPU_MODEL=/path/to/npu-model` checks the pinned source identity, the exact one-word change across all 7 model shapes, explicit CLI selection, refusal of changed tails, and bit-identical BF16 and packed FP8 execution on small cases. See [HALT.md](HALT.md) for the archived 45-program qualification scope, completion argument, and kit adoption requirements.

The shared [per-family make interface](../README.md) provides `gen`, `data`, `test`, and explicitly provisioned `rtl` targets for this family, with generated payloads in ignored output directories and reference gaps recorded as PENDING.
