# attention

Head-dimension-64 FP8 attention in one launch per attention call: QK^T on the MXU, a masked softmax on the VPU (exponentials with `vexp`, normalization with `vrecip`), and PV. Integral grouped-query attention is supported (query heads a multiple of KV heads).

Stage: standalone launch, one per attention call (all heads).

## Entry points

`atlas_attention_v2_program.py` (current; byte-identical to the source atlas-mlir pins by SHA-256 in `tools/tiny_full/attention_v2_worker.py`):

- `AttentionV2Config(query_rows, keys, query_heads, kv_heads, head_dim=64, mask="none", query_start=None, scale=0.125, q_exponent=0, k_exponent=0, v_exponent=0, p_exponent=-8, output_fp8=False, o_exponent=0, packed_decode=True)`. `mask` selects no mask or causal; `query_start` places decode rows; unnormalized FP8 exponentials use the fixed P scale 256 (`p_exponent=-8`); `output_fp8` writes FP8 output in the projection A layout with `o_exponent`, otherwise BF16. At most 64 key tiles and 128 query tiles.
- `stage_v2(config, q, k, v, mask_tiles=None)` returns the initial arena bytes and the offsets dictionary; `assembly_v2(config, offsets, *, dram_base=0x90000000)` returns the checked program; `check_program_v2(source)` is the static check; `pack_output_v2` and `unpack_output_v2` convert outputs.

`atlas_attention_program.py` is the first version (`AttentionConfig`, `stage`, `assembly`, `pack_tiles`, `unpack_tiles`), kept for the legacy TinyLlama attention worker.

## Shapes the models use

| Model | Calls | Configuration |
| --- | --- | --- |
| TinyLlama | prefill (128 query rows, 128 keys), decode (1 row, 129 and 130 keys) | 32 query heads, 4 KV heads, causal, FP8 output |
| SmolVLA | vision (1024 rows, 1024 keys, 12/12 heads), prefix (177 rows, 177 keys, 15/5 heads), expert (50 rows, 227 or 177 keys, 15/5 heads) | BF16 output |

## Generate

```sh
make gen-attention NPU_MODEL=/path/to/npu-model ARGS="--rows 128 --keys 128 --heads 32/4 --causal --fp8"
make gen-attention NPU_MODEL=/path/to/npu-model ARGS="--rows 177 --keys 177 --heads 15/5"
```

`gen.py` writes the program text (`.S`), the 32-bit words npu_model's assembler produces (`.bin`), and a `.json` record into `build/attention/`. `kernel.yaml` is the machine-readable manifest.

## Reference and bench results

`reference.py` defines the bench cases and their FP32 PyTorch references for `lib/kernel_bench.py`: 2 cases (TinyLlama prefill causal 128x128 with 32/4 heads and FP8 output, SmolVLA prefix 177x177 with 15/5 heads); no Atlas-arithmetic golden yet, so they report fidelity and cycles only. [`RESULTS.md`](RESULTS.md) lists bit-exactness, cosine similarity and relative error against the FP32 reference, and npu_model model cycles (not RTL cycles) for each case, run on npu_model 47b8599 with `numerics="rtl"`.

## Tests and checks

Run `make test-lib test-coverage NPU_MODEL=/path/to/npu-model` from the repository root. See [shared test coverage](../../README.md#running-the-tests).

The program check rejects malformed register use at generation time. In atlas-mlir, `test/test_tiny_attention_v2_export.py`, `test/test_smol_fast_attention.py`, and `test/test_atlas_execution.py` (version 1, with `ATLAS_EXECUTION_MODEL_ROOT`) cover staging and execution, and the TinyLlama and SmolVLA capture flows execute every attention launch on npu_model `47b8599` with `numerics="rtl"` (`tools/tiny_full/attention_v2_worker.py`, `tools/capture_smolvla_attention.py --execute`). There is no standalone PyTorch reference test in this repository yet.
