# projection_gemm

FP8 E4M3 matrix multiply `C[M,N] = A[M,K] @ B[K,N]` on MXU0, accumulating in BF16 in ascending K order, with one pop to BF16 and an optional power-of-two output rescale. Every linear layer of the three models (attention projections, MLPs, convolutions as GEMMs, connectors, and heads) runs through this family; a fused [epilogue](../epilogue/README.md) may replace the plain BF16 store.

## Entry points

`atlas_program.py`:

- `assembly(m, k, n, output_offset, rescale_exponent=None, *, resident_b=False, pretranspose_resident_b=False, resident_a=False, traffic_reuse=False, epilogue=None, direct_input=None, weight_source=None)` returns the program text. `m`, `k`, and `n` are positive multiples of 32. The arena holds A at offset 0, B after it, and the output at `output_offset`; `rescale_exponent` scales the BF16 result by `2**exponent`. `resident_a` or `resident_b` keeps one operand in VMEM bank 1 when it fits 256 KiB. `traffic_reuse=True` dispatches to the qualified recipe for the exact shape in the [catalog](kernel.yaml). `direct_input` gathers A rows on device ([resnet_patch_gather](../resnet_patch_gather/README.md)); `weight_source` reads FP8 weights in place from an immutable region (`lib/atlas_weight_source.py`).
- `dma_load_bytes(...)` returns the FP8 input bytes the program requests, for DMA accounting.
- `legacy_epilogue_assembly(...)` reproduces the archived non-Tiny C schedules with their original x30/x31 hook registers.
- `source_paths(*, traffic_reuse=False)` maps receipt source names to file paths, resolving library files through `atlas_kernels_path.module_path` and `run_model_tile.py` through the imported helper's origin. `code_sources(*, traffic_reuse=False)` returns the corresponding SHA-256 digests. With atlas-mlir's `tools/` importable, the set includes its `run_model_tile.py`; standalone use records `lib/fallback/tile_admission.py` under that receipt name. Set `traffic_reuse=True` to include every registered recipe.

The `atlas_program_reuse_*.py` modules are the per-shape traffic-reuse schedules: `tiny` handles TinyLlama prefill K 2048, `tiny_breadth` handles its down projection and decode, `resident` keeps A resident for SmolVLA vision 128x768x768, `stream` uses two-bank A and streamed B for SmolVLA fc2, `smol_connector` handles its connector, and `blocked` uses bank-bounded blocks for SmolVLA text, expert, and ResNet50 shapes. Each exposes `assembly(...)` and `dma_load_bytes(...)`; `atlas_program_reuse_common.py` holds the shared resident-column load and ordered output store.

## Variants and the model shapes they serve

Stage: standalone launch, one per physical panel. `assembly(..., traffic_reuse=True)` dispatches on the exact physical shape through `_reuse_module` to one recipe module; both `assembly` and `dma_load_bytes` raise `ValueError` for an unqualified traffic-reuse shape. Select `traffic_reuse=False` explicitly to use the default schedule. The canonical [catalog in `kernel.yaml`](kernel.yaml) lists every physical panel shape the qualified builds launch: TinyLlama from the default build's 840-panel table, SmolVLA from atlas-mlir `tools/smol_full_plan.py` `panels()` (2,715 panels). ResNet50 rows are the 20 convolution shapes of the blocked admission table (no per-panel census). `lib/tests/test_kernel_manifests.py` checks that each listed shape dispatches to its variant.

The blocked recipe also admits 192x960x320, 192x960x960, 192x960x2560, and 192x2560x960, which no current SmolVLA call uses. Rows split at 128 and pad to multiples of 32 (SmolVLA `panels()`); TinyLlama splits N into panels of at most 2048 columns (5632 as 2048, 2048, and 1536; 32000 as fifteen 2048 panels and one 1280 panel; the down projection's 2048 as two 1024 panels).

## Generate

```sh
make gen-projection_gemm NPU_MODEL=/path/to/npu-model ARGS="--shape 128x2048x2048 --traffic-reuse"
make gen-projection_gemm NPU_MODEL=/path/to/npu-model ARGS="--shape 64x736x32"
```

See [shared generation](../../README.md#generating-a-kernel) for `build/projection_gemm/` artifacts; this family records recipe, offsets, and DMA bytes in JSON.

## Reference and bench results

`reference.py` provides FP32 PyTorch references for the [shared bench](../../lib/README.md): 4 cases (TinyLlama prefill and decode lm_head panels, SmolVLA vision and text q panels), each bit-exact against `lib/golden` `ordered_matmul`. [`RESULTS.md`](RESULTS.md) reports exactness, FP32 fidelity, and model cycles under the [shared bench conventions](../../README.md#model-coverage).

## Tests and checks

- `tests/test_program_reuse_resident.py` proves every recipe symbolically with `tests/reuse_proof.py` (addresses, K order, bank ownership, DMA traffic, output halves) and includes mutation controls that must be detected.
- `tests/test_program_resident.py` checks the default and resident schedules, including frozen SHA-256 digests of the default programs.
- `tests/test_program_checks.py` and `tests/test_program_legacy_epilogue.py` check register admission for every recipe with and without fused hooks.
- `tests/test_program_sources.py` checks source paths and hashes in standalone and external-helper layouts, including changes to the imported helper's bytes and rejection of unqualified traffic-reuse shapes.
- `tests/test_program_reuse_resident.py` assembles a program with npu_model's assembler and needs `NPU_MODEL`.
- In atlas-mlir, `test/test_program_traffic_reuse.py` checks dispatch and the source closure, and the TinyLlama and SmolVLA capture flows execute every production launch on npu_model `47b8599` (`lib/atlas_execution.py`).
