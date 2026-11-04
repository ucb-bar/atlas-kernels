# projection_gemm

FP8 E4M3 matrix multiply `C[M,N] = A[M,K] @ B[K,N]` on MXU0, accumulating in BF16 in ascending K order, with one pop to BF16 and an optional power-of-two output rescale. Every linear layer of the three models (attention projections, MLPs, convolutions as GEMMs, connectors, and heads) runs through this family; a fused [epilogue](../epilogue/README.md) may replace the plain BF16 store.

## Entry points

`atlas_program.py`:

- `assembly(m, k, n, output_offset, rescale_exponent=None, *, resident_b=False, pretranspose_resident_b=False, resident_a=False, traffic_reuse=False, epilogue=None, direct_input=None, weight_source=None)` returns the program text. `m`, `k`, and `n` are positive multiples of 32. The arena holds A at offset 0, B after it, and the output at `output_offset`; `rescale_exponent` scales the BF16 result by `2**exponent`. `resident_a` or `resident_b` keeps one operand in VMEM bank 1 when it fits 256 KiB. `traffic_reuse=True` dispatches to the qualified recipe for the exact shape in the [catalog](kernel.yaml). `direct_input` gathers A rows on device ([resnet_patch_gather](../resnet_patch_gather/README.md)); `weight_source` reads FP8 weights in place from an immutable region (`lib/atlas_weight_source.py`).
- `dma_load_bytes(...)` returns the FP8 input bytes the program requests, for DMA accounting.
- `legacy_epilogue_assembly(...)` reproduces the archived non-Tiny C schedules with their original x30/x31 hook registers.
- `code_sources(traffic_reuse=False)` and `source_paths(...)` name and hash the files whose bytes program receipts pin. The set includes atlas-mlir's `run_model_tile.py`, so these two functions need atlas-mlir's `tools/` importable.

The `atlas_program_reuse_*.py` modules are the per-shape traffic-reuse schedules; each also exposes `assembly(...)` and `dma_load_bytes(...)`, and `atlas_program_reuse_common.py` holds the shared resident-column load and ordered output store.

## Variants and the model shapes they serve

Stage: standalone launch, one per physical panel. `assembly(..., traffic_reuse=True)` dispatches on the exact physical shape through `_reuse_module` to one recipe module; any shape without a recipe uses the default schedule in `atlas_program.py`. The canonical [catalog in `kernel.yaml`](kernel.yaml) lists every physical panel shape the qualified builds launch: TinyLlama from the default build's 840-panel table, SmolVLA from atlas-mlir `tools/smol_full_plan.py` `panels()` (2,715 panels). ResNet50 rows are the 20 convolution shapes of the blocked admission table (no per-panel census). `lib/tests/test_kernel_manifests.py` checks that each listed shape dispatches to its variant.

The blocked recipe also admits 192x960x320, 192x960x960, 192x960x2560, and 192x2560x960, which no current SmolVLA call uses. Rows split at 128 and pad to multiples of 32 (SmolVLA `panels()`); TinyLlama splits N into panels of at most 2048 columns (5632 as 2048, 2048, and 1536; 32000 as fifteen 2048 panels and one 1280 panel; the down projection's 2048 as two 1024 panels).

## Generate

```sh
make gen-projection_gemm NPU_MODEL=/path/to/npu-model ARGS="--shape 128x2048x2048 --traffic-reuse"
make gen-projection_gemm NPU_MODEL=/path/to/npu-model ARGS="--shape 64x736x32"
```

`gen.py` writes the program text (`.S`), the 32-bit words npu_model's assembler produces (`.bin`), and a `.json` record (recipe, offsets, DMA bytes) into `build/projection_gemm/`.

## Reference and bench results

`reference.py` defines the bench cases and their FP32 PyTorch references for `lib/kernel_bench.py`: 4 cases (TinyLlama prefill and decode lm_head panels, SmolVLA vision and text q panels), each bit-exact against `lib/golden` `ordered_matmul`. [`RESULTS.md`](RESULTS.md) lists bit-exactness, cosine similarity and relative error against the FP32 reference, and npu_model model cycles (not RTL cycles) for each case, run on npu_model 47b8599 with `numerics="rtl"`.

## Tests and checks

- `tests/test_program_reuse_*.py` and `tests/test_program_reuse_proof.py` prove every recipe symbolically with `tests/reuse_proof.py` (addresses, K order, bank ownership, DMA traffic, output halves) and include mutation controls that must be detected.
- `tests/test_program_resident.py` checks the default and resident schedules, including frozen SHA-256 digests of the default programs.
- `tests/test_program_checks.py` and `tests/test_program_legacy_epilogue.py` check register admission for every recipe with and without fused hooks.
- `tests/test_program_reuse_resident.py` assembles a program with npu_model's assembler and needs `NPU_MODEL`.
- In atlas-mlir, `test/test_program_traffic_reuse.py` checks dispatch and the source closure, and the TinyLlama and SmolVLA capture flows execute every production launch on npu_model `47b8599` (`lib/atlas_execution.py`).

The shared [per-family make interface](../README.md) provides `gen`, `data`, `test`, and explicitly provisioned `rtl` targets for this family, with generated payloads in ignored output directories and reference gaps recorded as PENDING.
