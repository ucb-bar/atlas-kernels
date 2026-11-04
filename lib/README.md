# lib

Helpers shared by the kernel families, and the references the kernels are checked against. Every module here is imported by bare name after `import atlas_kernels_path`.

| Module | Role |
| --- | --- |
| `atlas_execution.py` | `Executor(model_root)`: assembles a program with npu_model's own assembler and executes it on `ArchState` with `numerics="rtl"`. It requires npu_model at `MODEL_PIN` (`47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`), compares the complete final DRAM arena with the expected bytes, checks requested DMA traffic, and records the SHA-256 of itself and `atlas_path_guard.py` (it must sit next to that file). This is functional execution, not a timing measurement. |
| `atlas_rtl_reference.py` | `RtlReference(arena_base)`: executes scheduled physical instructions one at a time with `numerics="rtl"`, including scalar branch delay slots; used by the ResNet layout-kernel tests. |
| `atlas_program_checks.py` | Static register checks for every contraction entry point: `checked_assembly` (decorator), `check_matrix_registers`, `check_scalar_scratch`, `check_hook_footprint`. |
| `atlas_weight_source.py` | `WeightSource(base, size_bytes, arena_base)`: an immutable FP8 weight region read in place by fast builds, separate from the dynamic arena. |
| `smol_weight_relocate.py` | `relocate_assembly(source, old_base, new_base, *, force=False, weight_source=None)`: moves a program's DMA arena to a new physical base, leaving weight-source transfers in place. |
| `atlas_path_guard.py` | Physical-path admission (no parent traversal, no symbolic links, no restricted components) used before any file read. |
| `atlas_tooling.py` | File identities, JSON receipts, archive preflight, and bounded subprocesses. |
| `kernel_gen.py` | Command-line plumbing for `kernels/<family>/gen.py`: writes `.S`, the npu_model-assembled `.bin`, and a `.json` record. |
| `golden/atlas_tile_arithmetic.py` | Independent rational FP8 E4M3 and BF16 arithmetic: `mxu0_dot` (ordered MXU0 accumulation), `bf16_round`, `rescale_bf16`, `fp8_value`, `tensor_scale_exponent`, and output layout helpers. It imports no model, Torch, or assembler. |
| `golden/atlas_vector_arithmetic.py` | NumPy `ordered_matmul(uint8[M,K], uint8[K,N]) -> uint16[M,N]` and BF16 helpers that equal the rational oracle exactly; the module docstring proves the binary64 intermediates exact. |
| `fallback/` | Stand-ins for the two atlas-mlir host helpers that unmodified generators import (`run_model_tile.validate_archive_members`, `chipyard.arena.relocate_assembly`). `atlas_kernels_path` serves them only when atlas-mlir's real modules are not importable; atlas-mlir's `test/test_kernel_library.py` checks they match. |

## Tests

`tests/test_vector_arithmetic.py` compares the NumPy oracle with the rational one on all finite FP8 codes, pair products, BF16 decodes, ties, and random ordered chains. `tests/test_atlas_path_guard.py` checks path admission. `tests/test_library_layout.py` checks that module basenames are unique and that nothing earlier on `sys.path` shadows them. `tests/test_kernel_manifests.py` checks every family's README, `kernel.yaml` (entry points exist; projection variants are the recipes `atlas_program` dispatches each listed shape to), and generator.
