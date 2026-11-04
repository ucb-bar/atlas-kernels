# projection_gemm

FP8 E4M3 matrix multiply `C[M,N] = A[M,K] @ B[K,N]` on MXU0, accumulating in BF16 in ascending K order, with one pop to BF16 and an optional power-of-two output rescale. Every linear layer of the three models (attention projections, MLPs, convolutions as GEMMs, connectors, and heads) runs through this family; a fused [epilogue](../epilogue/README.md) may replace the plain BF16 store.

## Entry points

`atlas_program.py`:

- `assembly(m, k, n, output_offset, rescale_exponent=None, *, resident_b=False, pretranspose_resident_b=False, resident_a=False, traffic_reuse=False, epilogue=None, direct_input=None, weight_source=None)` returns the program text. `m`, `k`, and `n` are positive multiples of 32. The arena holds A at offset 0, B after it, and the output at `output_offset`; `rescale_exponent` scales the BF16 result by `2**exponent`. `resident_a` or `resident_b` keeps one operand in VMEM bank 1 when it fits 256 KiB. `traffic_reuse=True` dispatches to the qualified recipe for the exact shape (table below). `direct_input` gathers A rows on device ([resnet_patch_gather](../resnet_patch_gather/README.md)); `weight_source` reads FP8 weights in place from an immutable region (`lib/atlas_weight_source.py`).
- `dma_load_bytes(...)` returns the FP8 input bytes the program requests, for DMA accounting.
- `legacy_epilogue_assembly(...)` reproduces the archived non-Tiny C schedules with their original x30/x31 hook registers.
- `code_sources(traffic_reuse=False)` and `source_paths(...)` name and hash the files whose bytes program receipts pin. The set includes atlas-mlir's `run_model_tile.py`, so these two functions need atlas-mlir's `tools/` importable.

The `atlas_program_reuse_*.py` modules are the per-shape traffic-reuse schedules; each also exposes `assembly(...)` and `dma_load_bytes(...)`, and `atlas_program_reuse_common.py` holds the shared resident-column load and ordered output store.

## Variants and the model shapes they serve

Stage: standalone launch, one per physical panel. `assembly(..., traffic_reuse=True)` dispatches on the exact physical shape through `_reuse_module` to one recipe module; any shape without a recipe uses the default schedule in `atlas_program.py`. The table lists every physical panel shape the qualified builds launch: TinyLlama from the default build's 840-panel table, SmolVLA from atlas-mlir `tools/smol_full_plan.py` `panels()` (2,715 panels). ResNet50 rows are the 20 convolution shapes of the blocked admission table (no per-panel census). `kernel.yaml` holds the same table; `lib/tests/test_kernel_manifests.py` checks that each listed shape dispatches to its variant.

| Variant (`_reuse_module`) | Model | Phase | Physical MxKxN | Panels | Modules |
| --- | --- | --- | --- | --- | --- |
| `atlas_program_reuse_tiny` | tinyllama | prefill | 128x2048x256 | 44 | k_proj, v_proj |
| `atlas_program_reuse_tiny` | tinyllama | prefill | 128x2048x1280 | 1 | lm_head |
| `atlas_program_reuse_tiny` | tinyllama | prefill | 128x2048x1536 | 44 | gate_proj, up_proj |
| `atlas_program_reuse_tiny` | tinyllama | prefill | 128x2048x2048 | 44 | q_proj, o_proj |
| `atlas_program_reuse_tiny` | tinyllama | prefill | 128x2048x2048 | 88 | gate_proj, up_proj |
| `atlas_program_reuse_tiny` | tinyllama | prefill | 128x2048x2048 | 15 | lm_head |
| `atlas_program_reuse_tiny_breadth` | tinyllama | decode | 32x2048x256 | 88 | k_proj, v_proj |
| `atlas_program_reuse_tiny_breadth` | tinyllama | decode | 32x2048x1280 | 2 | lm_head |
| `atlas_program_reuse_tiny_breadth` | tinyllama | decode | 32x2048x1536 | 88 | gate_proj, up_proj |
| `atlas_program_reuse_tiny_breadth` | tinyllama | decode | 32x2048x2048 | 88 | q_proj, o_proj |
| `atlas_program_reuse_tiny_breadth` | tinyllama | decode | 32x2048x2048 | 176 | gate_proj, up_proj |
| `atlas_program_reuse_tiny_breadth` | tinyllama | decode | 32x2048x2048 | 30 | lm_head |
| `atlas_program_reuse_tiny_breadth` | tinyllama | decode | 32x5632x1024 | 88 | down_proj |
| `atlas_program_reuse_tiny_breadth` | tinyllama | prefill | 128x5632x1024 | 44 | down_proj |
| `atlas_program_reuse_resident` | smolvla |  | 128x768x768 | 784 | vlm.model.vision_model.embeddings.patch_embedding, vlm.model.vision_model.encoder.layers.*.self_attn.k_proj, vlm.model.vision_model.encoder.layers.*.self_attn.out_proj, vlm.model.vision_model.encoder.layers.*.self_attn.q_proj, vlm.model.vision_model.encoder.layers.*.self_attn.v_proj |
| `atlas_program_reuse_stream` | smolvla |  | 128x3072x768 | 192 | vlm.model.vision_model.encoder.layers.*.mlp.fc2 |
| `atlas_program_reuse_smol_connector` | smolvla |  | 64x12288x960 | 2 | vlm.model.connector.modality_projection.proj |
| `atlas_program_reuse_blocked` | smolvla |  | 128x768x3072 | 192 | vlm.model.vision_model.encoder.layers.*.mlp.fc1 |
| `atlas_program_reuse_blocked` | smolvla |  | 32x32x960 | 1 | state_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 128x960x960 | 32 | vlm.model.text_model.layers.*.self_attn.o_proj, vlm.model.text_model.layers.*.self_attn.q_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 64x960x960 | 32 | vlm.model.text_model.layers.*.self_attn.o_proj, vlm.model.text_model.layers.*.self_attn.q_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 128x960x320 | 32 | vlm.model.text_model.layers.*.self_attn.k_proj, vlm.model.text_model.layers.*.self_attn.v_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 64x960x320 | 32 | vlm.model.text_model.layers.*.self_attn.k_proj, vlm.model.text_model.layers.*.self_attn.v_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 128x960x2560 | 32 | vlm.model.text_model.layers.*.mlp.gate_proj, vlm.model.text_model.layers.*.mlp.up_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 64x960x2560 | 32 | vlm.model.text_model.layers.*.mlp.gate_proj, vlm.model.text_model.layers.*.mlp.up_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 128x2560x960 | 16 | vlm.model.text_model.layers.*.mlp.down_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 64x2560x960 | 16 | vlm.model.text_model.layers.*.mlp.down_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 64x32x736 | 10 | action_in_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 64x1440x736 | 10 | action_time_mlp_in |
| `atlas_program_reuse_blocked` | smolvla |  | 64x736x736 | 10 | action_time_mlp_out |
| `atlas_program_reuse_blocked` | smolvla |  | 64x736x960 | 160 | lm_expert.layers.*.self_attn.q_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 64x736x320 | 160 | lm_expert.layers.*.self_attn.k_proj, lm_expert.layers.*.self_attn.v_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 64x960x736 | 160 | lm_expert.layers.*.self_attn.o_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 64x736x2048 | 320 | lm_expert.layers.*.mlp.gate_proj, lm_expert.layers.*.mlp.up_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 64x2048x736 | 160 | lm_expert.layers.*.mlp.down_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 128x320x320 | 160 | lm_expert.layers.*.self_attn.k_proj, lm_expert.layers.*.self_attn.v_proj |
| `atlas_program_reuse_blocked` | smolvla |  | 64x320x320 | 160 | lm_expert.layers.*.self_attn.k_proj, lm_expert.layers.*.self_attn.v_proj |
| `atlas_program_reuse_blocked` | resnet50 |  | 224x2304x256 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 64x4608x512 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 800x1152x128 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 224x256x1024 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 3136x576x64 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 3136x64x256 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 224x1024x256 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 800x128x512 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 64x512x2048 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 800x512x128 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 12544x160x64 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 64x2048x512 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 64x1024x2048 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 224x512x1024 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 224x1024x512 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 800x256x512 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 3136x256x64 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 3136x256x128 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 800x512x256 |  | convolution GEMM |
| `atlas_program_reuse_blocked` | resnet50 |  | 3136x64x64 |  | convolution GEMM |
| `atlas_program` | smolvla |  | 64x736x32 | 10 | action_out_proj |

The blocked recipe also admits 192x960x320, 192x960x960, 192x960x2560, and 192x2560x960, which no current SmolVLA call uses. Rows split at 128 and pad to multiples of 32 (SmolVLA `panels()`); TinyLlama splits N into panels of at most 2048 columns (5632 as 2048, 2048, and 1536; 32000 as fifteen 2048 panels and one 1280 panel; the down projection's 2048 as two 1024 panels).

## Generate

```sh
make gen-projection_gemm NPU_MODEL=/path/to/npu-model ARGS="--shape 128x2048x2048 --traffic-reuse"
make gen-projection_gemm NPU_MODEL=/path/to/npu-model ARGS="--shape 64x736x32"
```

`gen.py` writes the program text (`.S`), the 32-bit words npu_model's assembler produces (`.bin`), and a `.json` record (recipe, offsets, DMA bytes) into `build/projection_gemm/`.

## Tests and checks

- `tests/test_program_reuse_*.py` and `tests/test_program_reuse_proof.py` prove every recipe symbolically with `tests/reuse_proof.py` (addresses, K order, bank ownership, DMA traffic, output halves) and include mutation controls that must be detected.
- `tests/test_program_resident.py` checks the default and resident schedules, including frozen SHA-256 digests of the default programs.
- `tests/test_program_checks.py` and `tests/test_program_legacy_epilogue.py` check register admission for every recipe with and without fused hooks.
- `tests/test_program_reuse_resident.py` assembles a program with npu_model's assembler and needs `NPU_MODEL`.
- In atlas-mlir, `test/test_program_traffic_reuse.py` checks dispatch and the source closure, and the TinyLlama and SmolVLA capture flows execute every production launch on npu_model `47b8599` (`lib/atlas_execution.py`).
