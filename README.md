# Atlas Kernels

Device-kernel generators for the Atlas NPU, with the shared helpers and golden references that check them. Each generator is a Python module that emits Atlas assembly text for one launch; npu_model's assembler encodes it, and programs are checked bit-exactly against unmodified npu_model `47b8599` with `numerics="rtl"`. The library holds every kernel that the TinyLlama, SmolVLA, and ResNet50 bringups launch. [atlas-mlir](https://github.com/ucb-bar/atlas-mlir) pins it as the `third_party/atlas-kernels` submodule and builds host programs, captures, and ELFs on top of it.

The generators are hand-written schedules in Python; no kernel is lowered from the Atlas MLIR dialect yet. `lib/atlas_backend.py` can encode their assembly through the Atlas compiler backend instead (machine IR, `atlas-opt` verifiers, `atlas-emit`) and refuses any program whose words differ from the assembler's. Host graphs (the C that sequences launches and runs host-side operators) live in atlas-mlir and Merlin, not here.

## How this fits with atlas-mlir and Merlin

This repository owns the Atlas Python kernel generators in `kernels/<family>/`, shared helpers in `lib/`, and golden references in `lib/golden/`. [atlas-mlir](https://github.com/ucb-bar/atlas-mlir/tree/sota/bringup-docs) pins it as the `third_party/atlas-kernels` submodule and retains the compiler, model bringup tooling, and host entry points. Merlin's `examples/atlas/comparisons/` retains the PyTorch captures, hand-written model C graphs, and comparison workflow. See the [model bringup guide](https://github.com/ucb-bar/atlas-mlir/blob/sota/bringup-docs/docs/bringing-up-a-model.md) for the end-to-end flow and its current limits.

The [model coverage table](coverage/COVERAGE.md) maps exported TinyLlama and SmolVLA ops and shapes to kernel families, host code, or build-time constants. Its `coverage/mapping.yaml` records placement in the qualified builds; it is not an automatic kernel selector. The `select` rules in each family's `kernel.yaml` describe which family and variant serve an op, dtype, and shape, but nothing reads them automatically yet either. The model kernels emit Atlas assembly text that npu_model's assembler encodes by default (or the compiler backend, through `lib/atlas_backend.py run`), and the host graphs remain hand-written C.

## Layout

```
atlas_kernels_path.py   puts lib/, lib/golden/, and every kernels/<family>/ on sys.path; module_path(name)
Makefile                make test, make test-<family>, make test-lib, make gen-<family> ARGS="..."
lib/                    shared helpers: npu_model runner, register checks, weight source, relocation, gen plumbing
lib/golden/             independent numeric references (rational and NumPy)
lib/fallback/           stand-ins for two atlas-mlir helpers, used only when this repository runs on its own
lib/tests/
kernels/<family>/       README.md, kernel.yaml (manifest), gen.py (generator CLI), generator modules, tests/
coverage/               per-model coverage: which kernel runs every PyTorch op (COVERAGE.md)
```

| Family | Stage | Computes |
| --- | --- | --- |
| [`projection_gemm`](kernels/projection_gemm/README.md) | standalone launch | FP8 E4M3 GEMM on MXU0 with ascending-K BF16 accumulation and a power-of-two output rescale, plus per-shape traffic-reuse schedules |
| [`epilogue`](kernels/epilogue/README.md) | fused into projection launches | ordered BF16 tile epilogues at the projection store (add, exact add, mul, ReLU, exp, reciprocal, SiLU, tanh GELU, connector scale, FP8 pack) |
| [`rne_add`](kernels/rne_add/README.md) | epilogue step | BF16 add rounded to nearest even from truncating VPU arithmetic |
| [`rmsnorm`](kernels/rmsnorm/README.md) | post-pass (a standalone launch in TinyLlama) | RMSNorm (and LayerNorm) over retained rows, FP8 output |
| [`rope`](kernels/rope/README.md) | post-pass on q and k projections | rotate-half rotary embedding |
| [`attention`](kernels/attention/README.md) | standalone launch | head-dimension-64 attention: QK^T, masked softmax, and PV |
| [`smol_fusion`](kernels/smol_fusion/README.md) | projection launch with fused epilogue | SmolVLA panels with fused bias, tanh GELU, connector scale, and FP8 output |
| [`resnet_pool`](kernels/resnet_pool/README.md) | standalone launch | 3x3 stride-two max pooling of FP8 channels-last tensors |
| [`resnet_patch_gather`](kernels/resnet_patch_gather/README.md) | A-operand source of a projection launch | device row gathers (1x1 and kxk im2col) into tiled MXU A operands |

[`lib/README.md`](lib/README.md) describes the shared modules.

### Module names

The generator modules keep the file names they had in atlas-mlir `tools/`, because model receipts and source pins name them and require their exact bytes. This table maps each name to its role.

| Module | Family | Role |
| --- | --- | --- |
| `atlas_program.py` | projection_gemm | GEMM entry point: `assembly`, `dma_load_bytes`, recipe dispatch, legacy hook entry point |
| `atlas_program_reuse_common.py` | projection_gemm | shared resident-column load and ordered BF16 output store |
| `atlas_program_reuse_tiny.py` | projection_gemm | TinyLlama prefill K 2048 recipe |
| `atlas_program_reuse_tiny_breadth.py` | projection_gemm | TinyLlama down projection and decode recipe |
| `atlas_program_reuse_resident.py` | projection_gemm | resident-A recipe (SmolVLA vision 128x768x768) |
| `atlas_program_reuse_stream.py` | projection_gemm | two-bank A, streamed B recipe (SmolVLA fc2) |
| `atlas_program_reuse_smol_connector.py` | projection_gemm | SmolVLA connector recipe |
| `atlas_program_reuse_blocked.py` | projection_gemm | bank-bounded blocked recipe (SmolVLA text, expert, and ResNet50 shapes) |
| `atlas_epilogue.py` | epilogue | `Epilogue`, `Step`, `TileInput` |
| `atlas_resident_output.py` | epilogue | BF16 output retention for post-passes |
| `atlas_rne_add.py` | rne_add | exact BF16 add sequence |
| `atlas_rmsnorm_postpass.py`, `atlas_norm_postpass.py` | rmsnorm | RMSNorm prototype and its RMS/LayerNorm extension |
| `atlas_rope_postpass.py` | rope | RoPE post-pass |
| `atlas_attention_v2_program.py`, `atlas_attention_program.py` | attention | attention v2 (current) and v1 |
| `smol_fusion_program.py`, `smol_fusion_fp8.py` | smol_fusion | fused SmolVLA panels and the FP8 fc1 producer |
| `atlas_resnet_pool.py` | resnet_pool | max pool |
| `atlas_resnet_input.py` | resnet_patch_gather | direct-input row gathers |
| `atlas_execution.py`, `atlas_rtl_reference.py` | lib | npu_model execution |
| `atlas_program_checks.py`, `atlas_weight_source.py`, `smol_weight_relocate.py`, `atlas_path_guard.py`, `atlas_tooling.py`, `kernel_gen.py` | lib | checks, weight source, relocation, path admission, receipts, generator plumbing |
| `atlas_tile_arithmetic.py`, `atlas_vector_arithmetic.py` | lib/golden | rational and NumPy references |

## Running the tests

The tests need Python 3.11 or newer with NumPy and PyTorch (PyYAML too for the `kernel.yaml` checks, which skip without it), and an npu_model checkout at `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` for the execution checks:

```sh
make test PYTHON=/path/to/venv/bin/python NPU_MODEL=/path/to/npu-model
make test-projection_gemm PYTHON=/path/to/venv/bin/python NPU_MODEL=/path/to/npu-model
```

`NPU_MODEL` is put on `PYTHONPATH` and exported as `ATLAS_NPU_MODEL_ROOT`. Without it, the tests that execute programs on npu_model skip (or, for one resident-reuse test, fail on the missing import), and the static tests still run. Each suite also runs directly with `python -m unittest discover -s kernels/<family>/tests -t kernels/<family>/tests -v`. atlas-mlir registers every test module here as a CTest test named `atlas-kernels-<family>-<module>`.

## Generating a kernel

Each family has a generator command line, `kernels/<family>/gen.py`, run through make:

```sh
make gen-projection_gemm PYTHON=/path/to/venv/bin/python NPU_MODEL=/path/to/npu-model ARGS="--shape 128x2048x2048 --traffic-reuse"
make gen-attention PYTHON=/path/to/venv/bin/python NPU_MODEL=/path/to/npu-model ARGS="--rows 128 --keys 128 --heads 32/4 --causal --fp8"
```

It writes the program text (`.S`), the 32-bit words npu_model's assembler produces (`.bin`, little endian), and a `.json` record into `build/<family>/`, which git ignores. `python kernels/<family>/gen.py --help` lists the arguments. Each family's `kernel.yaml` is a machine-readable manifest (entry points, ops, dtypes, shape constraints, stage, the models and shapes that use it, and `select` rules in the op, dtype, and shape terms of Merlin's radiance `kernel_library/selection.yaml`).

## How kernels are checked against npu_model

Kernels are checked at three levels, and their encoding can be checked by a fourth.

1. Static checks at generation time. Contraction entry points are wrapped by `lib/atlas_program_checks.py` `checked_assembly`, which rejects odd matrix-register pairs, bank aliases, and scalar-scratch collisions between a schedule and its fused epilogue. Attention v2 programs pass `check_program_v2`.
2. Symbolic and golden checks in the tests here. The reuse proofs (`kernels/projection_gemm/tests/reuse_proof.py`) interpret every DMA, wait, and MXU operation of a schedule and check addresses, K order, bank ownership, and output placement. `lib/golden/` provides an exact rational FP8/BF16 tile oracle and an equivalent NumPy ordered matmul for whole-matrix references. Frozen SHA-256 digests guard the default programs against unintended change.
3. Execution on npu_model. `lib/atlas_execution.py` `Executor` assembles a program with npu_model's own assembler, runs it on `ArchState` with `numerics="rtl"`, refuses any npu_model revision other than `47b8599`, and compares the complete final DRAM arena (the output and everything outside it) with the expected bytes. `lib/atlas_rtl_reference.py` executes instruction by instruction with `numerics="rtl"`, including scalar branch delay slots; the ResNet layout-kernel tests use it. atlas-mlir's capture and export flows run model launches through `Executor` on real captured inputs (for TinyLlama, `tools/tiny_full/export_capture.py --qualify-launches` checks every launch).
4. Encoding through the compiler backend. `lib/atlas_backend.py` translates each program to Atlas machine IR, runs `atlas-opt --verify-atlas-machine-stream` and `--verify-atlas-timing`, encodes it with `atlas-emit`, and requires the words to equal npu_model's assembler words. Each program gets a tier: `verified` (both verifiers accept it), `stream-verified` (the machine-stream verifier accepts it and the timing verifier rejects it), or `encode-only` (no verifier accepted it or verification was skipped). For the 1,589 distinct programs the TinyLlama, SmolVLA, and ResNet50 kit ELFs launch, all encode to the assembler's words: 13 are `verified`, 1,576 are `stream-verified` (the timing verifier cannot prove double-buffered DMA ranges disjoint and times one block at a time), and 0 are `encode-only`. [`lib/README.md`](lib/README.md) has the details and commands.

## Using the library from atlas-mlir

atlas-mlir imports these modules by bare name, exactly as it did when they lived in its `tools/` directory. Its `tools/kernel_library.py` puts this submodule on `sys.path` and imports `atlas_kernels_path`; every atlas-mlir file that imports a kernel imports `kernel_library` first. Module basenames are unique across the library (`lib/tests/test_library_layout.py` checks it), so `atlas_kernels_path.module_path("atlas_program.py")` names the one file that defines a module, and atlas-mlir resolves source pins through `kernel_library.source_path` and `kernel_library.code_sources`.

The generator files are byte-identical to the atlas-mlir sources that model receipts pin, so two of them still import atlas-mlir host helpers (`atlas_program` imports `run_model_tile.validate_archive_members`; the Smol fusion generators import `chipyard.arena.relocate_assembly`). Inside atlas-mlir the real modules load. When this repository runs on its own, `atlas_kernels_path` serves verbatim stand-ins from `lib/fallback/`, through a finder consulted only after every normal import has failed.

## Adding a kernel

1. Create `kernels/<family>/` with the generator module (a basename not used anywhere else in the library), a `README.md`, a `kernel.yaml`, and a `gen.py` in the form of the existing ones, and `tests/test_<name>.py`. `atlas_kernels_path.py` and the Makefile pick the directory up without edits; add the family and a `gen.py` argument set to `lib/tests/test_kernel_manifests.py`.
2. Emit Atlas assembly text that npu_model's assembler accepts. Wrap contraction-like entry points with `checked_assembly`, or add an equivalent register check.
3. Add a NumPy or rational reference (in `lib/golden/` when other kernels can share it) and a test that executes the program through `Executor` against npu_model `47b8599` and compares the whole arena with the reference.
4. Run `make test`, then bump the submodule pin in atlas-mlir and wire the kernel into the model's host graph there.

## Model coverage

[`coverage/COVERAGE.md`](coverage/COVERAGE.md) lists, for TinyLlama and SmolVLA, every operator `torch.export` finds at every shape and where it runs: on an Atlas kernel family here, in host C, or as a build-time constant, with FLOP and launch counts checked against the qualified builds. It is the basis for kernel ablation and optimization work.

## License

Apache License 2.0; see [LICENSE](LICENSE).
