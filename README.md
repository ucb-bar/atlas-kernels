# Atlas Kernels

Kernels written for the [Atlas NPU](https://github.com/ucb-bar/atlas-npu).

Python generators emit Atlas assembly for the TinyLlama, SmolVLA, and ResNet50 bringups. [atlas-mlir](https://github.com/ucb-bar/atlas-mlir) pins this library at `third_party/atlas-kernels` and builds host programs, captures, and ELFs.

Schedules are hand-written Python, and host graphs are hand-written C in atlas-mlir and Merlin. No kernel is lowered from the Atlas MLIR dialect yet. The default encoder is npu_model's assembler; the optional [compiler comparison gate](lib/README.md#compiler-backend-atlas_asm_mlirpy-atlas_backendpy) enforces identical words and returns the assembler-produced program.

## How this fits with atlas-mlir and Merlin

This repository owns the Atlas Python kernel generators in `kernels/<family>/`, shared helpers in `lib/`, and golden references in `lib/golden/`. [atlas-mlir](https://github.com/ucb-bar/atlas-mlir/tree/sota/bringup-performance) (PR [#26](https://github.com/ucb-bar/atlas-mlir/pull/26)) pins it as the `third_party/atlas-kernels` submodule and retains the compiler, model bringup tooling, and host entry points. Merlin's [`examples/atlas/comparisons/`](https://github.com/reed-nicolas/merlin/tree/atlas/sota-bringup-evidence/examples/atlas/comparisons) (PR [#55](https://github.com/ucb-bar/merlin/pull/55)) retains the PyTorch captures, hand-written model C graphs, and comparison workflow. See the [model bringup guide](https://github.com/ucb-bar/atlas-mlir/blob/sota/bringup-performance/docs/bringing-up-a-model.md) for the end-to-end flow and its current limits.

[Model coverage](coverage/README.md) maps exported ops and shapes to authored placement, with [generated tables](coverage/COVERAGE.md). It also explains the family `kernel.yaml` selection rules, Merlin registry, and limits of their checks.

## Layout

```
atlas_kernels_path.py   puts lib/, lib/golden/, and every kernels/<family>/ on sys.path; module_path(name)
Makefile                make test, make test-<family>, make test-lib, make gen-<family> ARGS="..."
lib/                    shared helpers: npu_model runner, register checks, weight source, relocation, gen plumbing
lib/golden/             independent numeric references (rational and NumPy)
lib/fallback/           stand-ins for two atlas-mlir helpers, used only when this repository runs on its own
lib/tests/
kernels/<family>/       README.md, kernel.yaml (manifest), gen.py (generator CLI), reference.py and RESULTS.md (bench), modules, tests/
coverage/               per-model coverage (COVERAGE.md), the standalone bench table (RESULTS.md), the model bench (bench_model.py), and examples/
```

| Family | Stage | Computes |
| --- | --- | --- |
| [`projection_gemm`](kernels/projection_gemm/README.md) | standalone launch | FP8 E4M3 GEMM on MXU0 with ascending-K BF16 accumulation and a power-of-two output rescale, plus per-shape traffic-reuse schedules |
| [`epilogue`](kernels/epilogue/README.md) | fused into projection launches | ordered BF16 tile epilogues at the projection store (add, exact add, mul, ReLU, exp, reciprocal, SiLU, tanh GELU, connector scale, FP8 pack) |
| [`rne_add`](kernels/rne_add/README.md) | epilogue step | BF16 add rounded to nearest even from truncating VPU arithmetic |
| [`rmsnorm`](kernels/rmsnorm/README.md) | post-pass (a standalone launch in TinyLlama) | RMSNorm (and LayerNorm) over retained rows, FP8 output |
| [`rope`](kernels/rope/README.md) | post-pass on q and k projections | rotate-half rotary embedding |
| [`attention`](kernels/attention/README.md) | standalone launch | head-dimension-64/128/256 attention: QK^T, masked softmax, and PV |
| [`smol_fusion`](kernels/smol_fusion/README.md) | projection launch with fused epilogue | SmolVLA panels with fused bias, tanh GELU, connector scale, and FP8 output |
| [`resnet_pool`](kernels/resnet_pool/README.md) | standalone launch | 3x3 stride-two max pooling of FP8 channels-last tensors |
| [`resnet_patch_gather`](kernels/resnet_patch_gather/README.md) | A-operand source of a projection launch | device row gathers (1x1 and kxk im2col) into tiled MXU A operands |

[`lib/README.md`](lib/README.md) describes the shared modules.

### Module names

Generator modules retain their atlas-mlir `tools/` names for bare-name imports. Module basenames must be unique.

## Running the tests

The tests need Python 3.14, the floor that npu_model `47b8599` declares, with NumPy and PyTorch (PyYAML too for the `kernel.yaml` checks, which skip without it), and an npu_model checkout at `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` for the execution checks. npu_model is at `git@github.com:ucb-ee194-tapeout/npu_model.git`, which needs access to the ucb-ee194-tapeout GitHub organization.

```sh
make test PYTHON=/path/to/venv/bin/python NPU_MODEL=/path/to/npu-model
make test-projection_gemm PYTHON=/path/to/venv/bin/python NPU_MODEL=/path/to/npu-model
```

`NPU_MODEL` is put on `PYTHONPATH` and exported as `ATLAS_NPU_MODEL_ROOT`. Without it, the tests that execute programs on npu_model skip (or, for one resident-reuse test, fail on the missing import), and the static tests still run. The five local suites are `attention`, `projection_gemm`, `rmsnorm`, `resnet_pool`, and `resnet_patch_gather`; each also runs directly with `python -m unittest discover -s kernels/<family>/tests -t kernels/<family>/tests -v`. All nine families share manifest and generator checks in `lib/tests/test_kernel_manifests.py` (`make test-lib`), and launch-plan checks live in `coverage/tests/test_bench_model.py` (`make test-coverage`). Epilogue, RNE add, RoPE, and Smol fusion have no local test suite; their READMEs and manifests name the shared suites and external checks that apply. atlas-mlir registers kernel-family and lib modules as `atlas-kernels-<suite>-<module>` CTests; its configured faithfulness target separately runs pinned coverage-table and launch-plan checks.

`lib/tests/test_kernel_bench.py` checks bench codecs and builds each reference case. The projection suite covers shared register admission and legacy epilogue hooks. These shared checks do not establish full-model numerical equivalence.

`make test-cites`, also included in `make test`, checks coverage citations and host/build destinations against adjacent atlas-mlir and Merlin checkouts by default. Set `ATLAS_MLIR_ROOT` and `MERLIN_ROOT` for other locations. Explicit missing roots or `CHECK_CITES=1` fail when the required checkouts are unavailable; without configured or adjacent roots, the target reports that citation checks are unavailable.

## Generating a kernel

Each family has a generator command line, `kernels/<family>/gen.py`, run through make. The [shared per-family interface](kernels/README.md) documents representative `gen`, `data`, `test`, and explicitly provisioned `rtl` targets, ignored outputs, and PENDING reference gaps:

```sh
make gen-projection_gemm PYTHON=/path/to/venv/bin/python NPU_MODEL=/path/to/npu-model ARGS="--shape 128x2048x2048 --traffic-reuse"
make gen-attention PYTHON=/path/to/venv/bin/python NPU_MODEL=/path/to/npu-model ARGS="--rows 128 --keys 128 --heads 32/4 --causal --fp8"
```

It writes the program text (`.S`), the 32-bit words npu_model's assembler produces (`.bin`, little endian), and a `.json` record into `build/<family>/`, which git ignores. `python kernels/<family>/gen.py --help` lists the arguments. Each family's `kernel.yaml` is a machine-readable manifest (entry points, ops, dtypes, shape constraints, stage, the models and shapes that use it, and `select` rules in the op, dtype, and shape terms of Merlin's radiance `kernel_library/selection.yaml`).

## How kernels are checked against npu_model

Checks combine static register admission, symbolic DMA/MXU schedule proofs, independent rational and NumPy references, frozen default-program digests, and execution against unmodified npu_model `47b8599` with `numerics="rtl"`. Execution compares the entire DRAM arena, including guards. Attention v2 uses `check_program_v2`; contraction entry points use `checked_assembly`. The reuse interpreter checks addresses, K order, banks, and output placement. atlas-mlir capture/export flows check real model inputs; TinyLlama uses `tools/tiny_full/export_capture.py --qualify-launches`.

See [shared helpers and backend tiers](lib/README.md) for execution APIs, compiler comparison commands, verifier census, and conditional contracts. See [family guides](#layout) for numerical scope and missing goldens.

## Using the library from atlas-mlir

atlas-mlir imports these modules by bare name, exactly as it did when they lived in its `tools/` directory. Its `tools/kernel_library.py` puts this submodule on `sys.path` and imports `atlas_kernels_path`; every atlas-mlir file that imports a kernel imports `kernel_library` first. Module basenames are unique across the library (`lib/tests/test_library_layout.py` checks it), so `atlas_kernels_path.module_path("atlas_program.py")` names the one file that defines a module, and atlas-mlir resolves source pins through `kernel_library.source_path` and `kernel_library.code_sources`.

Integration evidence covers atlas-kernels `6123bef` at atlas-mlir gate `0c0edf7`; later edits need fresh qualification. See [fallback helpers](lib/README.md) for standalone imports.

## Adding a kernel

1. Create `kernels/<family>/` with the generator module (a basename not used anywhere else in the library), a `README.md`, a `kernel.yaml`, and a `gen.py` in the form of the existing ones, and `tests/test_<name>.py`. `atlas_kernels_path.py` and the Makefile pick the directory up without edits; add the family and a `gen.py` argument set to `lib/tests/test_kernel_manifests.py`.
2. Emit Atlas assembly text that npu_model's assembler accepts. Wrap contraction-like entry points with `checked_assembly`, or add an equivalent register check.
3. Add a NumPy or rational reference (in `lib/golden/` when other kernels can share it) and a test that executes the program through `Executor` against npu_model `47b8599` and compares the whole arena with the reference.
4. Run `make test`, then bump the submodule pin in atlas-mlir and wire the kernel into the model's host graph there.

## Model coverage

[Coverage policy and regeneration](coverage/README.md) explain exported ops, authored placement, launch-count checks, and ablation data. `make coverage` regenerates the table.

[Per-kernel benchmarks](coverage/RESULTS.md) use representative shapes and seeded inputs, independent goldens where available, FP32 fidelity, and npu_model Core cycles. These are model cycles. [RTL results](coverage/RTL-RESULTS.md) and the [RTL workflow](RTL.md) report standalone AtlasCore measurements and explicitly label reduced shapes.

## Starting a new model

For partial random-weight Gemma 2 and PI0 workflows, see the [starter guides](coverage/examples/STARTERS.md).

Start with a PyTorch model, this repository, the Atlas compiler binaries, and unmodified npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`. Clone npu_model with `git clone git@github.com:ucb-ee194-tapeout/npu_model.git`, then check out that commit. This requires access to the ucb-ee194-tapeout GitHub organization. `EXPORT_PYTHON` runs `torch.export` and needs the model's PyTorch dependencies and NumPy; the recorded TinyLlama export uses Python 3.10.14 and Transformers 5.4.0. `MODEL_PYTHON` runs the generators and npu_model, and must be Python >=3.14 with torch, NumPy, and npu_model available. Keep these interpreters separate when their dependencies require different Python versions. The causal-decoder helper accepts `TORCH_PYTHON` as a compatibility fallback for either unset interpreter; it is suitable only when that interpreter meets both sets of requirements. The driver `python3` needs PyYAML for `build_table.py`.

```sh
export EXPORT_PYTHON=/path/to/export-venv/bin/python
export MODEL_PYTHON=/path/to/model-venv/bin/python
export NPU_MODEL=/path/to/npu-model
export ATLAS_MLIR_BIN=/path/to/atlas-mlir/build/bin
export OUT=/path/to/output
# 1. Export every ATen op and shape; the loader defines get_model_and_inputs() -> (torch.nn.Module, tuple_of_inputs).
"$EXPORT_PYTHON" coverage/export_ops.py mymodel --loader path/to/loader.py --out "$OUT/export" --fixture "$OUT/mymodel.json"
# 2. Copy coverage/examples/causal_decoder/mapping.yaml to mymodel-mapping.yaml and adapt its rules; unmapped ops retain their FLOPs.
python3 coverage/build_table.py --inventory mymodel="$OUT/mymodel.json" --mapping mymodel-mapping.yaml --allow-unmapped --out "$OUT/table"
# 3. Generate, compare compiler encoding with assembler words, and run every distinct Atlas launch at the model's shapes.
PYTHONPATH="$NPU_MODEL" ATLAS_NPU_MODEL_ROOT="$NPU_MODEL" "$MODEL_PYTHON" coverage/bench_model.py --coverage "$OUT/table/coverage.json" --model-root "$NPU_MODEL" --out "$OUT/bench" --workers 24 --atlas-mlir-bin "$ATLAS_MLIR_BIN" --min-tier stream-verified
# 4. Read the table.
less "$OUT/bench/MODEL-BENCH.md"
```

The bench uses family `gen.py` entry points, the [compiler gate](lib/README.md#tiers), and npu_model execution. `--min-tier stream-verified` requires stream verification while retaining timing rejections in the report.

Start the mapping from [`coverage/examples/causal_decoder/mapping.yaml`](coverage/examples/causal_decoder/mapping.yaml), which maps Merlin's small `examples/workloads/causal_decoder` loader: Linear layers to `projection_gemm`, RMSNorm to `rmsnorm`, SiLU and the gate multiply to `epilogue`, and residual adds to `rne_add`, each with the reason it fits, and leaves its head-dimension-8 attention unmapped. Each rule's optional `launch` hint (documented at the top of [`coverage/mapping.yaml`](coverage/mapping.yaml)) tells `bench_model.py` which launches the rule's rows become; Linear and Conv2d rows need none. [`coverage/examples/causal_decoder/MODEL-BENCH.md`](coverage/examples/causal_decoder/MODEL-BENCH.md) is regenerated with the separate interpreters and explicit compiler settings:

```sh
MERLIN_ROOT=/path/to/merlin python3 coverage/examples/causal_decoder/bench.py --export-python "$EXPORT_PYTHON" --model-python "$MODEL_PYTHON" --model-root "$NPU_MODEL" --atlas-mlir-bin "$ATLAS_MLIR_BIN" --min-tier stream-verified --out "$OUT/causal_decoder"
```

`MODEL-BENCH.md` lists, per distinct launch: the generator variant (for `projection_gemm`, the traffic-reuse recipe or the default schedule), program size, backend verification tier, npu_model model cycles, bit-exactness against `lib/golden` where the family has an independent golden, FP32 fidelity (cosine and relative error), the count in the model, and cycles times count. Shapes a generator rejects are `no-variant` with the generator's error. Its summary gives verification-tier counts, cycle estimates and shares by launch kind, and the unsupported list; the complete Launches table ranks supported shapes by total cycles, with unsupported or untimed shapes last, and is the starting point for kernel ablation and optimization. The cycles are npu_model model cycles, not RTL cycles, and a sum of per-launch cycles ignores host time and launch overhead. Inputs are seeded synthetic tensors, so this shows kernel coverage and per-launch behavior, not model accuracy; real-value capture, host graphs, and ELFs are the later steps in the [bringup guide](https://github.com/ucb-bar/atlas-mlir/blob/sota/bringup-performance/docs/bringing-up-a-model.md). [`coverage/examples/tinyllama/MODEL-BENCH.md`](coverage/examples/tinyllama/MODEL-BENCH.md) and [`coverage/examples/smolvla/MODEL-BENCH.md`](coverage/examples/smolvla/MODEL-BENCH.md) are the same table for the two focus models. Run `make bench-model MODEL=tinyllama OUT="$OUT/tinyllama" NPU_MODEL="$NPU_MODEL" MODEL_PYTHON="$MODEL_PYTHON" ATLAS_MLIR_BIN="$ATLAS_MLIR_BIN"` to regenerate the TinyLlama table from the committed inventory; the target rejects missing required variables.

## License

Apache License 2.0; see [LICENSE](LICENSE).
