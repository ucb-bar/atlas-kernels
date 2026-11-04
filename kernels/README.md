# Per-family make interface

Every family has a thin Makefile that includes [common.mk](common.mk), following the shared-rules layout in `ucb-bar/radiance-kernels`. The four targets below use one shared implementation in [family_interface.py](../lib/family_interface.py) and the existing generator and reference producers. They run bounded representative cases from [RTL-RESULTS.md](../coverage/RTL-RESULTS.md), rather than changing any kit generator default. The existing `gen.py` CLIs and root `gen-<family>` targets remain the interface for other admitted shapes.

Set `PYTHON` to Python 3.14 with NumPy, PyTorch, and the pinned model's dependencies, `NPU_MODEL` to an unmodified `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` checkout, and `ATLAS_MLIR_BIN` to the directory containing `atlas-opt` and `atlas-emit`. PyYAML is additionally required by the complete root test suite. These dependencies are provisioned separately; targets do not download them.

```sh
make -C kernels/projection_gemm gen PYTHON="$MODEL_PYTHON" NPU_MODEL="$NPU_MODEL"
make -C kernels/projection_gemm data PYTHON="$MODEL_PYTHON" NPU_MODEL="$NPU_MODEL"
make -C kernels/projection_gemm test PYTHON="$MODEL_PYTHON" NPU_MODEL="$NPU_MODEL" ATLAS_MLIR_BIN="$ATLAS_MLIR_BIN"
make -C kernels/projection_gemm rtl PYTHON="$MODEL_PYTHON" NPU_MODEL="$NPU_MODEL" ATLAS_MLIR_BIN="$ATLAS_MLIR_BIN" \
  RTL_SIMULATOR="$RTL_OUT/obj/VAtlasCore" RTL_BUILD_RECEIPT="$RTL_OUT/build-result.json"
```

Replace `projection_gemm` with any family in the table. `OUT` defaults to the repository's ignored `build/families/`, with files under `OUT/<family>/`. Use an absolute `OUT` when invoking make from different directories. Output inside the repository must be under an ignored `build/` or `out/` directory. Reviewed manifests, export fixtures, and evidence reports remain committed.

| Family | Physical case shape | Scope | Independent reference |
| --- | --- | --- | --- |
| projection_gemm | 32x32x960 | SmolVLA state projection, 1 logical row | Ordered BF16 MAC and rescale |
| attention | 1x129x32x4 | TinyLlama packed causal decode, head dimension 64 | FP32 fidelity; Atlas arithmetic PENDING |
| rmsnorm | 32x2048 | TinyLlama decode norm, 1 logical row | FP32 fidelity; Atlas arithmetic PENDING |
| rope | 32x2048x256 | TinyLlama decode K projection, synthetic positions 0..31 | Ordered MAC and RoPE, logical row 0 |
| epilogue | 32x32x960 | Reduced projection with SiLU | FP32 fidelity; Atlas arithmetic PENDING |
| rne_add | 32x32x960 | Reduced projection with residual add | Ordered MAC and exact BF16 sum |
| smol_fusion | 32x32x960 | Reduced projection with bias and GELU(tanh) | FP32 fidelity; Atlas arithmetic PENDING |
| resnet_pool | 32x4x32 | Reduced 3x3 max pool, stride 2, padded boundary | Host nonnegative FP8 byte maximum |
| resnet_patch_gather | 32x6x6 | Reduced 3x3 gather and DMA readback; excludes GEMM | Host im2col byte movement |

`gen` writes `program.S`, assembled little-endian `program.bin`, and `program.json`. The source is produced by the existing case builder, using the family's real generator implementation. The metadata includes shape, reduction, seed, output offset, and byte extent. Generation alone is assembler encoding; compiler verification is performed by `test`.

`data` depends on `gen` and runs the actual input and host reference producers. It writes `input_arena.bin`, named input tensors as `.npy`, `reference.npy`, and `manifest.json`, with `golden.bin` and `golden_decoded.npy` only where an independent reference exists. The manifest records seed, physical and logical shapes, scales, arithmetic policy, dependency source hashes, interpreter/library versions, and artifact hashes. The initial arena includes inputs, constants, poisoned output, and guards. The default `SEED=7` is recorded; another nonnegative `SEED` reruns the same case geometry with new synthetic values. Data generation never executes a device program or substitutes device output for a reference. Its PASS means production completed; device validation and missing independent arithmetic remain separately PENDING.

`test` depends on `data`, checks persisted payloads against their producer and manifest, translates the generated source to machine IR, runs both Atlas verifiers, requires at least `stream-verified`, and requires emitted words to equal the pinned model assembler and stored binary. It then executes the program on the pinned functional model with `numerics="rtl"`, checks input and guard preservation, checks the independent golden where present, and records FP32 fidelity separately without claiming a fidelity acceptance threshold. RoPE allows only documented padded BF16 zero-sign differences outside its logical golden scope. Missing arithmetic remains PENDING in `test-result.json`; the execution and guard checks still run. This target tests one representative program; the root `make test` runs the existing complete unit suites, including refusal tests. Root `make test-families` runs all nine representative family tests.

`rtl` depends on `test` and requires explicit `RTL_SIMULATOR` and `RTL_BUILD_RECEIPT`. It verifies the simulator SHA-256 and successful build against [RTL.md](../RTL.md) and uses the existing pinned runner with RTL revision `0079c0541111197741a231c002e3843fa6f545b2`. It requires `SEED=7`; the runner regenerates the identical selected program and initial arena. Both memory services run by default (`RTL_MEMORY=both`), including functional and timed model equality, complete RTL/model byte equality, instruction counts, and matched-memory DMA replay. `RTL_TIMEOUT` defaults to 3,600 seconds per simulator invocation. Generated RTL receipts stay under `OUT/<family>/rtl/`; each family report leaves the other eight families PENDING. The runner refuses existing case output directories, so rerun RTL with a fresh `OUT`. No simulator build is attempted by this target, and no full-model or full-shape RTL qualification is implied.

If a simulator wall limit expires, retain that failed receipt and rerun with a larger `RTL_TIMEOUT` and a fresh `OUT`. The program, input data, and runner's 2,000,000-cycle bound stay the same. A timeout remains a failed qualification until a complete rerun passes.

For clean reproducibility checks, run `gen` and `data` for all nine families twice with two fresh output roots, then compare all relative filenames and SHA-256 values, including manifests. Generation and data targets intentionally rerun on every invocation so seed or source changes cannot reuse stale payloads.
