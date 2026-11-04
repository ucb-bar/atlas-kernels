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
| `atlas_asm_mlir.py` | `translate(text, *, name)`: turns kernel assembly (the text the generators emit and npu_model's assembler accepts) into Atlas machine-IR MLIR, one generic-syntax op per instruction word, with a source-line location on every op. `parse(text)` returns the decoded instructions. It handles `#` comments, whole-line labels, the `li` and `nop` pseudo-instructions (expanded exactly as the assembler does), label or word-offset branch targets, `imm(xN)` operands, ABI register names, and all 101 npu_model mnemonics. It does not import npu_model, so comparing its words with the assembler's is an independent check. |
| `atlas_backend.py` | Encodes kernel programs through the Atlas compiler backend: `atlas_asm_mlir` translation, `atlas-opt --verify-atlas-machine-stream` and `--verify-atlas-timing`, then `atlas-emit`. `compile(text)` returns the words, the tier, and every verifier diagnostic. `cross_checked(text, assembler_words)` also requires the words to equal npu_model's assembler words. `python lib/atlas_backend.py run ...` runs a build or check script with every assembly cross-checked and writes a receipt. `python lib/atlas_backend.py compile ...` handles one file. |
| `kernel_bench.py` | Bench driver: runs every `kernels/<family>/reference.py` case on npu_model 47b8599 with `numerics="rtl"` through `Executor`, compares output bytes with the `lib/golden` result where one exists, reports cosine similarity and relative error against the FP32 PyTorch reference, and with `--timed` the npu_model Core model cycles (not RTL cycles). It writes `kernels/<family>/RESULTS.md` and `coverage/RESULTS.md`; `--report OUT` rebuilds them from saved results and writes only changed content. Both execution and report mode exit nonzero on an independent-golden or timed/functional mismatch. Empty selections fail, tables show timed equality, and reports list the cases without independent goldens. |
| `kernel_gen.py` | Command-line plumbing for `kernels/<family>/gen.py`: writes `.S`, the npu_model-assembled `.bin`, and a `.json` record. |
| `golden/atlas_tile_arithmetic.py` | Independent rational FP8 E4M3 and BF16 arithmetic: `mxu0_dot` (ordered MXU0 accumulation), `bf16_round`, `rescale_bf16`, `fp8_value`, `tensor_scale_exponent`, and output layout helpers. It imports no model, Torch, or assembler. |
| `golden/atlas_vector_arithmetic.py` | NumPy `ordered_matmul(uint8[M,K], uint8[K,N]) -> uint16[M,N]` and BF16 helpers that equal the rational oracle exactly; the module docstring proves the binary64 intermediates exact. |
| `fallback/` | Stand-ins for the two atlas-mlir host helpers that unmodified generators import (`run_model_tile.validate_archive_members`, `chipyard.arena.relocate_assembly`). `atlas_kernels_path` serves them only when atlas-mlir's real modules are not importable; atlas-mlir's `test/test_kernel_library.py` checks they match. |

## Compiler backend (`atlas_asm_mlir.py`, `atlas_backend.py`)

The kernel generators emit Atlas assembly text, and npu_model's assembler turns it into the 32-bit words the builds embed. `atlas_backend.py` sends the same text through the Atlas compiler backend in atlas-mlir: it translates it to machine IR, runs the `atlas-opt` verifiers, encodes the words with `atlas-emit`, and refuses any program whose words differ from the assembler's. A backend gap can therefore never change a kernel binary. It shows up as an error or as a lower tier.

### Tiers

| Tier | Meaning |
| --- | --- |
| `verified` | `--verify-atlas-machine-stream` and `--verify-atlas-timing` both accept the program, and the `atlas-emit` words equal the assembler's. |
| `stream-verified` | The machine-stream verifier accepts it (fields, encoding, one flat instruction chain, branch delay slots, in-program targets), the timing verifier rejects it, and the words equal the assembler's. The timing diagnostics are kept. |
| `encode-only` | No verifier accepts it (verification was skipped with `--no-verify`, or the machine-stream verifier rejected it), and the `atlas-emit` words still equal the assembler's. |

Status for the TinyLlama, SmolVLA, and ResNet50 kits (1,589 distinct programs, 1,487,694 words, all equal to the assembler's): 13 `verified`, 1,576 `stream-verified`, 0 `encode-only`. The timing verifier's rejections come from its DMA rules. It checks DMA overlap with per-block constant register values, so it cannot prove double-buffered loop addresses disjoint, and it times one block at a time, so a prefetch kept in flight across a loop branch is outside its scope. Running all 1,576 rejected programs on npu_model 47b8599 found no access to a pending transfer's VMEM or DRAM range. `--verify-atlas-timing` stops at its first violation, so a rejected program carries one timing diagnostic, and the issue-spacing, reservation, and drain checks after the DMA rules have not run for it. A run that continued past every violation found no issue-spacing, reservation, or drain diagnostic. It did find that the 45 attention programs end with `delay 128` directly before `ecall`, which the RTL does not wait for (harmless here: nothing is outstanding at the halt). The verifiers are not weakened to change these results.

`--verify-atlas-generated-schedule` does not apply to these programs. It needs the resource contracts that atlas-mlir's virtual-IR lowering attaches, and hand-written kernels have none.

### Requirements

`atlas-opt` and `atlas-emit` from an atlas-mlir build. Name the build's `bin` directory with `--atlas-mlir-bin` or `ATLAS_MLIR_BIN`; atlas-mlir's CTest passes `ATLAS_OOT_BIN_DIR`, and `PATH` is the last resort. Missing tools are an error, never a silent fallback to the assembler. The assembler comparison and `run` also need npu_model 47b8599 importable (on `PYTHONPATH`, or imported by the script being run).

### How to run

One program:

```sh
PYTHONPATH=/path/to/npu-model python3 lib/atlas_backend.py compile build/projection_gemm/projection_gemm-128x2048x2048-tiny.S \
  --atlas-mlir-bin /path/to/atlas-mlir/build/bin --check-assembler -o program.bin --mlir program.mlir
```

This prints a JSON record with the tier, the word count and SHA-256, and every stream and timing diagnostic, and writes the binary and the machine-IR module. `--min-tier verified|stream-verified|encode-only` sets the minimum accepted tier for `compile` and `run`, defaulting to `stream-verified`. Stream rejection fails compilation by default. `--no-verify` skips both verifiers and requires explicit `--min-tier encode-only`. A successful `stream-verified` result still carries a timing rejection; it is not a timing pass.

Any build, check, or generator script, unmodified:

```sh
PYTHONPATH=/path/to/npu-model python3 lib/atlas_backend.py run --backend atlas-mlir \
  --atlas-mlir-bin /path/to/atlas-mlir/build/bin --expect-min-programs 1 \
  --min-tier stream-verified --receipt receipt.json -- \
  kernels/projection_gemm/gen.py --out build/projection_gemm --shape 128x2048x2048 --traffic-reuse
```

`run` executes the script as `python SCRIPT ARGS` would, in the same process. The hook covers encodes inside the wrapped process and forked workers that inherit it; it does not instrument fresh Python subprocesses, including spawned workers. Wrap each fresh Python process separately. `--expect-min-programs N` requires at least N distinct program texts, defaults to 1, and rejects N below 1. Set N to the known expected coverage for qualification. The wrapper exits nonzero if too few programs are recorded or any compiler program falls below `--min-tier`, and writes those policy errors and rejected tiers into the receipt. The assembler backend enforces the count but supplies no compiler tier evidence. Each time the script (or `atlas_execution.Executor`) assembles a program with npu_model's `input_to_program`, the backend encodes the same text and the script stops with `BackendError` unless the words are equal. The receipt records the backend, the verification mode, the command, the exit code, the SHA-256 of this library and the translator, the `atlas-opt` and `atlas-emit` paths, SHA-256, and versions, the observed atlas-mlir source checkout commit, tracked-source SHA-256, and modified flag (the checkout path comes from `CMakeCache.txt`), the npu_model commit, a tier and diagnostic histogram, and one record per distinct program (source SHA-256, word count and SHA-256, tier, every diagnostic). `--backend assembler` writes the same receipt without the compiler, for machines without an atlas-mlir build. The source fields describe the checkout observed when the receipt is created; binary digests identify the exact tools used. They do not prove that a moved checkout HEAD was used to rebuild an older binary. A copied source tree without Git metadata has no inferred commit and must be authenticated separately before qualification. Nothing reads an environment variable to pick a backend.

The atlas-mlir kit builds run the same way, for example `tools/tiny_full/check_programs.py`, `tools/generate_smol_fast_assets.py`, or a script that calls `build_fast.generate` or `resnet_sims/fast.generate`. Their output files are unchanged, because the words are equal by construction.

From Python:

```python
import atlas_kernels_path  # noqa: F401
import atlas_backend
result = atlas_backend.compile(source, bin_dir='/path/to/atlas-mlir/build/bin')
result.tier, result.words, result.timing_diagnostics
atlas_backend.cross_checked(source, assembler_words)   # raises BackendError on any difference
```

### How to verify

- `make test-lib NPU_MODEL=/path/to/npu-model` with `ATLAS_MLIR_BIN` set runs `lib/tests/test_atlas_backend.py`: translator unit tests, one instance of each of the 101 mnemonics through the backend compared with the assembler, tier and diagnostic cases (`verified`, a timing rejection, a stream rejection, `--no-verify`), refusal of differing words and of missing tools, a generated projection kernel, and the `run` wrapper's receipts. Without the tools or npu_model those tests skip; they never pass vacuously.
- atlas-mlir's CTest registers the same file as `atlas-kernels-lib-test_atlas_backend` with the `atlas-opt` and `atlas-emit` built in that tree.
- The kit-wide word check compares every distinct program the 16 TinyLlama, SmolVLA, and ResNet50 kit ELFs launch with the assembler's words and with the binary each ELF embeds: 1,589 of 1,589 equal at this commit. It reads the kit assets, so it is not part of `make test`.

## Tests

`tests/test_vector_arithmetic.py` compares the NumPy oracle with the rational one on all finite FP8 codes, pair products, BF16 decodes, ties, and random ordered chains. `tests/test_atlas_path_guard.py` checks path admission. `tests/test_library_layout.py` checks that module basenames are unique and that nothing earlier on `sys.path` shadows them. `tests/test_kernel_manifests.py` checks every family's README, `kernel.yaml` (entry points exist; projection variants are the recipes `atlas_program` dispatches each listed shape to), and generator. `tests/test_atlas_backend.py` checks the translator and, when `atlas-opt`, `atlas-emit`, and npu_model 47b8599 are available, that every mnemonic and a generated kernel encode to the assembler's words through the backend, the three tiers, and the `run` receipts. `tests/test_kernel_bench.py` checks the bench driver and the reference cases.
