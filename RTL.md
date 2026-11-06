# Standalone AtlasCore kernel measurements

[Per-family RTL results](coverage/RTL-RESULTS.md) compare identical programs and input arenas on AtlasCore Verilator and unmodified npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` with `numerics="rtl"`, for both model matched and synthetic DMA memory services. Four cases use published model launch geometries, and five use explicitly labeled reductions. The ResNet patch case measures the 3x3 gather operation and its DMA readback, excluding the following GEMM.

The standalone build extracts Scala sources from Atlas RTL commit `0079c0541111197741a231c002e3843fa6f545b2` and its pinned FP submodules, compiles them with `java` and `scalac`, emits SystemVerilog with Chisel and firtool, and builds the C++ TileLink harness with Verilator. It needs cached dependencies, rather than invoking sbt, mill, Chipyard make, or the VCS simulator. Checkouts are read-only; extraction and artifact access reject restricted paths and symlinks, pruning Git trees before descent.

## Build

Required tools are Python 3.14 for the pinned model configuration, a JDK, Scala compiler/library/reflect 2.13.16, Chisel plugin 6.7.0, a compatible Chipyard dependency assembly containing Chisel 6.7.0 and TileLink classes, firtool 1.62.1, Verilator 5.022, and a C++17 compiler. This assembly is the Chipyard project uberjar: it packages the compiled project and dependency classes used for Chisel, Rocket TileLink, and diplomacy during direct Scala compilation and elaboration. The pinned `chipyard.jar` SHA-256 is `25c454b8f2179b13b4ada30c24cf6dab3c8db0517bf2913ffbe64b15c68961d1`, which the builder verifies. All 131 selected Atlas and FP Scala source files are compiled into a separate class directory that precedes that assembly on the runtime classpath. The source, tool, emitter, generated SystemVerilog, harness, and simulator hashes are recorded in `build-result.json`. The build also generates `dma_config.h` from pinned npu_model configuration and DMA curves and records their hashes. [Compact provenance](coverage/rtl-provenance.json) preserves the pins and artifact hashes without local absolute paths. Regenerate the compact manifest with `python3 lib/rtl_provenance.py --build-receipt "$RTL_OUT/build-result.json" --out coverage/rtl-provenance.json`. Large generated files stay outside the repository history.

Set `MODEL_PYTHON` to the Python 3.14 interpreter and `NPU_MODEL` to its unmodified model checkout before building.

```sh
"$MODEL_PYTHON" -B lib/rtl_build_pinned.py \
  --rtl-root /path/to/sp26-atlas-acc \
  --classpath /path/to/chipyard.jar \
  --classpath-revision 48c3a7d11f6de3f0cecb33ae6a31c24ec26ff9d8 \
  --model-root "$NPU_MODEL" \
  --compiler /path/to/scala-compiler-2.13.16.jar \
  --library /path/to/scala-library-2.13.16.jar \
  --reflect /path/to/scala-reflect-2.13.16.jar \
  --plugin /path/to/chisel-plugin_2.13.16-6.7.0.jar \
  --firtool /path/to/firtool \
  --verilator /path/to/verilator_bin \
  --out "$RTL_OUT" --jobs 12
```

`RTL_OUT` must be fresh unless `--resume` explicitly selects a previous attempt from this same pin. All paths are explicit. The build requires 20 GiB free and caps compilation at 12 workers. Assertions remain enabled.

The observed Chipyard dependency checkout is commit `48c3a7d11f6de3f0cecb33ae6a31c24ec26ff9d8`; its `variables.mk` defines the following dependency preparation command. This command is documentation only and was not run for these measurements. The cached assembly has no original build receipt or embedded commit, so its historical source cleanliness and exact origin remain unattested; a fresh teammate dependency assembly is PENDING. The accepted artifact is identified by its exact SHA-256 above, and `--classpath-revision` records the observed checkout separately.

```sh
# In a separately prepared Chipyard checkout at the stated commit:
java -jar scripts/sbt-launch.jar \
  ';project chipyard; set assembly / assemblyOutputPath := file(".classpath_cache/chipyard.jar"); assembly'
sha256sum .classpath_cache/chipyard.jar
```

## Run the family cases

The model interpreter needs Python 3.14 with PyTorch and NumPy. Set `MODEL_PYTHON`, `NPU_MODEL`, and `ATLAS_MLIR_BIN` to the pinned interpreter, model checkout, and compiler binaries. Programs use the same generator implementations as `kernels/<family>/gen.py`; `lib/rtl_cases.py` reuses their reference staging and provides the two ResNet adapters.

```sh
"$MODEL_PYTHON" -B lib/rtl_runner.py \
  --simulator "$RTL_OUT/obj/VAtlasCore" \
  --build-receipt "$RTL_OUT/build-result.json" \
  --rtl-revision 0079c0541111197741a231c002e3843fa6f545b2 \
  --model-root "$NPU_MODEL" --atlas-mlir-bin "$ATLAS_MLIR_BIN" \
  --out "$RESULTS_OUT" --timeout 900 --memory both \
  --report coverage/RTL-RESULTS.md
```

The default `--memory both` runs all nine cases with each service and writes `npu-model/<family>/` and `synthetic/<family>/` under the output directory. Use `--memory npu-model` or `--memory synthetic` to select one service and `--family NAME` to run one case. Each case writes assembly, emitted instruction words, machine IR, initial input bytes, model and RTL output bytes, a simulation log, and `result.json`. Backend emission must equal npu_model's assembler. The default minimum tier is `encode-only`, so verifier rejections remain explicit in the receipt rather than preventing an RTL experiment. Receipts report actual tiers. The runner checks the simulator hash and revision against its build receipt. A missing family remains `PENDING`; any output difference, timeout, or failed command fails qualification.

The complete output is compared bit for bit against the model, including padded rows. Decode RoPE additionally checks the independent arithmetic golden over its one logical row; its known signed-zero differences in padded rows are recorded separately. Those padding bytes still participate in the full RTL/model comparison.

## Run a generated program and input arena

`--program` accepts the assembly produced by a generator. Stage its complete DRAM arena in a binary file beginning at physical address `0x90000000`, then identify the output extent in that arena. The generator CLI emits code and metadata; staging tensor values remains the caller's responsibility.

```sh
"$MODEL_PYTHON" -B lib/rtl_runner.py \
  --simulator "$RTL_OUT/obj/VAtlasCore" \
  --build-receipt "$RTL_OUT/build-result.json" \
  --rtl-revision 0079c0541111197741a231c002e3843fa6f545b2 \
  --model-root "$NPU_MODEL" --atlas-mlir-bin "$ATLAS_MLIR_BIN" \
  --program build/projection_gemm/program.S --input staged-arena.bin \
  --output-offset 32000 --output-bytes 61440 \
  --shape 32x32x960 --family projection_gemm --memory both --out "$CUSTOM_OUT"
```

The provided program is encoded through `lib/atlas_backend.py`, checked against the assembler, and executed by both the functional and timed model before RTL simulation. A custom family run is a separate experiment and does not replace the nine-case table unless all nine cases are supplied.

## Cycle interpretation

RTL cycles count first instruction issue through ECALL inclusive. Completion requires ECALL status, no illegal PC, and an empty DMA response queue. With `--memory npu-model`, the TileLink service uses the default model queue depth, recent read/write mix window, and DMA latency and bandwidth curves; each run validates its DMA trace against the same model configuration. With `--memory synthetic`, requests are accepted during 4 of every 11 cycles and FIFO responses return after 43 cycles. Model cycles use `kernel_bench.timed_cycles` and its default Core configuration. The matched column differs by 1 startup cycle in all nine families. The synthetic column records a harness memory choice and does not establish SoC performance. Inputs are seeded synthetic values at the stated geometries, and the measurements do not establish whole-model execution or model accuracy.

The opt-in [streamed SmolVLA projection](kernels/projection_gemm/SMOL-FAST.md) additionally qualifies the full captured 128x768x768 shape on the same pinned matched-memory simulator. Its [RTL result](coverage/RTL-RESULTS.md#opt-in-streamed-smolvla-projection) is separate from the nine-family baseline table and identifies exact original-result reuse.

## Opt-in TinyLlama fused prefill

The opt-in [TinyLlama fused variant](kernels/projection_gemm/TINY-FAST.md) additionally confirms the full captured 128x2048x2048 SiLU epilogue with matched memory. Its [RTL result](coverage/RTL-RESULTS.md#opt-in-tinyllama-fused-prefill) compares every byte of the complete 4,981,248-byte DRAM arena, including the 524,288-byte output extent. The captured-shape replay uses an authenticated two-worker simulator built from the same pinned generated RTL and memory configuration; an exact isolated fixture checks the threading change.

The [whole-kit result](coverage/RTL-RESULTS.md#whole-kit-captured-programs) covers all 1,589 distinct captured kit programs: 1,589 exact in full output bytes, instruction counts, and matched-memory cycles (RTL = Core - 1), plus 50/50 synthetic-memory output checks. Its generated tables, simulator identities, and reproduction inputs are retained alongside the nine-family baseline.
