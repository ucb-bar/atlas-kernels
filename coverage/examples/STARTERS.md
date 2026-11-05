# Random-weight model starters

Run from the atlas-kernels repository root. Set `OUT` to a fresh writable output directory, `EXPORT_PYTHON` to the pinned model interpreter, `MODEL_PYTHON` to an interpreter with npu_model dependencies, `MODEL_ROOT` to an unmodified npu_model checkout at `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`, and `ATLAS_MLIR_BIN` to a directory containing `atlas-opt` and `atlas-emit`. The mapping driver needs PyYAML. The original simulator environment used Python 3.14. Full model initialization and export require substantial host memory; weights and generated artifacts stay in `OUT`.


The exporter calls `get_model_and_inputs()` in each starter's `loader.py`. `--allow-unmapped` deliberately produces an INCOMPLETE mapping; omit it to require complete placement. Candidate scheduling pads dimensions to 32, caps M at 128, and panels N at 2048 for K<=2048 or 1024 otherwise. This policy is unqualified for this model. Benchmark inputs are synthetic FP8/BF16 operands, independent of model weights; exact kernel arithmetic does not qualify model quantization, scales, layouts, or accuracy. Benchmark runs can resume with `--resume`. The historical `COVERAGE.md` retains all unmapped rows, and the `MODEL-BENCH.md` retains launch results. JSON receipts are generated under `OUT` and are not checked in.


Merlin real-value capture, model arithmetic admission, ordered dispatch, host graph, ELF, QEMU, FireSim, and pretrained accuracy remain PENDING. Generators, mappings, compatibility adapters, and prospective host providers are hand-authored. These are random-weight starter benches, not completed model bringups.

Choose [Gemma 2](gemma2/STARTER.md) or [PI0](pi0/STARTER.md) for architecture, export dependencies, commands, and remaining gaps.
