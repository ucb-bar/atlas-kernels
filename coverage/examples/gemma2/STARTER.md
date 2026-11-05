# Gemma 2 2B starter bench

**INCOMPLETE.** The loader constructs the full Transformers Gemma 2 2B architecture with deterministic random BF16 weights (seed 20261104), 128 input tokens, and one cached decode step. It downloads no checkpoint. The model has 26 layers, width 2304, MLP width 9216, vocabulary 256000, query/KV heads 8/4, head dimension 256, and 2,614,341,888 unique parameters. It does not exercise the 4096-token sliding-window boundary.

Use Python 3.12.12, torch 2.10.0, and Transformers 4.57.1 in `EXPORT_PYTHON`. The loader selects eager attention and verifies the architectural defaults, so no config file is needed. The original export used torch 2.10.0+cu128. The exporter includes functorch batch-wrapper handling required by this workload; generic phase tags remain `inference`, and call order distinguishes prefill from decode.

Historical export: 5,118 records, 286 aggregate rows, and 366 projection calls. Candidate projections account for 674,437,791,744 of 678,359,243,268 heuristic FLOPs (99.42192112%); 270 rows and 4,752 instances remain unmapped. The candidate plan has 1,904 panels across 10 launch keys. All 10 keys passed functional golden and timed/functional equality, totaling 22,795,072,704 count-weighted npu_model cycles. These cycles exclude host work and are not RTL cycles.

The largest remaining semantic gaps are attention d=256 with scale 1/16, softcap 50, FP32 softmax, and local/global masking; width-2304 offset-gamma RMSNorm; d=256 RoPE; GELU gate multiplication; and final-logit softcap 30. Existing shape probes rejected 10 generator geometries and returned two attention `no-variant` results. No prospective host implementation is counted as implemented.

Run from the atlas-kernels repository root. Set `OUT` to a fresh writable output directory, `EXPORT_PYTHON` to the pinned model interpreter, `MODEL_PYTHON` to an interpreter with npu_model dependencies, `MODEL_ROOT` to an unmodified npu_model checkout at `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`, and `ATLAS_MLIR_BIN` to a directory containing `atlas-opt` and `atlas-emit`. The mapping driver needs PyYAML. The original simulator environment used Python 3.14. Full model initialization and export require substantial host memory; weights and generated artifacts stay in `OUT`.

```sh
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$OUT/cache" TMPDIR="$OUT/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"
export GEMMA2_OUT="$OUT/metadata"
"$EXPORT_PYTHON" -B coverage/export_ops.py gemma2 \
  --loader coverage/examples/gemma2/loader.py \
  --out "$OUT/export" --fixture "$OUT/inventory.json"
python3 -B coverage/build_table.py \
  --inventory "gemma2=$OUT/inventory.json" \
  --mapping coverage/examples/gemma2/mapping.yaml \
  --allow-unmapped --out "$OUT/coverage"
"$MODEL_PYTHON" -B coverage/bench_model.py \
  --coverage "$OUT/coverage/coverage.json" --model gemma2 \
  --model-root "$MODEL_ROOT" --atlas-mlir-bin "$ATLAS_MLIR_BIN" \
  --out "$OUT/bench" --workers 10 --job-timeout 7200
```

The exporter calls `get_model_and_inputs()` in [loader.py](loader.py). `--allow-unmapped` deliberately produces an INCOMPLETE mapping; omit it to require complete placement. Candidate scheduling pads dimensions to 32, caps M at 128, and panels N at 2048 for K<=2048 or 1024 otherwise. This policy is unqualified for this model. Benchmark inputs are synthetic FP8/BF16 operands, independent of model weights; exact kernel arithmetic does not qualify model quantization, scales, layouts, or accuracy. Benchmark runs can resume with `--resume`. The historical [coverage report](COVERAGE.md) retains all unmapped rows, and the [benchmark report](MODEL-BENCH.md) retains launch results. JSON receipts are generated under `OUT` and are not checked in.

Merlin real-value capture, model arithmetic admission, ordered dispatch, host graph, ELF, QEMU, FireSim, and pretrained accuracy remain PENDING. Generators, mappings, compatibility adapters, and prospective host providers are hand-authored. These are random-weight starter benches, not completed model bringups.
