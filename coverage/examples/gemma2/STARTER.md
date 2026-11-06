# Gemma 2 2B starter bench

**INCOMPLETE.** The loader constructs the full Transformers Gemma 2 2B architecture with deterministic random BF16 weights (seed 20261104), 128 input tokens, and one cached decode step. It downloads no checkpoint. The model has 26 layers, width 2304, MLP width 9216, vocabulary 256000, query/KV heads 8/4, head dimension 256, and 2,614,341,888 unique parameters. It does not exercise the 4096-token sliding-window boundary.

Use Python 3.12.12, torch 2.10.0, and Transformers 4.57.1 in `EXPORT_PYTHON`. The loader selects eager attention and verifies the architectural defaults, so no config file is needed. The original export used torch 2.10.0+cu128. The exporter includes functorch batch-wrapper handling required by this workload; generic phase tags remain `inference`, and call order distinguishes prefill from decode.

Recorded export: 5,118 records, 286 aggregate rows, and 366 projection calls. Candidate projections account for 674,437,791,744 of 678,359,243,268 heuristic FLOPs (99.42192112%); 270 rows and 4,752 instances remain unmapped. The candidate plan has 1,904 panels across 10 launch keys. All 10 keys passed functional golden and timed/functional equality, totaling 22,795,072,704 count-weighted npu_model cycles. These cycles exclude host work and are not RTL cycles.

The largest remaining semantic gaps are attention d=256 with scale 1/16, softcap 50, FP32 softmax, and local/global masking; width-2304 offset-gamma RMSNorm; d=256 RoPE; GELU gate multiplication; and final-logit softcap 30. Existing shape probes rejected 10 generator geometries and returned two attention `no-variant` results. No prospective host implementation is counted as implemented.

Use the [shared setup and qualification limits](../STARTERS.md), then run from the repository root:
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

See the historical [coverage report](COVERAGE.md) for unmapped rows and [benchmark report](MODEL-BENCH.md) for launch receipts. Commands write JSON receipts under `OUT`.
