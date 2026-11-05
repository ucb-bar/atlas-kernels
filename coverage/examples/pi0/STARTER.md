# PI0 starter bench

**INCOMPLETE.** The loader constructs full LeRobot 0.5.1 PI0Pytorch with random FP32 weights (seed 0), PaliGemma (18 layers, width 2048), an action expert (18 layers, width 1024), and 224x224 SigLIP. It has 4,028,019,472 unique parameter elements, including unused logit heads. Three synthetic cameras, 48 token IDs, six state values padded to 32, and 50x32 noise feed 10 cached denoising steps, returning 50x32 actions. Preprocessing, tokenization, normalization, and six-coordinate output slicing are outside this core. No checkpoint is downloaded.

Use Python 3.12.12 and the pins in [requirements-export.txt](requirements-export.txt). The loader adapts the image-feature API and projection math, renames the causal-mask keyword, and replaces cache deep-copy calls with traceable tensor cloning in private function globals. These adapters enabled export with Transformers 4.57.1; they do not establish pretrained equivalence. Configuration is supplied directly by the loader, with no external config file. The historical export additionally used a phase annotation pass to label vision, prefix, and ten denoising steps. That helper is omitted from this minimal starter: the commands below retain the generic exporter's `inference` phase tags, so phase labels and aggregate row groupings differ from the historical reports. They do not reproduce the reports byte for byte.

Recorded export: 17,440 records, 289 aggregate rows, and 1,928 projection calls. Candidate matrix products account for 4,191,247,794,176 of 4,382,555,893,916 heuristic FLOPs (95.63478243%); 275 rows and 16,054 instances remain unmapped. Conv2d requires an unimplemented patch-gather provider; biases and all other unresolved work remain visible.

The plan has 5,912 panels across 21 launch configurations. All 21 passed logical/full-extent independent goldens and compiler/assembler equality; 20 passed timed/functional equality. Measured cycles total 45,300,319,472 over 5,696 panels. Timing for 216 panels of 128x16384x1024 remains PENDING after 7,200-second and 14,400-second host runtime caps; its cause is unknown, and no missing cycles are estimated. Attention matmuls contribute 187,846,602,240 unmapped FLOPs. Seven separate geometry probes returned `no-variant`, including attention head dimensions 72/256, RMSNorm row/width limits, and unsupported vision-fusion recipes.

Use the [shared setup and qualification limits](../STARTERS.md), then run from the repository root:
```sh
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$OUT/cache" TMPDIR="$OUT/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"
export PI0_METADATA="$OUT/identity.json"
"$EXPORT_PYTHON" -B coverage/export_ops.py pi0 \
  --loader coverage/examples/pi0/loader.py \
  --out "$OUT/export" --fixture "$OUT/inventory.json"
python3 -B coverage/build_table.py \
  --inventory "pi0=$OUT/inventory.json" \
  --mapping coverage/examples/pi0/mapping.yaml \
  --allow-unmapped --out "$OUT/coverage"
"$MODEL_PYTHON" -B coverage/bench_model.py \
  --coverage "$OUT/coverage/coverage.json" --model pi0 \
  --model-root "$MODEL_ROOT" --atlas-mlir-bin "$ATLAS_MLIR_BIN" \
  --out "$OUT/bench" --workers 10 --job-timeout 14400
```

See the historical [coverage report](COVERAGE.md) for unmapped rows and [benchmark report](MODEL-BENCH.md) for launch receipts. Commands write JSON receipts under `OUT`.
