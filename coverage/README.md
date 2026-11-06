# Model coverage

The op and shape inventory comes from each real PyTorch model through `torch.export`. Authored policy in `mapping.yaml` places each exported row on an atlas-kernels family, in host C, or in a build-time constant, with code citations. Coverage percentages partition exported FLOPs by this policy; they do not measure device execution. `build_table.py` fails unless the launch counts derived from the export and placement policy equal the authored qualified-kit expectations in `REQUIRED_LAUNCHES`: 465 projection calls, 66 attention launches, and 132 RMSNorm launches for TinyLlama, and 1,421 projection calls, 200 attention launches, and 24 fused fc1 calls for SmolVLA. The CTest `atlas-python-test_model_coverage` in atlas-mlir compares the hand-written TinyLlama and SmolVLA device plans against these fixtures. Regenerate [COVERAGE.md](COVERAGE.md) with `python3 coverage/build_table.py`.

## Files

The [roofline after opt-in variants](ROOFLINE-AFTER.md) records full-launch before/after device-cycle totals, speed-of-light fractions, remaining family headroom, selection scope, and reproduction limits beside the model benches.

The focus-model benches retain compact tables, full-precision summaries, and authenticated artifact manifests. [Evidence recovery](EVIDENCE.md) explains `make evidence`, which recovers the full generated JSON from pinned historical Git objects into ignored `out/evidence/` and regenerates reports without simulations. `make bench-model` runs new measurements with explicit dependencies.

| File | Role |
| --- | --- |
| `export_ops.py` | Runs `torch.export.export(strict=False)` on one complete inference and writes `fixture/<model>.json` (aggregated op rows with module path, source line, phase, shapes, dtypes, FLOPs, bias component FLOPs, counts, and the ordered projection and attention call indices) and, under the ignored `out/`, the full node list and graph text. A new model can supply `--loader FILE`. |
| `fixture/*.json` | The committed aggregated rows, read by `build_table.py`, its tests, and the atlas-mlir coverage test. |
| `mapping.yaml` | Authored first-match placement rules that place every fixture row: `atlas:<family>` (a `kernels/<family>/` directory, the family names of the atlas-kernels kernel manifests), `host:<file>`, or `build:<file>`, each with citations. A biased Linear or Conv2d uses `bias_where` for its addition component. |
| `coverage.json` | The machine-readable table, including each Atlas row and its launch hint. It is committed so the focus-model bench runs from a clean checkout without PyYAML, and `tests/test_build_table.py` checks that it equals a fresh `build_table.py` output. |
| `build_table.py` | Applies the rules and fails on any unmapped row, unused rule, or missing or incorrect mandatory device-work category; writes `COVERAGE.md` and `coverage.json`. An explicit capture cross-check `FAIL` rejects the report; an absent check is reported separately as `NOT_RUN`. `--check-cites ATLAS_MLIR MERLIN ATLAS_KERNELS` also requires every cited file and host/build destination to exist and every family to have a `kernels/<family>/kernel.yaml` manifest of that name. `make test-cites` runs this check with adjacent checkouts or `ATLAS_MLIR_ROOT` and `MERLIN_ROOT`; explicit missing roots or `CHECK_CITES=1` fail when the checkouts are unavailable. `--inventory MODEL=JSON` selects a new inventory; `--allow-unmapped` writes an explicitly INCOMPLETE report without bypassing other gates. |
| `tests/` | Checks exact regeneration, rule coverage, every mandatory launch category including missing-category mutations, complete bias placement, and unmapped-report behavior. |

## How mapping.yaml relates to Merlin's selection.yaml

Merlin's Atlas `examples/atlas/phase1/kernel_library/selection.yaml` is a pinned authoring registry, disabled for Atlas compiler authoring, with no Atlas compile/runtime selection consumer. It shares family names with this coverage policy. Merlin's `generate_selection.py` checks covered Linear K/N, declared dtypes, required mnemonics, and authored projection variants. It does not check Linear M or all Conv2d, attention, norm, and fusion geometry. Host rows are candidates for new kernels.

## Regenerate

`build_table.py` needs Python 3 with PyYAML. `export_ops.py` must run in the interpreter, with the model revision and the input, that the Merlin capture used (Merlin `examples/atlas/comparisons/README.md`), so the export sees exactly the captured model. Local paths come from environment variables.

| Model | Interpreter | Model and input |
| --- | --- | --- |
| TinyLlama | Python 3.10.14, torch 2.10.0, Transformers 5.4.0 (`modeling_llama.py` SHA-256 `46d313ec...`, pinned in Merlin `capture_tinyllama_native.py`) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` at revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`, BF16, eager attention; the first 128 tokens of Merlin's pinned development corpus (SHA-256 `573e1d24...`, token SHA-256 `24e1d064...`), then two greedy decode steps |
| SmolVLA | Python 3.12.12, torch 2.10.0, Transformers 4.57.1, LeRobot 0.5.1 (the versions `capture_smolvla_policy.py` checks) | `lerobot/smolvla_base` at revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205` with backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (config SHA-256 `ea6bc123...`); the compiled observation of `lerobot/svla_so100_pickplace` revision `3d6d687a25cdf1565cdf24550814f72d999a861d`, train episode 0, frame 0, noise seed 0 (`observation.npz`, SHA-256 `167d7bbe...`, the `--observation-input` of `capture_smolvla_policy.py`); ten denoising steps |

```sh
PYTHONPATH=$TINY_PACKAGES $TINY_PYTHON coverage/export_ops.py tinyllama --checkpoint $TINY_CHECKPOINT \
  --corpus $TINY_CORPUS --out coverage/out/tinyllama --fixture coverage/fixture/tinyllama.json
$SMOL_PYTHON coverage/export_ops.py smolvla --policy $SMOL_POLICY --backbone $SMOL_BACKBONE \
  --observation $SMOL_OBSERVATION --out coverage/out/smolvla --fixture coverage/fixture/smolvla.json
python3 coverage/build_table.py
```

`--capture-receipt RECEIPT` optionally compares the exported ordered Linear and Conv2d calls with the calls Merlin's forward hooks recorded in a local capture receipt (`capture_tinyllama_full.py` or `capture_smolvla_policy.py` output): module, step, and shapes for TinyLlama; module, shapes, and dtypes for SmolVLA. The committed fixtures record that local-receipt check as PASS: 465 of 465 TinyLlama calls and 1,421 of 1,421 SmolVLA calls.

## Adding a model

Write a Python loader exposing `get_model_and_inputs()`, which returns `(torch.nn.Module, tuple_of_positional_inputs)`. It must construct the model and exact captured inputs without network access. The exporter records the loader's SHA-256, exports in eval mode with gradients disabled, and assigns a single `inference` phase; custom cache or denoising phase semantics need an adapter like the focus-model loaders. Loader imports do not write bytecode alongside the source. This adapter exports shapes and operations; it does not capture real values or generate a host graph.

Merlin's `examples/workloads/causal_decoder/loader.py` is a deterministic small example with an 8-token input and no checkpoint dependency. Run these commands from the atlas-kernels root, with `MERLIN_ROOT` set to the Merlin checkout and `TORCH_PYTHON` set to a Python interpreter with torch and NumPy:

```sh
"$TORCH_PYTHON" -B coverage/export_ops.py causal_decoder \
  --loader "$MERLIN_ROOT/examples/workloads/causal_decoder/loader.py" \
  --out "$OUT/causal_decoder/export" --fixture "$OUT/causal_decoder/fixture.json"
python3 -B coverage/build_table.py --inventory "causal_decoder=$OUT/causal_decoder/fixture.json" \
  --allow-unmapped --out "$OUT/causal_decoder/report"
```

The committed small report is [examples/causal_decoder/REPORT.md](examples/causal_decoder/REPORT.md). One command regenerates it and retains the full export and placement artifacts under `$OUT`: `python3 -B coverage/examples/causal_decoder/regenerate.py --out "$OUT"` (with `MERLIN_ROOT` and `TORCH_PYTHON` set as above). The driver Python needs PyYAML.

The report is INCOMPLETE: no causal_decoder placement rules or qualified build are registered, and every exported row appears in the unmapped work list with phase, module, source, op, shape, count, and FLOPs. The JSON retains each complete unmapped record. Omitting `--allow-unmapped` exits nonzero and writes no report. This is a demonstrated inventory and unsupported-op report, not demonstrated model execution on Atlas. The same loader contract accepts other models, with [Gemma 2 and PI0 starters](examples/STARTERS.md) documenting partial random-weight benches.

To implement placement, add `rules.<model>` and `builds.<model>` to `mapping.yaml`, register a display column in `MODELS` if desired, and pass `--inventory MODEL=JSON` until no row is unmapped. Add explicit mandatory launch counts in `REQUIRED_LAUNCHES` when the model has a qualified build. A row that needs an absent kernel or host implementation is an implementation work item. Rules must place a biased projection's `bias_where` component separately; that addition does not create a second projection call.

RMSNorm launch hints accept `eps`, for example `launch: {family: rmsnorm, per: {op: aten.mul.Tensor, dtype: bfloat16}, eps: 1.0e-6}` for a decomposed Qwen norm. Direct `aten.rms_norm.default` exports retain their source epsilon in the inventory and coverage rows, including the input dtype's machine epsilon when PyTorch omits it. An explicit mapping epsilon must agree with a captured epsilon. A row with neither keeps the historical default `1e-5`. Epsilon is part of the launch identity, generator command, FP32 reference, JSON metadata, and Markdown shape description. The generator records both the requested epsilon and `eps_bf16_bits`; device arithmetic uses its rounded BF16 value. RMSNorm still has no independent random-input Atlas-arithmetic golden, so bit-exact status remains unavailable for that family.

Real-value projection qualification can call `coverage/bench_model.py` as a module and use `build(entry, work, model_root, projection_operands=(a, w))`. Both operands must be finite NumPy float32 arrays at physical shapes `[M, K]` and `[K, N]`; callers supply any padding. This path reuses the shared generator, FP8 quantization, FP32 reference, and independent ordered-matmul golden. Other families reject this argument. The returned `Built` exposes `source`, `arena`, `output_offset`, `output_bytes`, `reference`, `decode`, `golden()`, and `meta` for existing execution and reporting helpers.

## Limits

Placement is authored policy for the qualified builds, which place every op identically in default and fast; a build that moves an op needs a rule restricted to its phase or shape. Rules match model source lines and shapes of the pinned library versions, so a new Transformers or LeRobot version needs its rules rechecked. FLOPs count multiply-adds twice and other arithmetic once per element, so they rank work rather than time it. Linear and Conv2d totals include one bias addition per output element when a bias tensor exists; placement splits those additions from the GEMM component. SmolVLA has 172,539,136 bias additions: 75,497,472 in the fused vision fc1 post-pass and 97,041,664 on the host. The split preserves export node counts and projection invocation counts. Zero-FLOP rows (views, casts, and index construction) are placed with the code that performs that layout. The export runs on fake tensors, so it checks shapes and structure, not numerics.

## Coverage

A kernel family is marked for a model when the placement policy assigns at least one exported op of that model to it. `resnet_pool` and `resnet_patch_gather` serve ResNet50, which has no fixture here.
