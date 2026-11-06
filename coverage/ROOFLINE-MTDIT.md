# Multi-Task DiT launch roofline

The pretrained action-chunk export contains 3,545 linear calls and 624 attention calls across the complete 100-step Euler denoising session. It uses three cameras with two observation frames, 197 image tokens, 77 text tokens, a 32-action horizon, and width 512. The campaign retains 16 complete logical linear geometries and three complete attention geometries. 22 of 22 physical launch kinds have current passing receipts. These are isolated count-weighted device-launch estimates from real captured pretrained boundaries on seeded observations; connected quantized inference, task accuracy, host time, and hardware RTL execution remain PENDING.

## Count-weighted results

| Family | Qualified kinds | Launches | Model cycles | SOL cycles | SOL fraction | Headroom |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| projection_gemm | 19 | 44,016 | 29,164,253,820 | 22,124,940,600 | 75.863215% | 7,039,313,220 |
| attention | 3 | 624 | 264,634,356 | 123,646,152 | 46.723394% | 140,988,204 |

Qualified launch total: 29,428,888,176 npu_model model cycles. The count-weighted resource bound is 22,248,586,752 cycles, 75.601180% of the measured estimate, with 7,180,301,424 cycles of theoretical headroom.

SOL is the largest per-launch resource charge. Aggregate SOL fraction divides the sum of `bound * count` by the sum of `measured * count`. Headroom sums `(measured - bound) * count`; it is a theoretical scheduling limit, not predicted savings. Counts retain the full exported session, while each kind measures one real representative operand boundary. They exclude graph dependencies, host dispatch, inter-launch transfers, and cross-launch reuse.

## Launch kinds

| Kind | Complete source geometry | Physical panel or attention | Count | Model cycles | Bound | SOL fraction | Cosine | Relative error | Independent golden |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 000-00-projection_gemm | 1182x768x768 | 128x768x128 (128x768x128 logical) | 2,592 | 729,530 | DRAM | 76.023878% | 0.99898161 | 4.716053% | exact, full extent |
| 000-01-projection_gemm | 1182x768x768 | 32x768x128 (30x768x128 logical) | 288 | 182,420 | DRAM | 75.999342% | 0.99901794 | 4.601073% | exact, full extent |
| 001-00-projection_gemm | 1182x768x3072 | 128x768x128 (128x768x128 logical) | 2,592 | 729,530 | DRAM | 76.023878% | 0.99936090 | 3.606245% | exact, full extent |
| 001-01-projection_gemm | 1182x768x3072 | 32x768x128 (30x768x128 logical) | 288 | 182,420 | DRAM | 75.999342% | 0.99939640 | 3.494813% | exact, full extent |
| 002-00-projection_gemm | 1182x3072x768 | 128x3072x128 (128x3072x128 logical) | 648 | 2,837,690 | DRAM | 75.835451% | 0.99493952 | 10.598711% | exact, full extent |
| 002-01-projection_gemm | 1182x3072x768 | 32x3072x128 (30x3072x128 logical) | 72 | 709,460 | DRAM | 75.829222% | 0.99463789 | 10.768869% | exact, full extent |
| 003-00-projection_gemm | 77x512x512 | 96x512x128 (77x512x128 logical) | 192 | 371,480 | DRAM | 76.139765% | 0.99939909 | 3.475233% | exact, full extent |
| 004-00-projection_gemm | 77x512x2048 | 96x512x128 (77x512x128 logical) | 192 | 371,480 | DRAM | 76.139765% | 0.99928612 | 4.514979% | exact, full extent |
| 005-00-projection_gemm | 77x2048x512 | 96x2048x128 (77x2048x128 logical) | 48 | 1,425,560 | DRAM | 75.866537% | 0.99992042 | 2.719078% | exact, full extent |
| 006-00-projection_gemm | 1x512x512 | 32x512x128 (1x512x128 logical) | 4 | 123,860 | DRAM | 76.107702% | 0.99890573 | 4.696839% | exact, full extent |
| 007-00-projection_gemm | 1x256x512 | 32x256x128 (1x256x128 logical) | 400 | 65,300 | DRAM | 76.410413% | 0.99980778 | 2.060721% | exact, full extent |
| 008-00-projection_gemm | 1x512x256 | 32x512x128 (1x512x128 logical) | 200 | 123,860 | DRAM | 76.107702% | 0.99969374 | 2.478159% | exact, full extent |
| 009-00-projection_gemm | 32x14x512 | 32x32x128 (32x14x128 logical) | 400 | 14,060 | DRAM | 78.748222% | 0.99924127 | 3.903797% | exact, full extent |
| 010-00-projection_gemm | 1x5916x3072 | 32x5920x128 (1x5916x128 logical) | 14,400 | 1,360,942 | DRAM | 75.800879% | 0.98525935 | 32.627212% | exact, full extent |
| 011-00-projection_gemm | 32x512x1536 | 32x512x128 (32x512x128 logical) | 7,200 | 123,860 | DRAM | 76.107702% | 0.99911844 | 4.198052% | exact, full extent |
| 012-00-projection_gemm | 32x512x512 | 32x512x128 (32x512x128 logical) | 2,400 | 123,860 | DRAM | 76.107702% | 0.99918411 | 4.063666% | exact, full extent |
| 013-00-projection_gemm | 32x512x2048 | 32x512x128 (32x512x128 logical) | 9,600 | 123,860 | DRAM | 76.107702% | 0.99917463 | 4.062821% | exact, full extent |
| 014-00-projection_gemm | 32x2048x512 | 32x2048x128 (32x2048x128 logical) | 2,400 | 475,220 | DRAM | 75.858339% | 0.99891496 | 4.791645% | exact, full extent |
| 015-00-projection_gemm | 32x512x14 | 32x512x32 (32x512x14 logical) | 100 | 31,007 | DRAM | 75.953817% | 0.99953359 | 3.055389% | exact, full extent |
| 016-00-attention | B6 H12/12 197x197 d64 none | H72/72 197x197 d64 | 12 | 14,533,145 | DRAM | 38.499540% | 0.99970124 | 2.447540% | none |
| 017-00-attention | B1 H8/8 77x77 d64 arbitrary | H8/8 77x77 d64 | 12 | 435,118 | DRAM | 62.156013% | 0.99941413 | 3.454211% | none |
| 018-00-attention | B1 H8/8 32x32 d64 none | H8/8 32x32 d64 | 600 | 141,692 | DRAM | 62.645739% | 0.99995414 | 0.997487% | none |

## Coverage and arithmetic

The semantic inventory assigns 361,690,282,525 FLOPs to the complete exported session. Bias-free linear products account for 349,348,233,216 FLOPs, and complete QK, softmax, and PV attention accounts for 10,183,008,960 FLOPs. Together they are candidates for 359,531,242,176 FLOPs, 99.40306929621457% of the inventory denominator. The remaining 2,159,040,349 FLOPs include the unsupported stride-16 patch gather/convolution route, normalization, nonlinearities, biases, residual and adaLN arithmetic, rotary/timestep work, scheduler updates, and control or layout work. Candidate semantic coverage is not connected-model coverage or a runtime share. Zero-FLOP operations still require implementation.

Projection staging retains full K. M and N split at 128, and only tails pad to a multiple of 32: image M1182 has a 30-row tail, text M77 pads to 96, input K14 pads to 32, conditioning K5916 pads to 5920, and output N14 pads to 32. No K slice substitutes for a complete contraction. Different complete logical geometries retain distinct source boundaries even when their physical panels match. Panel multiplicities are exact. Fidelity uses the representative actual panel and the corresponding unquantized FP32 product, excluding bias; it does not measure every layer, panel, or denoising step.

The shared library calibrates FP8 E4M3 power-of-two operand scales on each representative physical panel, accumulates in ordered target BF16 arithmetic, and stores BF16. Independent ordered-matmul goldens cover each complete physical projection output, including padding. Every passing projection requires exact golden output, exact functional-versus-timed output, unchanged inputs and guard bytes, and complete termination. Attention stages real Q/K/V through the existing BF16-to-FP8 path and retains its actual scale and visibility mask. It has no independent target-arithmetic golden. Attention PASS means compiler checks, exact timed-versus-functional and official-versus-RTL agreement, and informational FP32 fidelity, not independent model-output exactness.

## Selected schedule and official APIs

The 19 projection kinds use the existing default or exact-shape traffic-reuse selector. All three attention kinds explicitly select [synchronous DMA](../kernels/attention/SYNCHRONOUS-DMA.md). The default attention source fails the full timing verifier on an outstanding-DMA/vector-store conflict for these shapes. The opt-in library variant waits after each DMA and inserts a scalar halt guard; all original arithmetic and layouts remain. It trades overlap for a provable schedule. Shipped defaults and kit dispatch are unchanged. No faster-variant ablation or performance improvement is claimed.

Every admitted program passes machine-IR lowering, `atlas-opt` stream and timing verification at tier `verified`, and `atlas-emit` word-for-word equality with the pinned model assembler. Qualification uses Python 3.14.7, PyTorch 2.11.0+cu130, NumPy 2.5.4, and one Torch/OMP/MKL/OpenBLAS thread per worker. Functional and cycle qualification use unmodified npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`, `DefaultHardwareConfig`, and `numerics="rtl"`. Official measurements use unmodified `npu_performance.speed_of_light.speed_of_light` and `npu_performance.roofline.measure` from `74982e7c68bd28ba6505b4df37290f198102ccce`, default curve memory, and explicitly selected RTL numerics. Imported execution-model sources are authenticated against the original RTL-model checkout. Unlike the retained Qwen campaign's default PyTorch arithmetic setting, this campaign selects RTL arithmetic in both APIs; no default-PyTorch numerical equivalence is claimed.

Both public APIs run their own complete official cycle simulation. A validation wrapper checks output bytes against the authenticated RTL output and checks all input/guard bytes before each simulation closes. Measured cycles, MXU FLOPs, and DRAM bytes must agree between the two APIs, and official cycles must equal the separate pinned RTL cycle count. Source files are checked against their original Git blobs before and after execution. Receipt reuse and report aggregation require exact current spec and runner hashes; stale receipts contribute no cycles.

![Multi-Task DiT MXU and VPU rooflines](../../out/roofline-mtdit.png)

Points represent complete launch kinds without count weighting. Blue points are projections, and orange points are attention. Circles use DRAM intensity, squares use VMEM intensity, solid lines are DRAM roofs, and dashed lines are VMEM roofs. Official peaks are 4096 MXU FLOP/cycle, 31.50769230769231 VPU op/cycle, 1.4769990737882062 DRAM B/cycle, and 58.51428571428571 VMEM B/cycle. SOL additionally includes scalar, port, issue, XLU, and LSU charges. MXU counters charge padded physical work; the semantic inventory excludes padding.

## Reproduction and evidence

Run the staged capture and launch inventory as described in Merlin `examples/atlas/model_bringup/multi_task_dit/README.md`, then use its `qualify.py`. Set `MODEL_PYTHON` to a Python 3.14 environment with NumPy, PyTorch, PyYAML, and tabulate, `NPU_RTL` to the pinned RTL model, `NPU_PERF` to the pinned performance model, and `ATLAS_BIN` to the verified compiler tools. These dependencies and the checkpoint must already exist locally.

```bash
EXAMPLE=merlin/examples/atlas/model_bringup/multi_task_dit
PY="${MODEL_PYTHON:?set a Python 3.14 environment with the simulator dependencies}"
export TMPDIR="$PWD/tmp" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 "$PY" -B "$EXAMPLE/qualify.py" --prepare --attention-sync-dma \
  --inventory out/inventory-checkpoint/boundaries.json --out out/qualification-new \
  --library atlas-kernels --model-root "$NPU_RTL" \
  --perf-root "$NPU_PERF" \
  --atlas-bin "$ATLAS_BIN"
nice -n 10 "$PY" -B "$EXAMPLE/qualify.py" --run --mode rtl --out out/qualification-new --workers 4
nice -n 10 "$PY" -B "$EXAMPLE/qualify.py" --run --mode official --out out/qualification-new --workers 4
nice -n 10 "$PY" -B "$EXAMPLE/qualify.py" --report --out out/qualification-new
```

The retained campaign is `out/qualification-checkpoint/`. Kernel sources are byte-identical to commit `70107c2d2849b084743c00e942af5a8fc9081bd9`. `out/inventory-final-binding.json` independently binds all 93 arrays across 19 NPZs to the final capture after its equivalent camera-index rewrite; actual inputs, eager results, logical counts, and FLOPs are unchanged. `out/qualification-checkpoint/campaign.json` binds the captured boundary inventory, generated source, arena, independent golden, full-precision reference, compiler tools, and assembled words. Per-kind `rtl.json` and `official.json` bind the exact specs and runner. `qualification.json` contains current validated aggregates. [Artifact identifiers](../../out/roofline-artifacts.json) bind the retained inventory, campaign, report, receipts, and plot by bytes and SHA-256. Checkpoints, full captures, arenas, and simulator receipts are retained job artifacts rather than source-repository payloads.

Remaining gates are connected quantized policy execution, trained task accuracy, independent attention arithmetic goldens, hardware RTL execution, host ELF/QEMU/FireSim, dispatch and transfer timing, and a fresh 16-ELF identity rebuild. RTL numerics is an execution-model arithmetic setting; it is not hardware RTL execution. No overall deployment or accuracy PASS is claimed.
