# Per-family standalone AtlasCore RTL results

The same generated instruction words and seeded DRAM inputs run on standalone AtlasCore Verilator and unmodified npu_model 47b8599 with numerics="rtl". Exact means every output byte equals the model output with both memory services, including padding.

| Family | Shape | Exact | npu_model cycles | RTL cycles, matched memory | Model minus matched RTL | RTL cycles, synthetic memory | RTL commit |
| --- | --- | :---: | ---: | ---: | ---: | ---: | --- |
| projection_gemm | 32x32x960 | yes | 83036 | 83035 | 1 | 27109 | 0079c0541111197741a231c002e3843fa6f545b2 |
| attention | 1x129x32x4 | yes | 197542 | 197541 | 1 | 99570 | 0079c0541111197741a231c002e3843fa6f545b2 |
| rmsnorm | 32x2048 | yes | 358882 | 358881 | 1 | 110446 | 0079c0541111197741a231c002e3843fa6f545b2 |
| rope | 32x2048x256 | yes | 485881 | 485880 | 1 | 182340 | 0079c0541111197741a231c002e3843fa6f545b2 |
| epilogue | 32x32x960 (reduced) | yes | 101396 | 101395 | 1 | 45433 | 0079c0541111197741a231c002e3843fa6f545b2 |
| rne_add | 32x32x960 (reduced) | yes | 216986 | 216985 | 1 | 121825 | 0079c0541111197741a231c002e3843fa6f545b2 |
| smol_fusion | 32x32x960 (reduced) | yes | 148586 | 148585 | 1 | 71498 | 0079c0541111197741a231c002e3843fa6f545b2 |
| resnet_pool | 32x4x32 (reduced) | yes | 10336 | 10335 | 1 | 5234 | 0079c0541111197741a231c002e3843fa6f545b2 |
| resnet_patch_gather | 32x6x6 (reduced) | yes | 50768 | 50767 | 1 | 39912 | 0079c0541111197741a231c002e3843fa6f545b2 |

The matched-memory column reproduces the default npu_model curve backend and differs from npu_model by 1 cycle in every family because the model counts its startup cycle before first issue.

The synthetic-memory column uses a harness choice of request acceptance during 4 of every 11 cycles and FIFO responses after 43 cycles; it does not measure SoC performance.

RTL cycles count first instruction issue through ECALL inclusive, excluding program loading and completion polling. Completion additionally requires ECALL status, no illegal PC, and no outstanding memory responses. Matched-memory request due cycles and response identities are replayed against the unchanged Python backend.

Build identities and source hashes are retained in [rtl-provenance.json](rtl-provenance.json). Both columns use the same simulator binary and pinned generated RTL.

Every shape reduction below preserves the family operation and uses the existing generator and reference staging where available. Full model shapes are not qualified by reduced runs.

Simulator SHA-256: `a568f3820eeb9dae1601a948353d040670bb622e63e2a72c14bd4b24ebe35c86`.

Compiler tiers: 6 stream-verified, 3 verified. No program silently falls back to assembler encoding.

Decode RoPE's independent golden is compared over its logical row. It differs in 656 padded BF16 words, including 0 nonzero differences; the complete padded output still participates in RTL/model equality.

- projection_gemm: source model shape 32x32x960. exact coverage/examples/smolvla/MODEL-BENCH.md state_proj shape; physical 32x32x960, logical 1x32x960.  Status: PASS.
- attention: source model shape 1x129x32x4. exact coverage/examples/tinyllama/MODEL-BENCH.md packed decode attention shape; one query, 129 keys, 32 query heads, four KV heads, head64, query position 128.  Status: PASS.
- rmsnorm: source model shape 32x2048. exact coverage/examples/tinyllama/MODEL-BENCH.md decode norm shape; physical 32x2048, one logical row.  Status: PASS.
- rope: source model shape 32x2048x256. exact coverage/examples/tinyllama/MODEL-BENCH.md decode K projection shape; physical 32x2048x256, logical 1x2048x256; synthetic RoPE table positions 0..31.  Status: PASS.
- epilogue: source model shape 128x2048x2048. smallest qualified traffic-reuse contraction; retains the family fused operation.  Status: PASS.
- rne_add: source model shape 128x2048x2048. smallest qualified traffic-reuse contraction; retains the family fused operation.  Status: PASS.
- smol_fusion: source model shape 128x768x3072. smallest qualified traffic-reuse contraction; retains the family fused operation.  Status: PASS.
- resnet_pool: source model shape 64x112x112. one full output tile with two output rows and boundary padding.  Status: PASS.
- resnet_patch_gather: source model shape 64x56x56. existing 3x3 gather execution-test geometry, padded rows, and VMEM bank jump; excludes GEMM. 3x3 gather_block only; excludes following projection GEMM Status: PASS.

## Opt-in streamed SmolVLA projection

The new `atlas_program_smol_projection_fast` full captured 128x768x768 projection passes pinned matched-memory RTL. Original RTL cycles are 1,077,911, and candidate cycles are 774,449, saving 303,462. Corresponding Core counts are 1,077,912 and 774,450. All 196,608 output bytes match the captured reference. The new run validates all 27,648 requests and responses against the unchanged model backend. Original results are authenticated reuse of an identical program, input, expected output, simulator, and build, with a newly revalidated archived DMA trace. No fresh original RTL simulation is claimed.

RTL pin is `0079c0541111197741a231c002e3843fa6f545b2`, and simulator SHA-256 is `a568f3820eeb9dae1601a948353d040670bb622e63e2a72c14bd4b24ebe35c86`. The candidate is compiler tier `verified`. The [variant guide](../kernels/projection_gemm/SMOL-FAST.md) and [result manifest](smol-projection-results.json) join the numerical, compiler, timing, and RTL receipts. Functional execution checks full arenas, while RTL dumps and compares the complete output extent. Other shapes have model validation and retain PENDING RTL status.

## Opt-in TinyLlama fused prefill

The opt-in `atlas_program_tiny_fused_fast` confirms the complete captured TinyLlama 128x2048x2048 SiLU epilogue on pinned matched-memory RTL. Candidate RTL cycles are 4,603,121. The original RTL replay remains CANCELLED; the before/after numbers in the variant guide are Core measurements. This candidate-only RTL confirmation makes no RTL speedup claim. All 524,288 output bytes and the complete 4,981,248-byte arena, including inputs, padding, and guards, equal unmodified npu_model `47b8599` with `numerics="rtl"`. The candidate validates all 155,648 requests and responses against the unchanged model backend. Completion records ECALL status 5, no illegal PC, no mismatches, and an empty response queue.

RTL pin is `0079c0541111197741a231c002e3843fa6f545b2`. This replay uses the two-worker simulator SHA-256 `1c12680e8b3e0173f4cc5e1f46b445c5f2bd7d700093e691b3f6a9663e8f2296`, built from the identical authenticated generated RTL and matched-memory configuration as the nine-family baseline. Its only harness change caps the Verilator context at two threads. An isolated fixture reproduces the baseline's exact cycles, output, and DMA trace before the captured-shape replay. The [variant guide](../kernels/projection_gemm/TINY-FAST.md) records selection, timing, and numerical scope; other TinyLlama fused shapes retain PENDING RTL status.

## Whole-kit captured programs

All 1,589 distinct TinyLlama, SmolVLA, and ResNet50 kit programs match every physical output byte (364,589,056 bytes, including padding and holes), instruction counts including ECALL, and matched-memory timing. There are 0 mismatches. Inputs are real captured values and linked weights; goldens use full unmodified npu_model Core execution with numerics="rtl" and seed 42. RTL = Core - 1 cycles throughout because Core includes one startup cycle.

| Model | Programs | Exact | Cycle relation |
| --- | ---: | ---: | --- |
| ResNet50 | 147 | 147 | RTL = Core - 1 |
| SmolVLA | 557 | 557 | RTL = Core - 1 |
| TinyLlama | 885 | 885 | RTL = Core - 1 |

| Model / family | Programs | Exact | Cycle relation |
| --- | ---: | ---: | --- |
| ResNet50/bottleneck_conv1x1a | 10 | 10 | RTL = Core - 1 |
| ResNet50/bottleneck_conv1x1b | 11 | 11 | RTL = Core - 1 |
| ResNet50/bottleneck_conv3x3 | 11 | 11 | RTL = Core - 1 |
| ResNet50/downsample_conv1x1 | 4 | 4 | RTL = Core - 1 |
| ResNet50/fast_bottleneck_conv1x1a | 32 | 32 | RTL = Core - 1 |
| ResNet50/fast_bottleneck_conv1x1b | 32 | 32 | RTL = Core - 1 |
| ResNet50/fast_bottleneck_conv3x3 | 32 | 32 | RTL = Core - 1 |
| ResNet50/fast_downsample_conv1x1 | 8 | 8 | RTL = Core - 1 |
| ResNet50/fast_fc | 2 | 2 | RTL = Core - 1 |
| ResNet50/fast_maxpool+maxpool | 1 | 1 | RTL = Core - 1 |
| ResNet50/fast_stem_conv7x7 | 2 | 2 | RTL = Core - 1 |
| ResNet50/fc | 1 | 1 | RTL = Core - 1 |
| ResNet50/stem_conv7x7 | 1 | 1 | RTL = Core - 1 |
| SmolVLA/attention | 20 | 20 | RTL = Core - 1 |
| SmolVLA/fast_projection | 437 | 437 | RTL = Core - 1 |
| SmolVLA/fast_projection/fused | 13 | 13 | RTL = Core - 1 |
| SmolVLA/projection | 87 | 87 | RTL = Core - 1 |
| TinyLlama/attention | 25 | 25 | RTL = Core - 1 |
| TinyLlama/fast_projection/down_plain | 112 | 112 | RTL = Core - 1 |
| TinyLlama/fast_projection/gate_silu | 150 | 150 | RTL = Core - 1 |
| TinyLlama/fast_projection/k | 50 | 50 | RTL = Core - 1 |
| TinyLlama/fast_projection/lm_head | 32 | 32 | RTL = Core - 1 |
| TinyLlama/fast_projection/o_plain | 51 | 51 | RTL = Core - 1 |
| TinyLlama/fast_projection/q | 50 | 50 | RTL = Core - 1 |
| TinyLlama/fast_projection/up_mul_fp8 | 180 | 180 | RTL = Core - 1 |
| TinyLlama/fast_projection/v | 50 | 50 | RTL = Core - 1 |
| TinyLlama/norm | 10 | 10 | RTL = Core - 1 |
| TinyLlama/projection/down_plain | 19 | 19 | RTL = Core - 1 |
| TinyLlama/projection/gate_silu | 20 | 20 | RTL = Core - 1 |
| TinyLlama/projection/k | 7 | 7 | RTL = Core - 1 |
| TinyLlama/projection/lm_head | 6 | 6 | RTL = Core - 1 |
| TinyLlama/projection/o_plain | 14 | 14 | RTL = Core - 1 |
| TinyLlama/projection/q | 6 | 6 | RTL = Core - 1 |
| TinyLlama/projection/up_mul_fp8 | 93 | 93 | RTL = Core - 1 |
| TinyLlama/projection/v | 10 | 10 | RTL = Core - 1 |

The synthetic-memory sample is 50/50 full-byte exact across all 36 model/family groups; its scope is output exactness, without a matched-cycle claim. These are standalone program replays, not whole-model accuracy or SoC performance measurements.

RTL commit: `0079c0541111197741a231c002e3843fa6f545b2`. Model commit: `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`. Verilator 5.022 simulator variants (four-thread, spin64, and single-thread) each qualified 18/18 nine-family cases across both memory modes. Executed matched-memory simulator SHA-256 identities:

- `8d1039ab9083f939183e71ae998b3e1b8159019b4fe8d4b5dadd2a2581729407`
- `da6bad930f786128f34894d11807827ae5f3c5cf4bed2e4c32e34c5ad41f3eb2`
- `dd4866a5c9975d66cdc9e0c57db26cb0466a6c443fcf029decd610f6cbe07691`

The portable [evidence manifest](rtl-whole-kit.json) retains per-model counts, all family cycle histograms, and the source results.json SHA-256. Regenerate this section without simulation with `python3 lib/rtl_runner.py --whole-kit-report-only --out build/report --report coverage/RTL-RESULTS.md`.

To replay each distinct program, use the committed [lib/rtl_runner.py](../lib/rtl_runner.py) custom-program interface in [RTL.md](../RTL.md#run-a-generated-program-and-input-arena): supply its generator assembly, complete captured DRAM arena beginning at 0x90000000 (real tensors and linked weights), output offset and physical byte extent, one runner family label, pinned model checkout, compiler binaries, simulator, and matching build receipt. Use a separate output directory per program, `--memory npu-model`, `--max-cycles 1000000000000`, and `--timeout 14400`; use `--memory synthetic` for the sample. The captured arenas and generated programs are external inputs, not distributed by the compact manifest. No new simulation was run for this documentation update.
