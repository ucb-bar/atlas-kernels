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
