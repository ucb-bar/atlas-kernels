# rope

Rotate-half rotary position embedding as a serialized post-pass over BF16 q and k that a producer projection retained in VMEM. Each output is `RNE(RNE(x * cos) + RNE(r * sin))`, the host rotary formula, where the low half uses `r = -x[d + 32]` (as `x[d + 32] * -sin`) and the add is the exact [rne_add](../rne_add/README.md) sequence. The producer stores no unrotated output; the pass writes the rotated BF16 tiles to the producer's original output extent.

Stage: a post-pass appended to the q_proj and k_proj projection launches.

## Entry points

`atlas_rope_postpass.py`:

- `ResidentRopeEpilogue`, an `Epilogue` with `cos_offset`, `sin_offset`, and `neg_sin_offset` (one BF16 32x32 cosine, sine, and negated-sine tile per physical row tile, shared by all heads), `head_dimension` (64), and `scratch_registers` (default x10/x11).
- `postpass(epilogue)` and `append_postpass(source, epilogue)` emit the pass and append it to a producer program.

## Where the models use it

TinyLlama q and k projections in prefill and decode (32 query heads and 4 KV heads of dimension 64). The cosine and sine tables come from atlas-mlir `tools/tiny_full/fusion.py` `rope_tables`, and `tools/tiny_full/check_programs.py` appends the pass when it rebuilds programs. SmolVLA RoPE runs on the host in Merlin C today.

## Generate

```sh
make gen-rope NPU_MODEL=/path/to/npu-model ARGS="--shape 128x2048x2048"
make gen-rope NPU_MODEL=/path/to/npu-model ARGS="--shape 32x2048x256"
```

## Reference and bench results

`reference.py` defines the bench cases and their FP32 PyTorch references for `lib/kernel_bench.py`: 1 case (TinyLlama prefill q_proj with RoPE), bit-exact against `ordered_matmul` plus the RoPE RNE formula. [`RESULTS.md`](RESULTS.md) lists bit-exactness, cosine similarity and relative error against the FP32 reference, and npu_model model cycles (not RTL cycles) for each case, run on npu_model 47b8599 with `numerics="rtl"`.

## Tests and checks

Run `make test-lib test-projection_gemm test-coverage NPU_MODEL=/path/to/npu-model` from the repository root. See [shared test coverage](../../README.md#running-the-tests).

Standalone numeric tests are described below. Existing producer exactness is established by executing every TinyLlama RoPE launch on npu_model `47b8599` with `numerics="rtl"` in the export flow (atlas-mlir `tools/tiny_full/export_capture.py --qualify-launches`), and atlas-mlir `tools/tiny_full/check_programs.py` checks the rebuilt programs.

## Opt-in standalone BF16 launch

`atlas_rope_standalone.py` reuses the same resident post-pass without a projection producer. `layout(M, N)` describes the arena and exact DMA counts, `assembly(M, N)` generates its independent launch, and `prepare(M, N, H, cosine, sine, negative_sine, logical_rows=...)` stages packed little-endian BF16 tile bytes. M is the physical row count and must be a positive multiple of 32; N must be a positive multiple of 64. H must fit VMEM banks 2 and 3 (`2*M*N <= 524288` bytes), and the coefficient tables must fit bank 4 (`192*M <= 262144` bytes). The output stage uses 4096 bytes in bank 5. This launch does not require or bypass any producer scratch admission.

```sh
make gen-rope NPU_MODEL=/path/to/npu-model ARGS="--postpass-shape 192x960"
make gen-rope NPU_MODEL=/path/to/npu-model ARGS="--postpass-shape 64x320"
```

`--postpass-shape MxN` is mutually exclusive with projection `--shape MxKxN`. Its `--rescale-exponent` must be 0. With neither shape flag, the original 128x2048x2048 projection and its instruction words remain the default. The standalone launch normally uses physical DRAM base 0x90000000; the Python entry points accept another aligned low32 arena that contains every staged byte.

The arena stores H at offset 0, a 256-byte guard, BF16 output at `2*M*N + 256`, another 256-byte guard, then cosine, sine, and negative-sine tables, followed by a final guard. Each table contains `64*M` bytes: one BF16 32x32 tile per physical row tile, shared by every head of dimension 64. DMA loads `2*M*N + 192*M` bytes and stores `2*M*N` bytes. Inputs, coefficient tables, and guards must remain unchanged. Logical tail rows require signed-zero H, cosine 1, and signed-zero sine. Negative sine must equal sine XOR 0x8000 word for word, including signed zeros. `prepare` validates exact extents, finite values, coefficient range [-1,1], padding, and this sign relation; it neither selects positions nor silently converts an FP32 table.

The arithmetic contract is BF16 coefficients, two RNE BF16 products per output half, and an exact RNE BF16 add or subtract. This matches TinyLlama's BF16-table contract. SmolVLA's existing host path uses FP32 coefficients and different rounding points, so rounding its tables to BF16 is an approximation requiring a separate full-action accuracy decision. Its host graph and defaults do not change merely because the standalone kernel is available. Whole-model graph integration, all-site launch qualification, host packing cost, and ELF replay are separate gates.

`tests/test_rope_standalone.py` uses independent NumPy rounding, signed zeros, synthetic trigonometric coefficients, partial logical rows, multiple row tiles and heads, full-arena checks, requested DMA checks, and invalid shape/address/table controls. It executes through unmodified original47 and needs no captured model data. Run `make test-rope NPU_MODEL=/path/to/npu-model`.
