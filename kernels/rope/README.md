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

`gen.py` writes the program text (`.S`), the 32-bit words npu_model's assembler produces (`.bin`), and a `.json` record into `build/rope/`. `kernel.yaml` is the machine-readable manifest.

## Reference and bench results

`reference.py` defines the bench cases and their FP32 PyTorch references for `lib/kernel_bench.py`: 1 case (TinyLlama prefill q_proj with RoPE), bit-exact against `ordered_matmul` plus the RoPE RNE formula. [`RESULTS.md`](RESULTS.md) lists bit-exactness, cosine similarity and relative error against the FP32 reference, and npu_model model cycles (not RTL cycles) for each case, run on npu_model 47b8599 with `numerics="rtl"`.

## Tests and checks

There is no standalone numeric unit test yet. Exactness is established by executing every TinyLlama RoPE launch on npu_model `47b8599` with `numerics="rtl"` in the export flow (atlas-mlir `tools/tiny_full/export_capture.py --qualify-launches`), and atlas-mlir `tools/tiny_full/check_programs.py` checks the rebuilt programs.
