# rne_add

A BF16 tile add rounded to nearest even, built from the VPU's truncating BF16 arithmetic. The 34-instruction sequence equals the host's FP32 add followed by BF16 round-to-nearest-even on every operand pair, except that NaN results are the canonical `0x7FC0`. It exists so a device residual add matches the host reference bit for bit.

Stage: an epilogue step (`Step("add_rne")`), a building block of the rope post-pass, or an opt-in standalone streamed add.

## Entry point

`atlas_rne_add.py` `rne_add(a, b, result, sign=4, *, constants=True)` returns the instruction list for one register pair add: `a` and `b` are the input matrix-register pairs, `result` the output pair, and `sign` a scratch pair. The sequence uses fixed temporaries m6 to m30, m48, and m50, and constants m32 to m46 (`CONSTANTS`); `constants=False` omits the eight constant loads when they are already resident. Instructions are spaced by `delay 66`. Every binary instruction reads two different physical MRF banks.

## Where the models use it

TinyLlama prefill residual adds after the o and down projections, through the `add_rne` step of an [epilogue](../epilogue/README.md); the [rope](../rope/README.md) post-pass also uses it for its final add.

## Generate

```sh
make gen-rne_add NPU_MODEL=/path/to/npu-model   # the sequence on one register pair, wrapped with ecall
make gen-rne_add NPU_MODEL=/path/to/npu-model ARGS='--stream-shape 192x960'
make gen-rne_add NPU_MODEL=/path/to/npu-model ARGS='--elements 2048'
```

## Reference and bench results

`reference.py` defines the bench cases and their FP32 PyTorch references for `lib/kernel_bench.py`: 1 case (TinyLlama prefill o_proj with the exact residual add), bit-exact against `ordered_matmul` plus the exact BF16 sum. [`RESULTS.md`](RESULTS.md) lists bit-exactness, cosine similarity and relative error against the FP32 reference, and npu_model model cycles (not RTL cycles) for each case, run on npu_model 47b8599 with `numerics="rtl"`.

## Tests and checks

Run `make test-lib test-projection_gemm test-coverage NPU_MODEL=/path/to/npu-model` from the repository root. See [shared test coverage](../../README.md#running-the-tests).

The opt-in `atlas_rne_add_stream.assembly(rows, width, *, dram_base=0x90000000)` variant streams two physical tiled BF16 tensors without requiring a projection producer. Both dimensions must be positive multiples of 32; callers pad logical rows and columns. The arena contains left, right, and output tensors in that order, each `2 * rows * width` bytes. Its VMEM footprint is 4096 bytes. The generator records offsets, dimensions, DMA bytes, and the assembled source hash. The existing no-argument generator and all model defaults remain unchanged. This option does not itself establish net host savings or whole-model accuracy; staging, launch, and checking costs still apply.

`--elements COUNT` selects a compact contiguous stream for row-major host buffers. Elementwise addition is independent of the shared lane permutation: the raw input buffers can be loaded without a layout conversion, and the raw output retains their element order. `flat_geometry(COUNT)` pads only the final 1024-word BF16 chunk and returns an equivalent physical tile shape. Allocate all three padded regions and initialize both input tails to zero. A 2048-element decode residual therefore needs two tiles instead of the 64 tiles of a padded 32x2048 projection epilogue. This is a separate launch, so compare its launch and memory costs before selecting it over fusion. The logical element count and padded count are both recorded in generator metadata.

`make test-rne_add NPU_MODEL=/path/to/npu-model` executes the streamed variant on npu_model `47b8599` with `numerics="rtl"`, compares every output byte against FP32 addition followed by BF16 RNE, and checks exact DMA counts, input preservation, and trailing guards. Cases exercise multiple tile rows and columns, a broad exponent range, ties, subnormals, cancellation, and overflow. Shape and address admission checks reject invalid arenas. The fused TinyLlama path retains its existing qualification and register tests.
