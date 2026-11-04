# rne_add

A BF16 tile add rounded to nearest even, built from the VPU's truncating BF16 arithmetic. The 34-instruction sequence equals the host's FP32 add followed by BF16 round-to-nearest-even on every operand pair, except that NaN results are the canonical `0x7FC0`. It exists so a device residual add matches the host reference bit for bit.

Stage: an epilogue step (`Step("add_rne")`) and a building block of the rope post-pass; never a launch by itself.

## Entry point

`atlas_rne_add.py` `rne_add(a, b, result, sign=4, *, constants=True)` returns the instruction list for one register pair add: `a` and `b` are the input matrix-register pairs, `result` the output pair, and `sign` a scratch pair. The sequence uses fixed temporaries m6 to m30, m48, and m50, and constants m32 to m46 (`CONSTANTS`); `constants=False` omits the eight constant loads when they are already resident. Instructions are spaced by `delay 66`. Every binary instruction reads two different physical MRF banks.

## Where the models use it

TinyLlama prefill residual adds after the o and down projections, through the `add_rne` step of an [epilogue](../epilogue/README.md); the [rope](../rope/README.md) post-pass also uses it for its final add.

## Generate

```sh
make gen-rne_add NPU_MODEL=/path/to/npu-model   # the sequence on one register pair, wrapped with ecall
```

`gen.py` writes the program text (`.S`), the 32-bit words npu_model's assembler produces (`.bin`), and a `.json` record into `build/rne_add/`. `kernel.yaml` is the machine-readable manifest.

## Reference and bench results

`reference.py` defines the bench cases and their FP32 PyTorch references for `lib/kernel_bench.py`: 1 case (TinyLlama prefill o_proj with the exact residual add), bit-exact against `ordered_matmul` plus the exact BF16 sum. [`RESULTS.md`](RESULTS.md) lists bit-exactness, cosine similarity and relative error against the FP32 reference, and npu_model model cycles (not RTL cycles) for each case, run on npu_model 47b8599 with `numerics="rtl"`.

## Tests and checks

Run `make test-lib test-projection_gemm test-coverage NPU_MODEL=/path/to/npu-model` from the repository root. See [shared test coverage](../../README.md#running-the-tests).

There is no standalone numeric unit test yet. Exactness is established by executing every TinyLlama launch that uses it on npu_model `47b8599` with `numerics="rtl"` (atlas-mlir `tools/tiny_full/export_capture.py --qualify-launches`), and register admission is covered by `kernels/projection_gemm/tests/test_program_checks.py`.

The shared [per-family make interface](../README.md) provides `gen`, `data`, `test`, and explicitly provisioned `rtl` targets for this family, with generated payloads in ignored output directories and reference gaps recorded as PENDING.
