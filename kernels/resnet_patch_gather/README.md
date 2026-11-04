# resnet_patch_gather

Device row gathers that build a convolution's A operand directly from an FP8 channels-last tensor, so a convolution runs as a [projection_gemm](../projection_gemm/README.md) launch without a host im2col. A 1x1 input gathers one 32-byte channel line per output row. A kxk input (odd kernel, same padding) orders K as (ky, kx, channel) with the weights permuted to match, so every A tile row is still one 32-byte channel line of one tap; padded taps and padded output rows read a declared zero line.

Stage: the A-operand source of a projection launch (`direct_input=`).

## Entry points

`atlas_resnet_input.py`:

- `DirectInput(offset, channels, height, width, stride=1, zero_offset=0, kernel=1, padding=0)`: pass as `direct_input=` to `atlas_program.assembly`. `validate(m, k, ...)` requires M to be the output positions padded to 32 and K to be `kernel * kernel * channels`; channels must be a power of two and a multiple of 32, stride 1 or 2. kxk gathers need the blocked traffic-reuse recipe.
- `kernel_major_weights(weights, channels, kernel, n)` permutes weights to the (ky, kx, channel) K order.
- `relocate_assembly(source, arena_base, *, arena_bytes=None, weight_source=None)` relocates a gathered program.

## Where the model uses it

ResNet50 1x1 and 3x3 convolutions after the stem (atlas-mlir `tools/resnet_sims/fast.py`; Merlin `examples/atlas/comparisons/refresh/capture_resnet_device.py` captures them).

## Generate

```sh
make gen-resnet_patch_gather NPU_MODEL=/path/to/npu-model ARGS="--input 64x56x56 --kernel 3 --n 64"
```

## Tests and checks

`tests/test_resnet_direct_input.py` checks address bounds, the kxk geometry and weight order, and, with `NPU_MODEL`, that a gathered block executed on npu_model `47b8599` equals the host patches. `kernels/projection_gemm/tests/test_program_checks.py` checks register admission with direct input.

The shared [per-family make interface](../README.md) provides `gen`, `data`, `test`, and explicitly provisioned `rtl` targets for this family, with generated payloads in ignored output directories and reference gaps recorded as PENDING.
