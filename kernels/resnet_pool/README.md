# resnet_pool

3x3 stride-two max pooling of FP8 channels-last tensors, as one standalone launch. The pool is exact for nonnegative normal FP8 inputs: VUNPACK and VPACK at unit scale round-trip every such byte, VMAXIMUM orders them as the bytes order, and an MXU0 product with a one-hot FP8 row moves a byte into BF16 without rounding. Padded taps contribute zero, which never exceeds a nonnegative maximum. Each input row is staged once in its own 8 KiB VMEM slot.

Stage: standalone launch.

## Entry point

`atlas_resnet_pool.py` `MaxPool(source_offset, constants_offset, output_offset, channels, height, width)`: `validate(arena_bytes=None)` checks geometry and alignment, `constants()` returns the one-hot shift tiles staged at `constants_offset`, `assembly(arena_base)` returns the program, and `dma_load_bytes`, `dma_store_bytes`, `output_height`, and `output_width` describe it.

## Where the model uses it

| Model | Input (CxHxW) | Output |
| --- | --- | --- |
| ResNet50 | 64x112x112 (stem output) | 64x56x56 |

## Generate

```sh
make gen-resnet_pool NPU_MODEL=/path/to/npu-model ARGS="--input 64x112x112"
```

## Tests and checks

`tests/test_resnet_maxpool.py` checks geometry, the one-hot shift matrices, and, with `NPU_MODEL`, executes the pool on npu_model `47b8599` through `lib/atlas_rtl_reference.py` and compares every output byte with the byte maximum, round-trips every admitted byte, and runs the timed core checks for alignment, banks, and hazards.

The shared [per-family make interface](../README.md) provides `gen`, `data`, `test`, and explicitly provisioned `rtl` targets for this family, with generated payloads in ignored output directories and reference gaps recorded as PENDING.
