"""Generate a SmolVLA fused projection panel: python gen.py --shape 128x768x3072 --operations bias,gelu_tanh --out DIR.

--fp8 emits the vision fc1 producer whose FP8 output feeds fc2 (smol_fusion_fp8.stage_fc1_fp8).
"""
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from kernel_gen import parser, shape, write  # noqa: E402
from atlas_program import dma_load_bytes  # noqa: E402
from smol_fusion_program import stage_panel  # noqa: E402
from smol_fusion_fp8 import stage_fc1_fp8  # noqa: E402

args = parser(__doc__)
args.add_argument("--shape", type=shape, default=(128, 768, 3072))
args.add_argument("--operations", default="bias,gelu_tanh", help="'', bias, bias,gelu_tanh, or connector_scale")
args.add_argument("--module", default="vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.fc1")
args.add_argument("--fp8", action="store_true", help="fc1 bias and GELU with FP8 output (output scale -4)")
args.add_argument("--rescale-exponent", type=int, default=-19, help="projection output scale 2**E")
args.add_argument("--output-scale-exponent", type=int, default=-4, help="FP8 output scale exponent (with --fp8)")
args = args.parse_args()
m, k, n = args.shape
offset = m * k + k * n + 256
row = dict(index=0, call=0, module=args.module, physical_mkn=[m, k, n], logical_mkn=[m, k, n], logical_row_start=0,
           output_offset=offset, dram_bytes=offset + 2 * m * n + 256, recipe="traffic-reuse-v1",
           program_policy="traffic-reuse-v1", output_rescale_exponent=args.rescale_exponent, dram_base=0x90000000,
           requested_dma=dict(load_bytes=dma_load_bytes(m, k, n, traffic_reuse=True), store_bytes=2 * m * n))
arena = b"A" * (m * k) + b"W" * (k * n) + b"\xa5" * 256 + b"\xcd" * (2 * m * n) + b"Z" * 256
bias = struct.pack(f"<{n}H", *(0x3c00 + (i % 64) for i in range(n)))
operations = tuple(op for op in args.operations.split(",") if op)
if args.fp8:
    program = stage_fc1_fp8(row, arena, bias_bf16=bias, output_scale_exponent=args.output_scale_exponent)
else:
    program = stage_panel(row, arena, operations=operations, bias_bf16=bias if "bias" in operations else None)
write(args, f"smol_fusion-{m}x{k}x{n}-{'fp8' if args.fp8 else '-'.join(operations) or 'plain'}", program.source,
      dict(family="smol_fusion", mkn=[m, k, n], operations=program.metadata.get("operations"), fp8=args.fp8))
