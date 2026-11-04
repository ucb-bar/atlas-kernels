"""Generate a projection launch with a fused epilogue: python gen.py --shape MxKxN --steps silu [--fp8] --out DIR.

Steps that take an operand (add, add_rne, mul) read a BF16 tile tensor staged right after the output.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from kernel_gen import parser, shape, write  # noqa: E402
import atlas_program  # noqa: E402
from atlas_epilogue import Epilogue, Step, TileInput  # noqa: E402

OPERAND = {"add", "add_rne", "mul"}
args = parser(__doc__)
args.add_argument("--shape", type=shape, required=True, help="physical MxKxN with a traffic-reuse recipe")
args.add_argument("--steps", required=True, help="comma-separated: add, add_rne, mul, relu, exp, recip, silu, gelu_tanh, connector_scale")
args.add_argument("--fp8", action="store_true", help="pack the output to FP8 E4M3")
args.add_argument("--rescale-exponent", type=int, default=0, help="projection output scale 2**E")
args.add_argument("--output-scale-exponent", type=int, default=0, help="FP8 output scale exponent")
args.add_argument("--output-row-stride", type=int, help="bytes between 32-row output blocks (the consumer layout)")
args = args.parse_args()
m, k, n = args.shape
offset = m * k + k * n + 256
common = dict(m=m, n=n, output_offset=offset, output_dtype="fp8" if args.fp8 else "bf16",
              scratch_registers=(27, 28) if atlas_program._reuse_module(m, k, n) == "atlas_program_reuse_resident" else (10, 11))
if args.output_scale_exponent:
    common["output_scale_exponent"] = args.output_scale_exponent
if args.output_row_stride is not None:
    common["output_row_stride"] = args.output_row_stride
extra = offset + Epilogue(**common).output_bytes + 256
names = args.steps.split(",")
epilogue = Epilogue(**common, steps=tuple(Step(op, TileInput(extra) if op in OPERAND else None) for op in names))
source = atlas_program.assembly(m, k, n, offset, args.rescale_exponent, traffic_reuse=True, epilogue=epilogue)
write(args, f"epilogue-{m}x{k}x{n}-{'-'.join(names)}{'-fp8' if args.fp8 else ''}", source,
      dict(family="epilogue", mkn=[m, k, n], steps=names, output_dtype=common["output_dtype"], operand_offset=extra,
           output_offset=offset, output_bytes=epilogue.output_bytes, recipe=atlas_program._reuse_module(m, k, n),
           scratch_registers=list(common["scratch_registers"])))
