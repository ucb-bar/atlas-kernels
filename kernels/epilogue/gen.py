"""Generate a projection launch with a fused epilogue: python gen.py --shape MxKxN --steps silu [--fp8] --out DIR.

Steps that take an operand (add, add_rne, mul) read a BF16 tile tensor staged right after the output.
Alternatively, --swiglu-elements COUNT emits a standalone contiguous BF16 gate/up stream.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from kernel_gen import parser, shape, write  # noqa: E402
import atlas_program  # noqa: E402
from atlas_epilogue import Epilogue, Step, TileInput  # noqa: E402

OPERAND = {"add", "add_rne", "mul"}
cli = parser(__doc__)
mode = cli.add_mutually_exclusive_group(required=True)
mode.add_argument("--shape", type=shape, help="physical MxKxN with a traffic-reuse recipe")
mode.add_argument("--swiglu-elements", type=int, help="opt-in contiguous finite BF16 gate/up stream")
cli.add_argument("--steps", help="comma-separated: add, add_rne, mul, relu, exp, recip, silu, gelu_tanh, connector_scale")
cli.add_argument("--fp8", action="store_true", help="pack the output to FP8 E4M3")
cli.add_argument("--rescale-exponent", type=int, default=None, help="projection output scale 2**E")
cli.add_argument("--output-scale-exponent", type=int, default=None, help="FP8 output scale exponent")
cli.add_argument("--output-row-stride", type=int, help="bytes between 32-row output blocks (the consumer layout)")
args = cli.parse_args()
if args.swiglu_elements is not None:
    if (args.steps is not None or args.fp8 or args.rescale_exponent is not None
            or args.output_scale_exponent is not None or args.output_row_stride is not None):
        cli.error('--swiglu-elements rejects projection steps, FP8, rescale, and output-layout options')
    from atlas_swiglu_stream import assembly, layout
    try:
        spec = layout(args.swiglu_elements)
        source = assembly(args.swiglu_elements)
    except ValueError as error:
        cli.error(str(error))
    write(args, f'epilogue-swiglu-stream-{args.swiglu_elements}', source,
          dict(family='epilogue', variant='compact-swiglu-stream', logical_elements=spec['elements'],
               padded_elements=spec['padded_elements'], input_layout='contiguous_flat_bf16',
               output_dtype='bf16', dram_base=0x90000000, gate_offset=spec['gate_offset'],
               up_offset=spec['up_offset'], output_offset=spec['output_offset'], output_bytes=spec['size'],
               arena_bytes=spec['arena_bytes'], guard_bytes=spec['guard_bytes'], vmem_bytes=spec['vmem_bytes'],
               dma_load_bytes=4 * spec['padded_elements'], dma_store_bytes=2 * spec['padded_elements'],
               numerical_policy='RTL EXP/RCP tables, truncating denominator add, RNE products; not host SiLU'))
    sys.exit(0)
if args.steps is None:
    cli.error('--steps is required with --shape')
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
source = atlas_program.assembly(m, k, n, offset, 0 if args.rescale_exponent is None else args.rescale_exponent, traffic_reuse=True, epilogue=epilogue)
write(args, f"epilogue-{m}x{k}x{n}-{'-'.join(names)}{'-fp8' if args.fp8 else ''}", source,
      dict(family="epilogue", mkn=[m, k, n], steps=names, output_dtype=common["output_dtype"], operand_offset=extra,
           output_offset=offset, output_bytes=epilogue.output_bytes, recipe=atlas_program._reuse_module(m, k, n),
           scratch_registers=list(common["scratch_registers"])))
