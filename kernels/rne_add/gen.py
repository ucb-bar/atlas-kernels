"""Generate the exact BF16 add on one register pair as a minimal program: python gen.py --out DIR.

The sequence is an epilogue stage, not a launch by itself; this program wraps it with ecall so the
assembler encodes exactly the 8 constant loads and 34 instructions (each followed by delay 66).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from kernel_gen import parser, write  # noqa: E402
from atlas_rne_add import rne_add  # noqa: E402

args = parser(__doc__)
args.add_argument("--registers", default="0,2,52,4", help="a,b,result,sign even matrix-register pairs")
stream = args.add_mutually_exclusive_group()
stream.add_argument("--stream-shape", type=str, help="opt-in standalone tiled add, physical ROWSxWIDTH")
stream.add_argument("--elements", type=int, help="opt-in contiguous BF16 stream, padding only the last 1024-word chunk")
args = args.parse_args()
if args.stream_shape is not None or args.elements is not None:
    from atlas_rne_add_stream import assembly, flat_geometry
    from kernel_gen import shape
    dimensions = flat_geometry(args.elements) if args.elements is not None else shape(args.stream_shape)
    if len(dimensions) != 2:
        raise ValueError('--stream-shape requires ROWSxWIDTH')
    rows, width = dimensions
    source = assembly(rows, width)
    write(args, f'rne_add-stream-{rows}x{width}', source,
          dict(family='rne_add', variant='contiguous-stream' if args.elements is not None else 'standalone-stream', rows=rows, width=width,
               logical_elements=args.elements, padded_elements=rows * width,
               dram_base=0x90000000, left_offset=0, right_offset=2 * rows * width,
               output_offset=4 * rows * width, output_bytes=2 * rows * width,
               dma_load_bytes=4 * rows * width, dma_store_bytes=2 * rows * width))
    sys.exit(0)
a, b, result, sign = (int(value) for value in args.registers.split(","))
source = "\n".join(rne_add(a, b, result, sign=sign) + ["ecall"]) + "\n"
write(args, f"rne_add-m{a}-m{b}-m{result}", source, dict(family="rne_add", a=a, b=b, result=result, sign=sign))
