"""Generate one projection GEMM launch: python gen.py --shape 128x2048x2048 [--traffic-reuse] --out DIR."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from kernel_gen import parser, shape, write  # noqa: E402
import atlas_program  # noqa: E402

args = parser(__doc__)
args.add_argument("--shape", type=shape, required=True, help="physical MxKxN, each a multiple of 32")
args.add_argument("--rescale-exponent", type=int, default=0, help="output scale 2**E")
mode = args.add_mutually_exclusive_group()
mode.add_argument("--traffic-reuse", action="store_true", help="use the qualified recipe for this exact shape")
mode.add_argument("--resident-a", action="store_true")
mode.add_argument("--resident-b", action="store_true")
args = args.parse_args()
m, k, n = args.shape
flags = dict(traffic_reuse=args.traffic_reuse, resident_a=args.resident_a, resident_b=args.resident_b)
recipe = atlas_program._reuse_module(m, k, n) if args.traffic_reuse else "default"
source = atlas_program.assembly(m, k, n, m * k + k * n + 256, args.rescale_exponent, **flags)
write(args, f"projection_gemm-{m}x{k}x{n}-{recipe.removeprefix('atlas_program_reuse_')}", source,
      dict(family="projection_gemm", mkn=[m, k, n], recipe=recipe, output_offset=m * k + k * n + 256,
           dma_load_bytes=atlas_program.dma_load_bytes(m, k, n, **flags)))
