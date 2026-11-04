"""Generate a q or k projection launch with the RoPE post-pass: python gen.py --shape 128x2048x2048 --out DIR.

As in atlas-mlir tools/tiny_full/fusion.py: the producer retains its BF16 output, and the post-pass reads
one cosine, sine, and negated-sine BF16 tile per row tile staged right after the output.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from kernel_gen import parser, shape, write  # noqa: E402
import atlas_program  # noqa: E402
from atlas_epilogue import Epilogue  # noqa: E402
from atlas_rope_postpass import ResidentRopeEpilogue, append_postpass  # noqa: E402

args = parser(__doc__)
args.add_argument("--shape", type=shape, default=(128, 2048, 2048), help="q (N 2048) or k (N 256); M 32 or 128")
args = args.parse_args()
m, k, n = args.shape
offset = m * k + k * n + 256
common = dict(m=m, n=n, output_offset=offset, scratch_registers=(10, 11))
extra = offset + Epilogue(**common).output_bytes + 256
spec = ResidentRopeEpilogue(**common, cos_offset=extra, sin_offset=extra + 64 * m, neg_sin_offset=extra + 128 * m)
source = append_postpass(atlas_program.assembly(m, k, n, offset, 0, traffic_reuse=True, epilogue=spec), spec)
write(args, f"rope-{m}x{k}x{n}", source, dict(family="rope", mkn=[m, k, n], table_offset=extra))
