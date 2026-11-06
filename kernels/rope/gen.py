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
shapes = args.add_mutually_exclusive_group()
shapes.add_argument("--shape", type=shape, help="projection MxKxN; default 128x2048x2048")
shapes.add_argument("--postpass-shape", type=shape, help="opt-in standalone BF16 physical MxN, head64")
args.add_argument("--rescale-exponent", type=int, default=0, help="projection output scale 2**E")
parsed = args.parse_args()
if parsed.postpass_shape is not None:
    if len(parsed.postpass_shape) != 2:
        args.error('--postpass-shape requires physical MxN')
    if parsed.rescale_exponent != 0:
        args.error('--rescale-exponent must be 0 for standalone BF16 RoPE')
    from atlas_rope_standalone import assembly, layout
    m, n = parsed.postpass_shape
    try:
        metadata = layout(m, n)
        source = assembly(m, n)
    except ValueError as error:
        args.error(str(error))
    write(parsed, f"rope-postpass-{m}x{n}", source,
          dict(metadata, family="rope", variant="standalone-bf16-postpass"))
    sys.exit(0)
if parsed.shape is not None and len(parsed.shape) != 3:
    args.error('--shape requires projection MxKxN')
args = parsed
m, k, n = (128, 2048, 2048) if args.shape is None else args.shape
offset = m * k + k * n + 256
common = dict(m=m, n=n, output_offset=offset, scratch_registers=(10, 11))
extra = offset + Epilogue(**common).output_bytes + 256
spec = ResidentRopeEpilogue(**common, cos_offset=extra, sin_offset=extra + 64 * m, neg_sin_offset=extra + 128 * m)
source = append_postpass(atlas_program.assembly(m, k, n, offset, args.rescale_exponent, traffic_reuse=True, epilogue=spec), spec)
write(args, f"rope-{m}x{k}x{n}", source, dict(family="rope", mkn=[m, k, n], table_offset=extra, output_offset=offset,
      recipe=atlas_program._reuse_module(m, k, n)))
