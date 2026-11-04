"""Generate a convolution GEMM whose A rows are gathered on device: python gen.py --input 64x56x56 --kernel 3 --n 64 --out DIR.

The input is FP8 channels-last CxHxW; M is the output positions padded to 32 and K is kernel*kernel*C. kxk
gathers need the blocked recipe, so the shape must be one atlas_program_reuse_blocked admits (ResNet50's are).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from kernel_gen import parser, shape, write  # noqa: E402
import atlas_program  # noqa: E402
from atlas_resnet_input import DirectInput  # noqa: E402

args = parser(__doc__)
args.add_argument("--input", type=shape, default=(64, 56, 56), help="CxHxW of the FP8 channels-last input")
args.add_argument("--kernel", type=int, default=3, choices=(1, 3))
args.add_argument("--stride", type=int, default=1, choices=(1, 2))
args.add_argument("--n", type=int, default=64, help="output channels")
args.add_argument("--default-schedule", action="store_true", help="1x1 only: skip the blocked traffic-reuse recipe")
args = args.parse_args()
channels, height, width = args.input
probe = DirectInput(0, channels, height, width, stride=args.stride, kernel=args.kernel, padding=args.kernel // 2)
direct = DirectInput(0, channels, height, width, stride=args.stride, kernel=args.kernel, padding=args.kernel // 2,
                     zero_offset=probe.source_bytes)
m = -(-direct.output_rows // 32) * 32
k, n = args.kernel * args.kernel * channels, args.n
offset = m * k + k * n + 256
source = atlas_program.assembly(m, k, n, offset, 0, traffic_reuse=not args.default_schedule, direct_input=direct)
write(args, f"resnet_patch_gather-{channels}x{height}x{width}-k{args.kernel}s{args.stride}-n{n}", source,
      dict(family="resnet_patch_gather", input=[channels, height, width], kernel=args.kernel, stride=args.stride, mkn=[m, k, n]))
