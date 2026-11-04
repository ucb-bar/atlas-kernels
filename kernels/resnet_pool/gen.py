"""Generate the 3x3 stride-two max pool launch: python gen.py --input 64x112x112 --out DIR (channels x height x width)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from kernel_gen import parser, shape, write  # noqa: E402
from atlas_resnet_pool import MaxPool  # noqa: E402

args = parser(__doc__)
args.add_argument("--input", type=shape, default=(64, 112, 112), help="CxHxW of the FP8 channels-last input")
args.add_argument("--arena-base", type=lambda text: int(text, 0), default=0x90000000)
args = args.parse_args()
channels, height, width = args.input
source_bytes = channels * height * width
pool = MaxPool(source_offset=0, constants_offset=source_bytes + 256, output_offset=source_bytes + 256 + 4096 + 256,
               channels=channels, height=height, width=width)
pool.validate()
write(args, f"resnet_pool-{channels}x{height}x{width}", pool.assembly(args.arena_base),
      dict(family="resnet_pool", input=[channels, height, width], output=[channels, pool.output_height, pool.output_width]))
