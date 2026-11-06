"""Generate the opt-in 128x768x3072 streamed bias/GELU/FP8 fc1 producer."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from atlas_program import assembly  # noqa: E402
from kernel_gen import parser, shape, write  # noqa: E402
from smol_fusion_fast import _epilogue, describe, optimize  # noqa: E402
from smol_weight_relocate import DEFAULT_ARENA, relocate_assembly  # noqa: E402


def main():
    args = parser(__doc__)
    args.add_argument('--shape', type=shape, default=(128, 768, 3072))
    args.add_argument('--rescale-exponent', type=int, default=-19)
    args.add_argument('--output-scale-exponent', type=int, default=-4)
    args.add_argument('--low-arena', action='store_true')
    args = args.parse_args()
    m, k, n = args.shape
    epilogue = _epilogue(m, k, n, args.output_scale_exponent)
    source = assembly(m, k, n, epilogue.output_offset, args.rescale_exponent,
                      traffic_reuse=True, epilogue=epilogue)
    if args.low_arena:
        source = relocate_assembly(source, DEFAULT_ARENA, 0x90000000)
    metadata = describe(source, m, k, n)
    write(args, f'smol_fusion-fast-{m}x{k}x{n}-fp8', optimize(source, m, k, n),
          dict(metadata, family='smol_fusion', variant='streamed-fc1-fp8-v1', mkn=[m, k, n]))


if __name__ == '__main__':
    main()
