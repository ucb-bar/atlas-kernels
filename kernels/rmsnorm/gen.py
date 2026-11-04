"""Generate a standalone RMSNorm launch over retained BF16 rows: python gen.py --rows 128 --out DIR.

This is the TinyLlama norm launch shape (atlas-mlir tools/tiny_full/norm.py): DMA the BF16 h tiles into
VMEM banks 2 and 3, then run the post-pass, which writes FP8 tiles. Width 2048; 128 prefill rows or one
decode row padded to 32. New widths stream 32-row blocks; offsets are output after h, then the 32 diagonal masks and gamma.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from kernel_gen import parser, write  # noqa: E402
from atlas_norm_postpass import H_WORD, ResidentNormEpilogue, postpass  # noqa: E402
from atlas_program_checks import check_matrix_registers  # noqa: E402

WIDTH, MASK_BYTES = 2048, 65536
args = parser(__doc__)
args.add_argument("--rows", type=int, choices=(32, 128), default=128, help="physical rows (32 is one decode row)")
args.add_argument("--width", type=int, default=WIDTH, help="physical row width, a multiple of 32 through 7136")
args.add_argument("--eps", type=float, default=1e-5, help="RMS epsilon, rounded FP32 to finite normal BF16 (default 1e-5)")
args.add_argument("--norm-scale-exponent", type=int, default=-8)
args = args.parse_args()
m, width = args.rows, args.width
h_bytes, output_offset, output_bytes = 2 * m * width, 2 * m * width, m * width
mask_offset = output_offset + output_bytes + 256
from atlas_rmsnorm_tiled import epsilon_bits
epsilon_bits(args.eps)
if width not in (960, 2048) or args.eps != 1e-5:
    from atlas_rmsnorm_tiled import standalone_source
    source = standalone_source(m, width, mask_offset, mask_offset + MASK_BYTES, args.norm_scale_exponent, eps=args.eps)
    write(args, f"rmsnorm-{m}x{width}", source, dict(family="rmsnorm", rows=m, width=width, output_offset=output_offset,
          mask_offset=mask_offset, weight_offset=mask_offset + MASK_BYTES, variant="streamed-tile-rows"))
    sys.exit(0)
epilogue = ResidentNormEpilogue(m, width, 0, scratch_registers=(10, 11), norm_kind="rms", real_n=width,
                                norm_mask_offset=mask_offset, norm_weight_offset=mask_offset + MASK_BYTES,
                                norm_scale_exponent=args.norm_scale_exponent)
lines = ["li x30, 1", "dma.config x30, ch0", f"li x15, {H_WORD}", "li x16, 0", "li x17, 4096",
         f"li x18, {h_bytes // 4096}", "norm_h_load:", "dma.load x15, x16, x17, ch0", "dma.wait ch0",
         "addi x15, x15, 1024", "li x19, 4096", "add x16, x16, x19", "addi x18, x18, -1",
         "bne x18, x0, norm_h_load", "nop"]
source = "\n".join(lines + postpass(epilogue) + ["ecall"]) + "\n"
check_matrix_registers(source)
write(args, f"rmsnorm-{m}x{width}", source, dict(family="rmsnorm", rows=m, width=width, output_offset=output_offset,
      mask_offset=mask_offset, weight_offset=mask_offset + MASK_BYTES))
