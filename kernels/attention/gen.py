"""Generate one attention v2 launch: python gen.py --rows 128 --keys 128 --heads 32/4 --causal --fp8 --out DIR.

The arena is staged from deterministic BF16 q, k, and v so the program and offsets are reproducible.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from kernel_gen import parser, write  # noqa: E402
import numpy as np  # noqa: E402
from atlas_attention_v2_program import AttentionV2Config, assembly_v2, stage_v2  # noqa: E402

args = parser(__doc__)
args.add_argument("--variant", choices=("v2", "v2-no-halt-delay", "vision-fast", "synchronous-dma"), default="v2",
                  help="explicit generator selection; v2 preserves existing kit programs")
args.add_argument("--rows", type=int, required=True, help="query rows (1 for decode)")
args.add_argument("--keys", type=int, required=True)
args.add_argument("--head-dim", type=int, default=64, choices=(64, 128, 256))
args.add_argument("--heads", default="32/4", help="query heads/KV heads")
args.add_argument("--causal", action="store_true")
args.add_argument("--query-start", type=int, help="first query position (decode)")
args.add_argument("--fp8", action="store_true", help="FP8 output in the projection A layout")
args.add_argument("--mask", choices=("none", "causal", "arbitrary"), help="mask kind (default: causal with --causal, else none)")
args.add_argument("--mask-npy", help="arbitrary mask: NumPy bool array [rows, keys], True where a key is visible")
for name in ("q", "k", "v", "o"):
    args.add_argument(f"--{name}-exponent", type=int, default=0, help=f"FP8 scale exponent of {name}")
args = args.parse_args()
if args.variant == "v2-no-halt-delay" and args.head_dim != 64:
    raise SystemExit("v2-no-halt-delay requires --head-dim 64")
if args.variant == "vision-fast" and args.head_dim != 64:
    raise SystemExit("vision-fast requires --head-dim 64")
if args.variant == "synchronous-dma" and args.head_dim != 64:
    raise SystemExit("synchronous-dma requires --head-dim 64")
if args.variant == "v2-no-halt-delay":
    from atlas_attention_v2_halt import assembly_v2
elif args.variant == "vision-fast":
    from atlas_attention_vision_fast import assembly_v2
query_heads, kv_heads = (int(value) for value in args.heads.split("/"))
mask = args.mask or ("causal" if args.causal else "none")
if (mask == "arbitrary") != (args.mask_npy is not None):
    raise SystemExit("--mask-npy is required with, and only with, --mask arbitrary")
if args.head_dim == 64:
    config_type, stage_attention, assembly_attention = AttentionV2Config, stage_v2, assembly_v2
    extra = {}
else:
    from atlas_attention_wide_program import AttentionWideConfig, stage_wide, assembly_wide
    config_type, stage_attention, assembly_attention = AttentionWideConfig, stage_wide, assembly_wide
    extra = {"head_dim": args.head_dim}
config = config_type(query_rows=args.rows, keys=args.keys, query_heads=query_heads, kv_heads=kv_heads,
                           mask=mask, query_start=args.query_start, output_fp8=args.fp8, q_exponent=args.q_exponent,
                           k_exponent=args.k_exponent, v_exponent=args.v_exponent, o_exponent=args.o_exponent, **extra)


def mask_tiles(allowed):
    padded = np.zeros(((args.rows + 31) // 32 * 32, (args.keys + 31) // 32 * 32), dtype=bool)
    padded[:args.rows, :args.keys] = allowed
    return [[padded[r:r + 32, c:c + 32] for c in range(0, padded.shape[1], 32)] for r in range(0, padded.shape[0], 32)]


tiles = mask_tiles(np.load(args.mask_npy)) if args.mask_npy else None
words = lambda heads, rows: (np.arange(heads * rows * args.head_dim, dtype=np.uint32) % 251 + 0x3c00).astype("<u2").reshape(heads, rows, args.head_dim)
arena, offsets = stage_attention(config, words(query_heads, args.rows), words(kv_heads, args.keys), words(kv_heads, args.keys),
                         mask_tiles=tiles)
metadata_config = vars(args) | {"out": None}
if args.head_dim == 64:
    metadata_config.pop("head_dim")
wide_suffix = "" if args.head_dim == 64 else f"-d{args.head_dim}"
variant_suffix = "-vision-fast" if args.variant == "vision-fast" else ""
source = assembly_attention(config, offsets)
if args.variant == "synchronous-dma":
    from atlas_attention_sync import synchronous_dma
    source = synchronous_dma(source)
    variant_suffix = "-synchronous-dma"
write(args, f"attention-{args.rows}x{args.keys}-{query_heads}h{kv_heads}{'' if mask == 'none' else '-' + mask}{'-fp8' if args.fp8 else ''}{wide_suffix}{variant_suffix}",
      source, dict(family="attention", config=metadata_config, offsets=offsets, arena_bytes=len(arena)))
