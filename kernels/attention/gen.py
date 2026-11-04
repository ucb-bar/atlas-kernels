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
args.add_argument("--rows", type=int, required=True, help="query rows (1 for decode)")
args.add_argument("--keys", type=int, required=True)
args.add_argument("--heads", default="32/4", help="query heads/KV heads")
args.add_argument("--causal", action="store_true")
args.add_argument("--query-start", type=int, help="first query position (decode)")
args.add_argument("--fp8", action="store_true", help="FP8 output in the projection A layout")
args.add_argument("--mask", choices=("none", "causal", "arbitrary"), help="mask kind (default: causal with --causal, else none)")
args.add_argument("--mask-npy", help="arbitrary mask: NumPy bool array [rows, keys], True where a key is visible")
for name in ("q", "k", "v", "o"):
    args.add_argument(f"--{name}-exponent", type=int, default=0, help=f"FP8 scale exponent of {name}")
args = args.parse_args()
query_heads, kv_heads = (int(value) for value in args.heads.split("/"))
mask = args.mask or ("causal" if args.causal else "none")
if (mask == "arbitrary") != (args.mask_npy is not None):
    raise SystemExit("--mask-npy is required with, and only with, --mask arbitrary")
config = AttentionV2Config(query_rows=args.rows, keys=args.keys, query_heads=query_heads, kv_heads=kv_heads,
                           mask=mask, query_start=args.query_start, output_fp8=args.fp8, q_exponent=args.q_exponent,
                           k_exponent=args.k_exponent, v_exponent=args.v_exponent, o_exponent=args.o_exponent)


def mask_tiles(allowed):
    padded = np.zeros(((args.rows + 31) // 32 * 32, (args.keys + 31) // 32 * 32), dtype=bool)
    padded[:args.rows, :args.keys] = allowed
    return [[padded[r:r + 32, c:c + 32] for c in range(0, padded.shape[1], 32)] for r in range(0, padded.shape[0], 32)]


tiles = mask_tiles(np.load(args.mask_npy)) if args.mask_npy else None
words = lambda heads, rows: (np.arange(heads * rows * 64, dtype=np.uint32) % 251 + 0x3c00).astype("<u2").reshape(heads, rows, 64)
arena, offsets = stage_v2(config, words(query_heads, args.rows), words(kv_heads, args.keys), words(kv_heads, args.keys),
                         mask_tiles=tiles)
write(args, f"attention-{args.rows}x{args.keys}-{query_heads}h{kv_heads}{'' if mask == 'none' else '-' + mask}{'-fp8' if args.fp8 else ''}",
      assembly_v2(config, offsets), dict(family="attention", config=vars(args) | {"out": None}, offsets=offsets, arena_bytes=len(arena)))
