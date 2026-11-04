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
args = args.parse_args()
query_heads, kv_heads = (int(value) for value in args.heads.split("/"))
config = AttentionV2Config(query_rows=args.rows, keys=args.keys, query_heads=query_heads, kv_heads=kv_heads,
                           mask="causal" if args.causal else "none", query_start=args.query_start, output_fp8=args.fp8)
words = lambda heads, rows: (np.arange(heads * rows * 64, dtype=np.uint32) % 251 + 0x3c00).astype("<u2").reshape(heads, rows, 64)
arena, offsets = stage_v2(config, words(query_heads, args.rows), words(kv_heads, args.keys), words(kv_heads, args.keys))
write(args, f"attention-{args.rows}x{args.keys}-{query_heads}h{kv_heads}{'-causal' if args.causal else ''}{'-fp8' if args.fp8 else ''}",
      assembly_v2(config, offsets), dict(family="attention", config=vars(args) | {"out": None}, offsets=offsets, arena_bytes=len(arena)))
