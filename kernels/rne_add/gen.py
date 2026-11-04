"""Generate the exact BF16 add on one register pair as a minimal program: python gen.py --out DIR.

The sequence is an epilogue stage, not a launch by itself; this program wraps it with ecall so the
assembler encodes exactly the 8 constant loads and 34 instructions (each followed by delay 66).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from kernel_gen import parser, write  # noqa: E402
from atlas_rne_add import rne_add  # noqa: E402

args = parser(__doc__)
args.add_argument("--registers", default="0,2,52,4", help="a,b,result,sign even matrix-register pairs")
args = args.parse_args()
a, b, result, sign = (int(value) for value in args.registers.split(","))
source = "\n".join(rne_add(a, b, result, sign=sign) + ["ecall"]) + "\n"
write(args, f"rne_add-m{a}-m{b}-m{result}", source, dict(family="rne_add", a=a, b=b, result=result, sign=sign))
