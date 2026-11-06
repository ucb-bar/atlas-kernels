#!/usr/bin/env python3
"""Generate an explicitly selected Qwen panel with exact bias or residual add."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from atlas_path_guard import checked  # noqa: E402
from atlas_program_qwen_fused import ROLES, assembly, layout  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=ROLES, required=True)
    for dimension in ('m', 'k', 'n'):
        parser.add_argument('--' + dimension, type=int, required=True)
    parser.add_argument('--rescale-exponent', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = assembly(args.role, args.m, args.k, args.n, args.rescale_exponent)
    path = checked(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    print(json.dumps(layout(args.role, args.m, args.k, args.n), sort_keys=True))


if __name__ == '__main__':
    main()
