#!/usr/bin/env python3
"""Select the paired-row fast core for a generated family assembly program."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402,F401
from atlas_path_guard import checked  # noqa: E402
from atlas_program_gemm_fast import optimize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    for dim in ('m', 'k', 'n'):
        parser.add_argument('--' + dim, type=int, required=True)
    args = parser.parse_args()
    source = checked(args.program).read_text()
    result = optimize(source, args.m, args.k, args.n)
    target = checked(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result)


if __name__ == '__main__':
    main()
