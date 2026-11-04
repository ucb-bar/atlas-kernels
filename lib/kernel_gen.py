"""Shared command-line plumbing for kernels/<family>/gen.py: write a program as .S and assembled .bin.

The .bin holds the 32-bit instruction words that npu_model's own assembler produces, little endian, in
order. npu_model must be importable (put its checkout on PYTHONPATH, as `make gen-<family>` does).
"""
import argparse
import hashlib
import io
import json
from pathlib import Path


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--out", type=Path, required=True, help="output directory (make uses build/<family>)")
    result.add_argument("--name", help="output file stem (default: derived from the arguments)")
    return result


def shape(text):
    """Parse MxKxN (or any x-separated list of positive integers)."""
    values = tuple(int(part) for part in text.lower().split("x"))
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("shape must be positive integers joined by x, for example 128x2048x2048")
    return values


def write(args, stem, source, metadata=None):
    from npu_model.util.converter import input_to_program
    words = input_to_program(io.StringIO(source)).assemble()
    raw = b"".join(word.to_bytes(4, "little") for word in words)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stem = args.name or stem
    (out / f"{stem}.S").write_text(source)
    (out / f"{stem}.bin").write_bytes(raw)
    record = dict(metadata or {}, source=f"{stem}.S", binary=f"{stem}.bin", words=len(words),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(), binary_sha256=hashlib.sha256(raw).hexdigest())
    (out / f"{stem}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))
    return record
