"""Standalone stand-in for atlas-mlir's chipyard.arena, served only when that package is not importable.

smol_fusion_program and smol_fusion_fp8 import relocate_assembly from atlas-mlir tools/chipyard/arena.py.
lib/smol_weight_relocate.py holds the same relocation, extended with an optional weight source that these
callers do not pass. atlas-mlir's test/test_kernel_library.py checks that both give identical programs.
"""

from smol_weight_relocate import relocate_assembly as _relocate


def relocate_assembly(source, old_base, new_base, *, force=False):
    """Keep local offsets/VMEM operands, change high32 and displace DMA bytes."""
    return _relocate(source, old_base, new_base, force=force)
