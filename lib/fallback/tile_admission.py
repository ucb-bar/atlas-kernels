"""Standalone stand-in for atlas-mlir's run_model_tile, served only when that module is not importable.

atlas_program.preflight_bundle imports validate_archive_members from atlas-mlir tools/run_model_tile.py.
atlas_kernels_path serves this file under the name run_model_tile only after every normal import path has
failed, so atlas-mlir's module always wins. The function is a verbatim copy; atlas-mlir's
test/test_kernel_library.py checks that the two stay identical.
"""


def validate_archive_members(members):
    required = {"A.npy", "B.npy", "native_C.npy"}
    optional = {"A_bf16.npy", "B_bf16.npy", "native_C_bf16.npy"}
    names = [member.filename for member in members]
    if len(names) != len(set(names)) or not required <= set(names) or set(names) - required - optional:
        raise ValueError("NPZ requires unique flat A/B/native_C.npy and only optional *_bf16.npy members")
    if (any(member.file_size < 0 for member in members) or sum(member.file_size for member in members) > 32 * 1024 * 1024):
        raise ValueError("NPZ exceeds bounded uncompressed size")
