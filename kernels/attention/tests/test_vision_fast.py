"""The opt-in schedule preserves arithmetic and rejects unsupported masks."""
from dataclasses import replace
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[3]))
import atlas_kernels_path  # noqa: F401,E402
from atlas_attention_v2_program import AttentionV2Config, assembly_v2, stage_v2
from atlas_attention_vision_fast import assembly_v2 as fast
from atlas_execution import Executor


class VisionFast(unittest.TestCase):
    def test_cli_rejects_wide_heads(self):
        generator = Path(__file__).absolute().parents[1] / "gen.py"
        with tempfile.TemporaryDirectory() as temporary:
            for head_dim in (128, 256):
                with self.subTest(head_dim=head_dim):
                    output = Path(temporary) / f"d{head_dim}"
                    command = [sys.executable, str(generator), "--rows", "32", "--keys", "32",
                               "--heads", "1/1", "--head-dim", str(head_dim),
                               "--variant", "vision-fast", "--out", str(output)]
                    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("vision-fast requires --head-dim 64", result.stderr)
                    self.assertFalse(output.exists())

    def test_rejects_unsupported_layouts(self):
        base = AttentionV2Config(32, 32, 2, 2)
        for config in (replace(base, mask="causal"), replace(base, query_rows=31),
                       replace(base, keys=31), replace(base, kv_heads=1),
                       replace(base, output_fp8=True), replace(base, query_rows=1)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                fast(config, {})

    def test_arithmetic_with_nonzero_scales(self):
        import numpy as np
        model = os.environ.get("ATLAS_NPU_MODEL_ROOT")
        if not model:
            self.skipTest("ATLAS_NPU_MODEL_ROOT is required")
        executor = Executor(model)
        rng = np.random.default_rng(638)
        for exponents in ((0, 0, 0), (-2, 1, -1)):
            config = AttentionV2Config(32, 64, 1, 1, q_exponent=exponents[0],
                                       k_exponent=exponents[1], v_exponent=exponents[2])
            def words(rows):
                values = (rng.standard_normal((1, rows, 64)) * 0.25).astype("<f4")
                values[0, 0, :3] = [-10000, 0, 10000]
                # Deterministic BF16 truncation, with explicit clamp cases.
                return (values.view("<u4") >> 16).astype("<u2")
            arena, offsets = stage_v2(config, words(32), words(64), words(64))
            old = executor.execute(assembly_v2(config, offsets), arena, offsets["output"], offsets["output_bytes"])
            new = executor.execute(fast(config, offsets), arena, offsets["output"], offsets["output_bytes"],
                                   expected_arena=old.arena, requested_dma=old.observed)
            self.assertEqual(old.arena, new.arena)


if __name__ == "__main__":
    unittest.main()
