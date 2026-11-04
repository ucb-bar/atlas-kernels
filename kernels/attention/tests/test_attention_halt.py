"""Keep the halt variant's delta to one instruction; execute only small cases."""

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).absolute().parents[3]
sys.path.insert(0, str(ROOT))
import atlas_kernels_path  # noqa: E402,F401
import atlas_attention_v2_program as original  # noqa: E402
import atlas_attention_v2_halt as halt  # noqa: E402

MODEL_ROOT = os.environ.get("ATLAS_NPU_MODEL_ROOT")
if MODEL_ROOT:
    sys.path.insert(0, MODEL_ROOT)
try:
    from npu_model.util.converter import input_to_program
except ModuleNotFoundError:
    input_to_program = None

OLD_TAIL = "dma.wait ch0\ndma.wait ch1\ndelay 128\necall\n"
NEW_TAIL = "dma.wait ch0\ndma.wait ch1\necall\n"


def fixture(rows, keys, heads, kv_heads, *, mask="none", prefix=False, **kwargs):
    config = original.AttentionV2Config(rows, keys, heads, kv_heads, mask=mask, **kwargs)
    # Distinct bounded BF16 values exercise quantization and both output formats.
    words = lambda h, r, shift: (0x3d00 + (np.arange(h * r * 64, dtype=np.uint32) + shift) % 512).astype("<u2").reshape(h, r, 64)
    tiles = None
    if mask == "arbitrary":
        allowed = np.zeros((config.query_tiles * 32, config.key_tiles * 32), dtype=bool)
        allowed[:rows, :keys] = True
        if prefix:
            allowed[:rows - 1, keys - 1] = False
        tiles = [[allowed[r:r + 32, k:k + 32] for k in range(0, allowed.shape[1], 32)]
                 for r in range(0, allowed.shape[0], 32)]
    arena, offsets = original.stage_v2(config, words(heads, rows, 0), words(kv_heads, keys, 71),
                                       words(kv_heads, keys, 137), mask_tiles=tiles)
    return config, arena, offsets


def real_fixtures():
    yield "tl-prefill", fixture(128, 128, 32, 4, mask="causal", query_start=0, output_fp8=True)
    for keys in (129, 130):
        yield f"tl-decode-{keys}", fixture(1, keys, 32, 4, mask="causal", query_start=keys - 1, output_fp8=True)
    yield "sv-vision", fixture(1024, 1024, 12, 12)
    yield "sv-prefix", fixture(177, 177, 15, 5, mask="arbitrary", prefix=True)
    for keys in (227, 177):
        yield f"sv-expert-{keys}", fixture(50, keys, 15, 5, mask="arbitrary")


class AttentionHaltSourceTests(unittest.TestCase):
    def test_legacy_generator_remains_pinned(self):
        self.assertEqual(hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
                         "16a6e96e5cbb801d7fe58a53c37e15d18fd098a2e72daae16abd337b3c97a774")

    def test_source_delta_across_seven_real_shapes(self):
        for name, (config, _, offsets) in real_fixtures():
            with self.subTest(case=name):
                before = original.assembly_v2(config, offsets)
                after = halt.assembly_v2(config, offsets)
                self.assertTrue(before.endswith(OLD_TAIL))
                self.assertTrue(after.endswith(NEW_TAIL))
                self.assertEqual(after, before[:-len(OLD_TAIL)] + NEW_TAIL)
                self.assertEqual(after.splitlines(), before.splitlines()[:-2] + ["ecall"])

    def test_forwards_config_offsets_and_dram_base(self):
        config, _, offsets = fixture(2, 33, 2, 1)
        for base in (0x90000000, 0x80010000):
            with self.subTest(base=base), patch.object(halt, "_original_assembly_v2", wraps=original.assembly_v2) as wrapped:
                after = halt.assembly_v2(config, offsets, dram_base=base)
                wrapped.assert_called_once_with(config, offsets, dram_base=base)
                self.assertEqual(after, original.assembly_v2(config, offsets, dram_base=base)[:-len(OLD_TAIL)] + NEW_TAIL)
        with self.assertRaises(ValueError):
            halt.assembly_v2(config, offsets, dram_base=0x90000001)

    def test_changed_tail_fails_closed(self):
        config, _, offsets = fixture(1, 1, 1, 1)
        source = original.assembly_v2(config, offsets)
        for changed in (source.replace(OLD_TAIL, NEW_TAIL), source[:-1], source + "nop\n",
                        source.replace(OLD_TAIL, "dma.wait ch1\ndma.wait ch0\ndelay 128\necall\n"),
                        source.replace(OLD_TAIL, OLD_TAIL.replace("128", "127")),
                        source.replace(OLD_TAIL, "dma.wait ch0\ndelay 128\necall\n")):
            with self.subTest(tail=changed[-70:]), patch.object(halt, "_original_assembly_v2", return_value=changed):
                with self.assertRaisesRegex(ValueError, "halt tail changed"):
                    halt.assembly_v2(config, offsets)


@unittest.skipUnless(input_to_program is not None, "npu_model assembler required")
class AttentionHaltModelTests(unittest.TestCase):
    def test_word_delta_across_seven_real_shapes(self):
        delay = input_to_program(io.StringIO("delay 128\necall\n")).assemble()[0]
        for name, (config, _, offsets) in real_fixtures():
            with self.subTest(case=name):
                before = list(input_to_program(io.StringIO(original.assembly_v2(config, offsets))).assemble())
                after = list(input_to_program(io.StringIO(halt.assembly_v2(config, offsets))).assemble())
                self.assertEqual(before[-2], delay)
                self.assertEqual(after, before[:-2] + before[-1:])

    def test_small_rtl_execution_matches_output_and_preserved_arena(self):
        from atlas_rtl_reference import RtlReference
        for packed in (False, True):
            config, arena, offsets = fixture(1 if packed else 2, 33, 2, 1, mask="causal",
                                              query_start=32 if packed else 0, output_fp8=packed)
            results = []
            for generator in (original.assembly_v2, halt.assembly_v2):
                reference = RtlReference(0x90000000)
                try:
                    results.append(reference.run(generator(config, offsets), arena, offsets["output"],
                                                 offsets["output_bytes"], {}))
                finally:
                    reference.close()
            with self.subTest(packed=packed):
                self.assertEqual(results[0][0], results[1][0])
                self.assertNotEqual(results[1][0], arena[offsets["output"]:offsets["output"] + offsets["output_bytes"]])
                self.assertEqual(results[0][1]["physical_instructions"], results[1][1]["physical_instructions"] + 1)
                self.assertEqual(results[0][1]["observed_dma"], results[1][1]["observed_dma"])
                self.assertEqual(results[1][1]["numerics"], "rtl")
                self.assertTrue(results[1][1]["preserved"])

    def test_cli_default_and_explicit_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            outputs = []
            for variant in (None, "v2", "v2-no-halt-delay"):
                directory = Path(temporary) / (variant or "default")
                command = [sys.executable, str(ROOT / "kernels/attention/gen.py"), "--rows", "1", "--keys", "33",
                           "--heads", "2/1", "--causal", "--fp8", "--name", "probe", "--out", str(directory)]
                if variant:
                    command += ["--variant", variant]
                env = os.environ.copy()
                env["PYTHONPATH"] = os.pathsep.join(filter(None, [MODEL_ROOT, env.get("PYTHONPATH")]))
                subprocess.run(command, env=env, check=True, capture_output=True, text=True, timeout=60)
                source = (directory / "probe.S").read_text()
                binary = (directory / "probe.bin").read_bytes()
                metadata = json.loads((directory / "probe.json").read_text())
                self.assertEqual(metadata["config"]["variant"], variant or "v2")
                self.assertEqual(metadata["source_sha256"], hashlib.sha256(source.encode()).hexdigest())
                self.assertEqual(metadata["binary_sha256"], hashlib.sha256(binary).hexdigest())
                outputs.append((source, binary, metadata))
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[2][0], outputs[0][0][:-len(OLD_TAIL)] + NEW_TAIL)
            self.assertEqual(outputs[2][1], outputs[0][1][:-8] + outputs[0][1][-4:])
            self.assertEqual(outputs[2][2]["words"], outputs[0][2]["words"] - 1)


if __name__ == "__main__":
    unittest.main()
