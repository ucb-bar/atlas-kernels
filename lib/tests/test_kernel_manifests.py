"""Every family has a README, a kernel.yaml manifest that matches the code, and a working generator."""
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT))
import atlas_kernels_path  # noqa: E402,F401
import atlas_program  # noqa: E402

FAMILIES = sorted(path.name for path in (ROOT / "kernels").iterdir() if path.is_dir())
GENERATE = {"projection_gemm": ["--shape", "128x2048x2048", "--traffic-reuse"],
            "epilogue": ["--shape", "128x2048x1536", "--steps", "silu"], "rne_add": [], "rmsnorm": ["--rows", "32"],
            "rope": ["--shape", "32x2048x256"], "attention": ["--rows", "1", "--keys", "129", "--heads", "32/4", "--causal",
            "--query-start", "128", "--fp8"], "smol_fusion": ["--shape", "128x768x3072", "--fp8"],
            "resnet_pool": ["--input", "64x16x16"], "resnet_patch_gather": ["--input", "64x56x56", "--kernel", "1", "--n", "256"]}


def manifest(family):
    import yaml
    return yaml.safe_load((ROOT / "kernels" / family / "kernel.yaml").read_text())


class KernelManifests(unittest.TestCase):
    def test_every_family_has_readme_manifest_and_generator(self):
        self.assertEqual(sorted(GENERATE), FAMILIES)
        for family in FAMILIES:
            for name in ("README.md", "kernel.yaml", "gen.py"):
                with self.subTest(family=family, file=name):
                    self.assertTrue((ROOT / "kernels" / family / name).is_file())

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is needed to read kernel.yaml")
    def test_manifest_entry_points_exist(self):
        for family in FAMILIES:
            data = manifest(family)
            self.assertEqual((data["schema"], data["family"]), ("atlas.kernel.v1", family))
            for entry in data["entry_points"]:
                with self.subTest(family=family, entry=entry["function"]):
                    self.assertTrue(hasattr(importlib.import_module(entry["module"]), entry["function"]))

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is needed to read kernel.yaml")
    def test_projection_variants_are_the_dispatched_recipes(self):
        variants = manifest("projection_gemm")["variants"]
        recipes = {variant["module"] for variant in variants} - {"atlas_program"}
        self.assertEqual(recipes, set(atlas_program._TRAFFIC_REUSE_MODULES.values()))
        for variant in variants:
            for use in variant["uses"]:
                with self.subTest(variant=variant["name"], mkn=use["mkn"]):
                    try:
                        dispatched = atlas_program._reuse_module(*use["mkn"])
                    except ValueError:
                        dispatched = "atlas_program"
                    self.assertEqual(dispatched, variant["module"])

    @unittest.skipUnless(importlib.util.find_spec("npu_model"), "npu_model is needed to assemble programs")
    def test_generators_write_source_and_assembled_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            for family, arguments in GENERATE.items():
                with self.subTest(family=family):
                    out = Path(tmp) / family
                    subprocess.run([sys.executable, "-B", str(ROOT / "kernels" / family / "gen.py"), "--out", str(out),
                                    "--name", "program", *arguments], check=True, capture_output=True)
                    source, binary = (out / "program.S").read_text(), (out / "program.bin").read_bytes()
                    self.assertTrue(source.rstrip().endswith("ecall"))
                    self.assertGreater(len(binary), 0)
                    self.assertEqual(len(binary) % 4, 0)


if __name__ == "__main__":
    unittest.main()
