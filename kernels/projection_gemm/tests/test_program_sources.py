"""Receipt source resolution follows the library layout and the imported helper."""

from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import atlas_kernels_path  # noqa: E402
import atlas_program  # noqa: E402
import run_model_tile  # noqa: E402


class ProgramSources(unittest.TestCase):
    def test_paths_and_hashes_include_shared_helpers_and_all_recipes(self):
        base = atlas_program.source_paths()
        self.assertEqual(len(base), 9)
        self.assertEqual(base["atlas_program_checks.py"], ROOT / "lib" / "atlas_program_checks.py")
        for reuse in (False, True):
            paths = atlas_program.source_paths(traffic_reuse=reuse)
            expected = set(base)
            if reuse:
                expected.update(name + ".py" for name in atlas_program._TRAFFIC_REUSE_MODULES.values())
            self.assertEqual(set(paths), expected)
            for name, path in paths.items():
                with self.subTest(reuse=reuse, name=name):
                    origin = Path(run_model_tile.__file__) if name == "run_model_tile.py" else atlas_kernels_path.module_path(name)
                    self.assertEqual(path, origin)
            self.assertEqual(atlas_program.code_sources(traffic_reuse=reuse),
                             {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()})

    def run_isolated(self, script, *arguments):
        # Isolate imports from the runner's sys.path and Python environment.
        result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(ROOT), *arguments],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_standalone_hashes_the_fallback_helper(self):
        self.run_isolated("""
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import atlas_kernels_path
import atlas_program
paths = atlas_program.source_paths()
assert paths['run_model_tile.py'] == Path(sys.argv[1]) / 'lib/fallback/tile_admission.py'
assert len(atlas_program.code_sources()) == 9
assert len(atlas_program.code_sources(traffic_reuse=True)) > 9
""")

    def test_external_helper_origin_and_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = Path(temporary) / "run_model_tile.py"
            helper.write_text("def validate_archive_members(members):\n    return None\n")
            self.run_isolated("""
from hashlib import sha256
from pathlib import Path
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import atlas_kernels_path
import atlas_program
helper = Path(sys.argv[2]) / 'run_model_tile.py'
assert atlas_program.source_paths()['run_model_tile.py'] == helper
before = atlas_program.code_sources()['run_model_tile.py']
assert before == sha256(helper.read_bytes()).hexdigest()
helper.write_text(helper.read_text() + '# changed source bytes after import\\n')
after = atlas_program.code_sources()['run_model_tile.py']
assert after == sha256(helper.read_bytes()).hexdigest()
assert after != before
""", temporary)

    def test_source_apis_reject_nonboolean_reuse(self):
        for function in (atlas_program.source_paths, atlas_program.code_sources):
            with self.subTest(function=function.__name__), self.assertRaisesRegex(ValueError, "must be a boolean"):
                function(traffic_reuse=1)

    def test_unqualified_reuse_shape_requires_explicit_default(self):
        m = k = n = 32
        offset = m * k + k * n + 256
        with self.assertRaisesRegex(ValueError, "only for the qualified shapes"):
            atlas_program.assembly(m, k, n, offset, 0, traffic_reuse=True)
        with self.assertRaisesRegex(ValueError, "only for the qualified shapes"):
            atlas_program.dma_load_bytes(m, k, n, traffic_reuse=True)
        self.assertTrue(atlas_program.assembly(m, k, n, offset, 0, traffic_reuse=False))
        self.assertEqual(atlas_program.dma_load_bytes(m, k, n, traffic_reuse=False), 2048)


if __name__ == "__main__":
    unittest.main(verbosity=2)
