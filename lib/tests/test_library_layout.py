"""Every library module has a unique basename and is the file `import` loads."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
import atlas_kernels_path  # noqa: E402


class LibraryLayout(unittest.TestCase):
    def modules(self):
        return [path for directory in atlas_kernels_path.DIRECTORIES for path in sorted(directory.glob("*.py"))]

    def test_basenames_are_unique(self):
        names = [path.name for path in self.modules()]
        self.assertEqual(len(names), len(set(names)), sorted(n for n in names if names.count(n) > 1))
        self.assertGreaterEqual(len(names), 29)

    def test_nothing_earlier_on_sys_path_shadows_a_module(self):
        for path in self.modules():
            with self.subTest(module=path.name):
                spec = importlib.util.find_spec(path.stem)
                self.assertIsNotNone(spec)
                self.assertEqual(Path(spec.origin), path)
                self.assertEqual(atlas_kernels_path.module_path(path.name), path)

    def test_fallbacks_are_not_library_modules(self):
        fallback = atlas_kernels_path.FALLBACK
        self.assertNotIn(fallback, atlas_kernels_path.DIRECTORIES)
        for path in fallback.rglob("*.py"):
            with self.subTest(fallback=path.name):
                self.assertFalse(any((directory / path.name).exists() for directory in atlas_kernels_path.DIRECTORIES))


if __name__ == "__main__":
    unittest.main()
