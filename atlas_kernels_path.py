"""Make every atlas-kernels module importable by its bare name.

Kernel generators and shared helpers import one another by module basename
(``from atlas_program import assembly``). Importing this module appends
``lib/``, ``lib/golden/``, and each ``kernels/<family>/`` directory that holds
Python modules to ``sys.path`` once. Basenames are unique across the library,
so ``module_path(name)`` names the one file that defines a module.

The generators are byte-identical to the sources that model receipts pin, so
two of them still import atlas-mlir host helpers: ``atlas_program`` imports
``run_model_tile.validate_archive_members`` and the Smol fusion generators import
``chipyard.arena.relocate_assembly``. When atlas-mlir's ``tools/`` is importable
those real modules are used. Otherwise (this repository on its own) a finder
placed last on ``sys.meta_path`` serves the stand-ins in ``lib/fallback/``.
"""

import importlib.abc
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parent
DIRECTORIES = (ROOT / "lib", ROOT / "lib" / "golden",
               *sorted(path for path in (ROOT / "kernels").iterdir() if path.is_dir() and any(path.glob("*.py"))))
FALLBACK = ROOT / "lib" / "fallback"

for _directory in DIRECTORIES:
    if str(_directory) not in sys.path:
        sys.path.append(str(_directory))


def module_path(name):
    """Return the library file for a module file name such as ``atlas_program.py``."""
    for directory in DIRECTORIES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise ValueError("not an atlas-kernels module: " + name)


class StandaloneFallback(importlib.abc.MetaPathFinder):
    """Serve lib/fallback stand-ins for atlas-mlir helpers that no earlier finder provides."""

    FILES = {"run_model_tile": FALLBACK / "tile_admission.py", "chipyard": FALLBACK / "chipyard" / "__init__.py",
             "chipyard.arena": FALLBACK / "chipyard" / "arena.py"}

    def find_spec(self, name, path=None, target=None):
        if name not in self.FILES:
            return None
        locations = [str(FALLBACK / "chipyard")] if name == "chipyard" else None
        return importlib.util.spec_from_file_location(name, self.FILES[name], submodule_search_locations=locations)


if not any(isinstance(finder, StandaloneFallback) for finder in sys.meta_path):
    sys.meta_path.append(StandaloneFallback())
