"""Make every atlas-kernels module importable by its bare name.

Kernel generators and shared helpers import one another by module basename
(``from atlas_program import assembly``). Importing this module appends
``lib/``, ``lib/golden/``, and each ``kernels/<family>/`` directory that holds
Python modules to ``sys.path`` once. Basenames are unique across the library,
so ``module_path(name)`` names the one file that defines a module.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parent
DIRECTORIES = (ROOT / "lib", ROOT / "lib" / "golden",
               *sorted(path for path in (ROOT / "kernels").iterdir() if path.is_dir() and any(path.glob("*.py"))))

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
