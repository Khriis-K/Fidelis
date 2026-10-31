"""Import gsplat with the environment its JIT build needs.

gsplat JIT-compiles its CUDA extension the first time a CUDA function is called, and
two things about that need care:

- It wipes its build cache whenever its build flags change, so the flags must be the
  same on every run, whatever the shell has set. Fidelis renders with a pinhole camera
  and doesn't need 3DGUT (nonlinear cameras), whose kernels take hours to compile on
  Windows.
- On Windows, torch looks up MSVC's linker on every load, not just the first build,
  so MSVC must be on PATH even when the extension is already compiled.

Import gsplat through `import_gsplat()` rather than directly (ruff enforces this).
See docs/environment.md.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

VSWHERE = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
    r"Microsoft Visual Studio\Installer\vswhere.exe"
)

# Every environment variable gsplat's build reads that changes the compiled extension.
# Values match gsplat's defaults except BUILD_3DGUT; None means "leave unset".
BUILD_FLAGS = {
    "BUILD_3DGUT": "0",
    "DEBUG": "0",
    "FAST_MATH": "1",
    "WITH_SYMBOLS": "0",
    "NVCC_FLAGS": "",
    "BUILD_CAMERA_WRAPPERS": "0",
    "BUILD_3DGS": None,
    "BUILD_2DGS": None,
    "BUILD_ADAM": None,
    "BUILD_RELOC": None,
    "BUILD_LOSSES": None,
    "NUM_CHANNELS": None,
}


def import_gsplat() -> ModuleType:
    for name, value in BUILD_FLAGS.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    if sys.platform == "win32" and shutil.which("cl") is None:
        os.environ.update(_msvc_env())

    import gsplat  # noqa: TID251 -- the one sanctioned import

    return gsplat


def parse_set_output(dump: str) -> dict[str, str]:
    """Parse the NAME=value lines that cmd's `set` prints."""
    return dict(line.split("=", 1) for line in dump.splitlines() if "=" in line)


def _msvc_env() -> dict[str, str]:
    """The environment a VS developer shell would have: the newest VS with C++ tools."""
    if not VSWHERE.exists():
        raise RuntimeError(f"Visual Studio Installer not found at {VSWHERE}; gsplat needs MSVC to load")
    install = subprocess.run(
        [
            str(VSWHERE),
            *["-latest", "-products", "*", "-property", "installationPath"],
            *["-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"],
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not install:
        raise RuntimeError("No Visual Studio with C++ build tools found; gsplat needs MSVC to load")
    vcvars = Path(install) / r"VC\Auxiliary\Build\vcvars64.bat"
    dump = subprocess.run(f'"{vcvars}" >nul && set', shell=True, capture_output=True, text=True, check=True).stdout
    return parse_set_output(dump)
