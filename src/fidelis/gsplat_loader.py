"""Import gsplat with the environment its JIT build needs.

gsplat compiles its CUDA extension on import, and two things about that need care:

- It wipes its build cache whenever its build flags change, so the flags must be the
  same on every run. Fidelis renders with a pinhole camera and doesn't need 3DGUT
  (nonlinear cameras), whose kernels take over an hour to compile on Windows.
- On Windows, torch looks up MSVC's linker on every load, not just the first build,
  so MSVC must be on PATH even when the extension is already compiled.

Import gsplat through `import_gsplat()` rather than directly. See docs/environment.md.
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


def import_gsplat() -> ModuleType:
    os.environ.setdefault("BUILD_3DGUT", "0")
    if sys.platform == "win32" and shutil.which("cl") is None:
        os.environ.update(_msvc_env())

    import gsplat

    return gsplat


def _msvc_env() -> dict[str, str]:
    """The environment a VS developer shell would have: the newest VS with C++ tools."""
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
    return dict(line.split("=", 1) for line in dump.splitlines() if "=" in line)
