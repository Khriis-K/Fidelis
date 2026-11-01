"""Preflight: check on the submitting machine what a GPU job will need, before it queues.

`uv run fidelis-preflight` runs in seconds and needs no GPU. The workflow runs it as a
localrule ahead of `smoke`, with the same modules, so on CARC it runs on the login node
and a broken module line fails before the job waits hours in the queue.
"""

import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from fidelis.gsplat_loader import msvc_env
from fidelis.smoke import Check
from fidelis.smoke import main as run_and_report

KERNEL = "__global__ void k() {}\nint main() { k<<<1, 1>>>(); }\n"


def cuda_release(nvcc_version: str) -> str:
    """The toolkit version ("13.0") from `nvcc --version` output."""
    match = re.search(r"release (\d+\.\d+)", nvcc_version)
    if match is None:
        raise ValueError(f"no 'release X.Y' line in nvcc --version output: {nvcc_version!r}")
    return match.group(1)


def require_same_cuda_major(nvcc_release: str, torch_cuda: str) -> None:
    """torch refuses to JIT-build extensions (gsplat) with a toolkit of another CUDA major."""
    if nvcc_release.split(".")[0] != torch_cuda.split(".")[0]:
        raise AssertionError(f"nvcc is CUDA {nvcc_release} but torch was built for CUDA {torch_cuda}")


def check_nvcc() -> str:
    """nvcc is on PATH and shares torch's CUDA major version."""
    # Snakemake runs a rule's command even when its module load fails, so this is the
    # check that catches a bad module line.
    if shutil.which("nvcc") is None:
        raise AssertionError("nvcc not on PATH; did the CUDA module load?")
    import torch

    release = cuda_release(subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True).stdout)
    require_same_cuda_major(release, torch.version.cuda)
    return f"nvcc {release}, torch built for CUDA {torch.version.cuda}"


def require_python_headers(include_dir: Path) -> None:
    """torch builds C++ extensions (gsplat's host code) against the interpreter's Python.h."""
    if not (include_dir / "Python.h").is_file():
        raise AssertionError(f"no Python.h in {include_dir}; use a uv-managed Python, which ships headers")


def check_python_headers() -> str:
    """The venv's interpreter has the headers torch's JIT build includes."""
    include_dir = Path(sysconfig.get_paths()["include"])
    require_python_headers(include_dir)
    return f"Python.h in {include_dir}"


def check_nvcc_compiles() -> str:
    """nvcc accepts the host compiler: it compiles a trivial kernel (no GPU needed)."""
    env = os.environ.copy()
    if sys.platform == "win32" and shutil.which("cl") is None:
        env.update(msvc_env())
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "kernel.cu"
        source.write_text(KERNEL)
        result = subprocess.run(
            ["nvcc", "-std=c++17", "-c", str(source), "-o", str(Path(tmp) / "kernel.o")],
            capture_output=True,
            text=True,
            env=env,
        )
    if result.returncode != 0:
        raise AssertionError(f"nvcc failed to compile a trivial kernel:\n{result.stderr.strip()}")
    return "compiled a trivial kernel"


CHECKS: dict[str, Check] = {
    "nvcc": check_nvcc,
    "nvcc compile": check_nvcc_compiles,
    "python headers": check_python_headers,
}


def main() -> int:
    return run_and_report(CHECKS)
