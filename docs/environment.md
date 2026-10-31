# Environment setup and smoke test

How to build the Fidelis Python environment and confirm every core library works on a GPU.

> **Status (dev laptop, 2026-10-30):** all five libraries pass `fidelis-smoke` on the RTX 5070, from an ordinary shell. CARC is untested.

## Quick start

```bash
git config core.hooksPath .githooks
uv sync
uv run pytest            # GPU-free suite, runs anywhere
uv run fidelis-smoke     # real GPU work in each core library
uv run pytest -m gpu     # the same GPU checks, as tests
```

`fidelis-smoke` prints one `PASS`/`FAIL` line per library and exits non-zero if any fail. Each check does a small piece of real work and compares the result with a known answer:

| Library | What the check does | Ground truth |
|---|---|---|
| torch | 512×512 matmul on the GPU | The same matmul on the CPU |
| mitsuba | Path-traces a 64×64 Cornell box with `cuda_ad_rgb` | Image is finite, lit, and not flat |
| gsplat | Rasterizes one Gaussian straight ahead of the camera | It lands on the principal point |
| lightglue | ALIKED + LightGlue match a synthetic texture to a warped copy | Matches agree with the known homography to within 3 px |
| mxmcpy | MFMC and ACV allocations for three correlated toy models | Variance below plain Monte Carlo at the same cost |

The first LightGlue run downloads the ALIKED and LightGlue weights (from the projects' GitHub releases) into the torch hub cache. The first gsplat GPU call compiles its CUDA extension (about 6 minutes on the laptop) into the torch extensions cache; later runs load it in seconds.

On Windows, gsplat needs Visual Studio 2022 with the C++ build tools installed, but not on `PATH`: Fidelis finds it itself (see below).

## Pinned versions and why

- **Python 3.12** (`.python-version`, `requires-python`). The laptop's global Python is 3.14; uv installs and uses 3.12 instead, which every core library ships wheels for.
- **torch 2.15.0 + CUDA 13.4 (`cu134`)**, from PyTorch's own index on both Windows and Linux. PyPI's Windows torch is CPU-only. `cu134` includes `sm_120` kernels for the Blackwell RTX 5070, and it matches the laptop's CUDA toolkit (13.4), which gsplat's JIT build needs (torch checks that the toolkit and its own CUDA share a major version).
- **MXMCPy from GitHub**, commit `367c7de`, not PyPI. See workarounds below.
- **gsplat from our fork**, [`Khriis-K/gsplat`](https://github.com/Khriis-K/gsplat/tree/fidelis-windows) branch `fidelis-windows`, commit `f18cca6`, not PyPI. See workarounds below.
- **LightGlue from GitHub** (not on PyPI); `uv.lock` pins the commit.
- **uv ≥ 0.12.5** (`required-version`), so the laptop (0.12.5) and CARC (0.13.6) both read the same `uv.lock`.

## Workarounds

### MXMCPy 1.0 on PyPI calls a removed torch function

**Symptom:** `RuntimeError: This function was deprecated since version 1.9 and is now removed. torch.solve ...` from any ACV optimizer.

**Cause:** PyPI's MXMCPy 1.0 is from 2020. Its ACV optimizer calls `torch.solve`, which torch removed. Its dependencies are unpinned, so nothing stops a modern torch from being installed alongside it. The "old dependency pins" risk in the spec showed up as an old *API call* instead.

**Fix:** upstream fixed it in 2022 (switched to `torch.linalg.solve`) but never released to PyPI. `pyproject.toml` installs MXMCPy from GitHub at commit `367c7de` (2024-01-17), which also includes an MLMC fix for models with no samples. Small API note: ACV optimizers return `variance` as a one-element array, MFMC as a scalar.

### gsplat on Windows: unreleased fixes, a fork, and two build settings

gsplat has no prebuilt Windows wheels for torch 2.15 (the published ones stop at torch 2.4), so it compiles its CUDA extension on first import. PyPI's 1.5.3 doesn't build with MSVC and CUDA 13, and neither does upstream `main`. The failures, in the order we hit them:

| Error | Cause | Fixed by |
|---|---|---|
| `cl : D8021 invalid numeric argument '/Wno-attributes'` | 1.5.3 passes a GCC-only flag to MSVC (gsplat issue 809) | upstream `main` |
| `C1189: MSVC/cl.exe with traditional preprocessor is used` | CUDA 13's CCCL headers need `/Zc:preprocessor` | upstream `main` |
| `C1083: Cannot open include file: 'cuda/std/limits'` | CUDA 13 on Windows moved CCCL headers to `include\cccl` | gsplat PR 1085 (unmerged) |
| `calling a __host__ function("isfinite<float>")` | MSVC's `std::isfinite` isn't callable from device code | gsplat PR 1085 (unmerged) |
| `calling a __host__ function("std::array<float, 0>::operator[]")` | MSVC's zero-length `std::array` isn't device-safe | gsplat PR 1055 (unmerged) |

The two PRs fix different bugs and uv installs one commit, so the fork's `fidelis-windows` branch is upstream `main` @ `512d366`, plus PR 1085's three commits, plus PR 1055's two-line `cuda::std::array` fix. **Switch back to an official gsplat release once both land upstream.**

Two settings live in `src/fidelis/gsplat_loader.py`. Import gsplat through `import_gsplat()`, never directly:

- **`BUILD_3DGUT=0`.** gsplat deletes its whole build cache whenever its build flags change, so the flags must match on every run. 3DGUT (unscented-transform projection for fisheye and rolling-shutter cameras) isn't needed for Fidelis's pinhole descent camera, and its kernels were still compiling after more than an hour each on the laptop. `import_gsplat()` sets it (and gsplat's other build flags) outright, so a stray shell variable can't trigger a rebuild. To enable it later, change `BUILD_FLAGS` in `src/fidelis/gsplat_loader.py` and expect a full rebuild.
- **MSVC environment.** On Windows, torch rewrites the extension's ninja file on *every* load and runs `where cl` to find MSVC's linker, so MSVC must be on `PATH` even when nothing needs compiling. When `cl` isn't on `PATH`, the loader asks `vswhere` for the newest Visual Studio with C++ tools and copies the environment its `vcvars64.bat` sets up. A VS developer shell works too; the loader then leaves the environment alone.

Reinstalling gsplat (for example after `uv sync` changes its commit) gives its sources new timestamps, so the next import recompiles everything.

The `_jit_compile() missing 2 required positional arguments` warning sometimes seen in build logs is harmless: gsplat tries torch's old private signature first, catches the error, and retries with the new one.

### Harmless warnings

- `jitc_llvm_init(): LLVM API initialization failed` on importing Mitsuba: Dr.Jit's CPU (LLVM) backend can't find an LLVM install. Fidelis only uses the CUDA variant.

## Open

- **gsplat upstream**: watch PRs 1055 and 1085; once both are released, drop the fork.
- **CARC.** Not yet checked. gsplat will JIT-compile there too. The fork's changes target MSVC, but PR 1085 also swaps a CUB radix sort for `at::sort` on every platform, so the fork is untested on Linux until the cluster smoke test runs. `cu134` torch needs an NVIDIA driver from the R580 series or newer (CUDA 13 minor-version compatibility). Check the driver version with `nvidia-smi` on a GPU node before the cluster smoke test.
