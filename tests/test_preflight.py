import pytest

from fidelis.preflight import check_nvcc, cuda_release, require_same_cuda_major

# `nvcc --version` on CARC after `module load ver/2607 gcc/15.3.0 cuda/13.0.3` (2026-10-31).
CARC_NVCC_VERSION = """\
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
"""


def test_cuda_release_reads_the_toolkit_version_from_nvcc():
    assert cuda_release(CARC_NVCC_VERSION) == "13.0"


def test_cuda_release_rejects_output_without_a_release_line():
    with pytest.raises(ValueError, match="release"):
        cuda_release("nvcc: command output we don't recognise")


def test_a_toolkit_with_torchs_cuda_major_passes():
    # CARC's 13.0 toolkit against torch's cu134 build.
    require_same_cuda_major("13.0", "13.4")


def test_a_toolkit_with_another_cuda_major_fails():
    # The trap on CARC: every toolkit except cuda/13.0.3 is 12.x or older.
    with pytest.raises(AssertionError, match="12.9.*13.4"):
        require_same_cuda_major("12.9", "13.4")


def test_check_nvcc_fails_when_nvcc_is_not_on_path(monkeypatch, tmp_path):
    # What a job sees when its CUDA module failed to load.
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(AssertionError, match="nvcc not on PATH"):
        check_nvcc()
