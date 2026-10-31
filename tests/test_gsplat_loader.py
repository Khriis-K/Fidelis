import os
import sys
import types

import pytest

from fidelis import gsplat_loader
from fidelis.gsplat_loader import import_gsplat, parse_set_output


@pytest.fixture
def env(monkeypatch):
    """A throwaway copy of the environment, so the loader's writes don't leak between tests."""
    copy = dict(os.environ)
    monkeypatch.setattr(os, "environ", copy)
    return copy


@pytest.fixture
def fake_gsplat(monkeypatch):
    """Stand in for gsplat so these tests check the environment, not the CUDA build."""
    module = types.ModuleType("gsplat")
    monkeypatch.setitem(sys.modules, "gsplat", module)
    return module


def test_build_flags_override_whatever_the_shell_set(monkeypatch, env, fake_gsplat):
    monkeypatch.setattr(gsplat_loader.shutil, "which", lambda _: r"C:\msvc\cl.exe")
    env.update(BUILD_3DGUT="1", DEBUG="1", NVCC_FLAGS="-G", NUM_CHANNELS="4")

    assert import_gsplat() is fake_gsplat

    assert env["BUILD_3DGUT"] == "0"
    assert env["DEBUG"] == "0"
    assert env["NVCC_FLAGS"] == ""
    assert "NUM_CHANNELS" not in env


def test_missing_visual_studio_installer_gives_a_clear_error(monkeypatch, tmp_path, env, fake_gsplat):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(gsplat_loader.shutil, "which", lambda _: None)
    missing = tmp_path / "vswhere.exe"
    monkeypatch.setattr(gsplat_loader, "VSWHERE", missing)

    with pytest.raises(RuntimeError, match="Visual Studio") as error:
        import_gsplat()

    assert str(missing) in str(error.value)


def test_parse_set_output_keeps_equals_signs_inside_values():
    dump = "PATH=C:\\a;C:\\b\nODD=x=y\n\nnot a variable\n"

    assert parse_set_output(dump) == {"PATH": "C:\\a;C:\\b", "ODD": "x=y"}
