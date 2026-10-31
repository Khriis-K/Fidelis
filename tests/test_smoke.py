import pytest

from fidelis.smoke import CHECKS, main, run_checks


def _passes():
    return "ok"


def _raises():
    raise RuntimeError("no kernel image is available for execution on the device")


def test_a_failing_check_does_not_stop_the_others():
    results = run_checks({"alpha": _raises, "beta": _passes})

    assert [(r.name, r.passed) for r in results] == [("alpha", False), ("beta", True)]
    assert "no kernel image" in results[0].detail


def test_main_reports_each_library_and_fails_if_any_check_fails(capsys):
    exit_code = main({"alpha": _raises, "beta": _passes})

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL  alpha" in out
    assert "PASS  beta" in out


def test_main_exits_zero_when_every_check_passes(capsys):
    assert main({"alpha": _passes, "beta": _passes}) == 0


def test_registry_checks_every_core_library():
    assert set(CHECKS) == {"torch", "mitsuba", "gsplat", "lightglue", "mxmcpy"}


@pytest.mark.gpu
@pytest.mark.parametrize("library", ["torch", "mitsuba", "gsplat", "lightglue"])
def test_gpu_library_check_passes_on_this_machine(library):
    CHECKS[library]()  # raises if the library's GPU result disagrees with its ground truth


def test_mxmcpy_check_passes():
    CHECKS["mxmcpy"]()
