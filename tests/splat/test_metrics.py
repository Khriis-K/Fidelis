"""Image quality of splat renders against mesh renders: PSNR and SSIM relative to the camera's
full scale, a gate on both, and a constant-image baseline that the gate must reject."""

import math

import cv2
import numpy as np
import pytest
import torch

from fidelis.render.degrade import EXPOSURE_PERCENTILE
from fidelis.splat.metrics import QualityGate, full_scale, psnr, quality_report, ssim


def texture(size=96, seed=0):
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), np.float32)
    for sigma in (1, 2, 4, 8):
        image += cv2.GaussianBlur(rng.random((size, size), np.float32), (0, 0), sigma)
    return 0.3 * cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)


def as_tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


# --- full scale


def test_full_scale_is_where_the_camera_puts_full_scale():
    # The same percentile degrade.expose sets the nominal exposure from.
    image = texture()
    assert full_scale(image) == pytest.approx(float(np.percentile(image, EXPOSURE_PERCENTILE)))


# --- PSNR


def test_psnr_of_a_uniform_error_of_a_tenth_of_the_peak_is_20_db():
    reference = np.zeros((8, 8), np.float32)
    assert psnr(as_tensor(reference + 0.1), as_tensor(reference), peak=1.0) == pytest.approx(20.0, abs=1e-4)


def test_psnr_is_relative_to_the_peak():
    a, b = texture(seed=1), texture(seed=2)
    assert psnr(as_tensor(5 * a), as_tensor(5 * b), peak=5.0) == pytest.approx(
        psnr(as_tensor(a), as_tensor(b), peak=1.0), abs=1e-3
    )


# --- SSIM


def test_ssim_of_an_image_with_itself_is_one():
    a = as_tensor(texture())
    assert float(ssim(a, a, peak=0.3)) == pytest.approx(1.0, abs=1e-6)


def test_ssim_of_two_flat_images_is_the_luminance_term():
    # With no structure only the luminance term is left: (2 m1 m2 + C1) / (m1^2 + m2^2 + C1),
    # C1 = (0.01 peak)^2 (Wang et al. 2004).
    m1, m2, peak = 0.2, 0.5, 1.0
    c1 = (0.01 * peak) ** 2
    expected = (2 * m1 * m2 + c1) / (m1**2 + m2**2 + c1)
    flat = np.ones((32, 32), np.float32)
    assert float(ssim(as_tensor(m1 * flat), as_tensor(m2 * flat), peak=peak)) == pytest.approx(expected, abs=1e-5)


def test_ssim_is_symmetric_and_scales_with_the_peak():
    a, b = texture(seed=3), texture(seed=4)
    ab = float(ssim(as_tensor(a), as_tensor(b), peak=0.3))
    assert float(ssim(as_tensor(b), as_tensor(a), peak=0.3)) == pytest.approx(ab, abs=1e-6)
    assert float(ssim(as_tensor(4 * a), as_tensor(4 * b), peak=1.2)) == pytest.approx(ab, abs=1e-5)


def test_ssim_falls_as_noise_grows():
    a = texture()
    rng = np.random.default_rng(5)
    values = [float(ssim(as_tensor(a + rng.normal(0, sd, a.shape)), as_tensor(a), peak=0.3)) for sd in (0.003, 0.03)]
    assert 1 > values[0] > values[1]


def test_ssim_is_differentiable_for_training():
    a = as_tensor(texture()).requires_grad_()
    loss = 1 - ssim(a, as_tensor(texture(seed=9)), peak=0.3)
    loss.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0


# --- the gate and the report


GATE = QualityGate(psnr_db_min=30.0, ssim_min=0.9)


def test_a_render_close_to_the_mesh_passes_the_gate():
    mesh = [texture(seed=s) for s in range(3)]
    rng = np.random.default_rng(6)
    splat = [m + rng.normal(0, 0.001, m.shape).astype(np.float32) for m in mesh]
    report = quality_report(splat, mesh, GATE)
    assert report["passed"]
    assert report["worst"]["psnr_db"] > 30 and report["worst"]["ssim"] > 0.9
    assert len(report["views"]) == 3


def test_a_constant_image_fails_the_gate():
    # The null case: a splat that learned only the mean brightness must not pass.
    mesh = [texture(seed=s) for s in range(3)]
    report = quality_report([np.full_like(m, m.mean()) for m in mesh], mesh, GATE)
    assert not report["passed"]


def test_the_report_carries_the_constant_image_baseline_beside_the_splat():
    mesh = [texture(seed=s) for s in range(3)]
    report = quality_report(mesh, mesh, GATE)
    baseline = report["constant_baseline"]
    assert baseline["worst"]["ssim"] < 0.1
    assert baseline["worst"]["psnr_db"] < report["worst"]["psnr_db"]


@pytest.mark.parametrize("view", [{"psnr_db": 45.0, "ssim": 0.5}, {"psnr_db": 20.0, "ssim": 0.99}])
def test_a_view_must_reach_both_bars_not_just_one(view):
    # PSNR forgives blur that SSIM catches, and SSIM forgives a brightness offset PSNR catches.
    assert not GATE.passes(view)
    assert GATE.passes({"psnr_db": 30.0, "ssim": 0.9})


def test_one_bad_view_fails_the_gate():
    mesh = [texture(seed=s) for s in range(3)]
    splat = [m.copy() for m in mesh]
    splat[1] = np.full_like(mesh[1], mesh[1].mean())
    report = quality_report(splat, mesh, GATE)
    assert not report["passed"]
    assert report["worst"]["psnr_db"] == pytest.approx(min(v["psnr_db"] for v in report["views"]))


def test_the_report_is_plain_json_types():
    import json

    mesh = [texture()]
    report = quality_report(mesh, mesh, GATE)
    assert json.loads(json.dumps(report)) == report
    assert not math.isnan(report["median"]["ssim"])
