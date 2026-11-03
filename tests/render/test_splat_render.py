"""The simulator's "splat" fidelity: a trained Gaussian scene per sun, rendered with gsplat
from a camera pose in Fidelis's one convention (docs/camera-conventions.md)."""

import math

import cv2
import numpy as np
import pytest
import torch

from fidelis.render.camera import Intrinsics, look_at
from fidelis.render.splat_render import SplatRenderer
from fidelis.splat.model import gaussians, save_checkpoint


def test_a_sun_with_no_trained_scene_is_a_clear_error(tmp_path):
    renderer = SplatRenderer({0: tmp_path / "sun0" / "splat.pt"}, device="cpu")
    with pytest.raises(ValueError, match=r"sun 2.*\[0\]"):
        renderer.use_sun(2)


def test_the_renderer_says_which_suns_it_has(tmp_path):
    renderer = SplatRenderer({3: tmp_path / "a.pt", 0: tmp_path / "b.pt"}, device="cpu")
    assert renderer.suns == [0, 3]


@pytest.mark.gpu
def test_gaussians_land_where_opencv_projects_them(tmp_path):
    # Like the mesh renderer's projection test: an oblique, rolled camera with an off-centre
    # principal point, small bright Gaussians, and each one's image centroid against
    # cv2.projectPoints of its centre. Pins the pose, axes and half-pixel convention together.
    points = np.array([[0.0, 0.0, 0.0], [6.0, -4.0, 1.0], [-5.0, 3.0, -1.0]])
    params = gaussians(means=points, greys=np.ones(3), spacing=np.full(3, 0.15))
    params["opacities"] = torch.full((3,), 6.0)  # sigmoid(6) ~ 0.998: solid
    save_checkpoint(tmp_path / "splat.pt", params, {"sun_index": 0})

    pose = look_at(eye=[8.0, -20.0, 25.0], target=[1.0, 0.5, 0.0], up=[0.3, 1.0, 0.2])
    k = Intrinsics(width=320, height=240, fx=300.0, fy=300.0, cx=150.3, cy=131.8)
    image = SplatRenderer({0: tmp_path / "splat.pt"})(pose, k, sun_index=0, seed=0)
    assert image.shape == (240, 320) and image.dtype == np.float32

    w2c = pose.world_to_camera()
    expected, _ = cv2.projectPoints(points, cv2.Rodrigues(w2c[:3, :3])[0], w2c[:3, 3], k.matrix(), None)
    ys, xs = np.mgrid[0:240, 0:320]
    for u, v in expected[:, 0]:
        window = (np.abs(xs - u) < 8) & (np.abs(ys - v) < 8)
        weights = image * window
        assert weights.sum() > 0, f"nothing rendered near ({u:.1f}, {v:.1f})"
        centroid = (float((weights * xs).sum() / weights.sum()), float((weights * ys).sum() / weights.sum()))
        assert math.dist(centroid, (u, v)) < 0.1


@pytest.mark.gpu
def test_the_same_pose_renders_the_same_image(tmp_path):
    rng = np.random.default_rng(0)
    params = gaussians(means=rng.uniform(-10, 10, (500, 3)), greys=rng.random(500), spacing=np.full(500, 0.8))
    save_checkpoint(tmp_path / "splat.pt", params, {"sun_index": 0})
    renderer = SplatRenderer({0: tmp_path / "splat.pt"})
    pose = look_at(eye=[0.0, 0.0, 40.0], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    k = Intrinsics.from_fov(128, 128, 60)
    once, again = renderer(pose, k, 0, seed=1), renderer(pose, k, 0, seed=2)
    assert once.max() > 0
    # Same tolerance as the mesh renderer's GPU jitter (fidelis.render.mesh_render).
    assert np.abs(once - again).max() < 1e-6
