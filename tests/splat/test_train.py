"""Splat training on a toy scene: a flat textured plane seen from above, ray-cast for ground
truth. The Gaussians start on the plane with the wrong (flat grey) colour; training must
learn the texture well enough to beat that start and a constant image at held-out poses."""

import numpy as np
import pytest
import torch

from fidelis.render.camera import Intrinsics, look_at
from fidelis.splat.metrics import QualityGate, quality_report
from fidelis.splat.model import gaussians
from fidelis.splat.train import TrainSettings, scene_scale, train

PLANE = 60.0  # metres a side
SPACING = 0.5
K = Intrinsics.from_fov(96, 96, 60)


def plane_texture(east, north):
    """A smooth pattern with detail at several scales, in [0.1, 0.5]."""
    return 0.3 + 0.1 * np.sin(east / 2.3) * np.cos(north / 1.7) + 0.1 * np.sin((east + 2 * north) / 5.1)


def ray_cast(pose, intrinsics):
    u, v = np.meshgrid(np.arange(intrinsics.width, dtype=np.float64), np.arange(intrinsics.height, dtype=np.float64))
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(intrinsics.matrix()).T @ pose.rotation.T
    s = -pose.position[2] / rays[..., 2]
    return plane_texture(pose.position[0] + s * rays[..., 0], pose.position[1] + s * rays[..., 1]).astype(np.float32)


def poses(seed, n):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        eye = [rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(25, 32)]
        target = [eye[0] + rng.uniform(-3, 3), eye[1] + rng.uniform(-3, 3), 0.0]
        heading = rng.uniform(0, 2 * np.pi)
        out.append(look_at(eye, target, [np.sin(heading), np.cos(heading), 0.0]))
    return out


def plane_gaussians():
    grid = np.arange(-PLANE / 2, PLANE / 2, SPACING)
    east, north = np.meshgrid(grid, grid)
    means = np.stack([east.ravel(), north.ravel(), np.zeros(east.size)], axis=1)
    return gaussians(means=means, greys=np.full(len(means), 0.3), spacing=np.full(len(means), SPACING))


def test_scene_scale_is_the_camera_spread_with_a_margin():
    # As gsplat's simple_trainer: 1.1 x the farthest camera from the cameras' mean.
    centres = [look_at([x, 0.0, 10.0], [x, 0.0, 0.0], [0, 1, 0]) for x in (-4.0, 0.0, 4.0)]
    assert scene_scale(centres) == pytest.approx(4.4)


def test_settings_come_from_the_scene_configs_splat_section():
    settings = TrainSettings.from_config({"steps": 1234, "ssim_weight": 0.3, "log_every": 50})
    assert (settings.steps, settings.ssim_weight, settings.log_every) == (1234, 0.3, 50)


@pytest.mark.gpu
def test_training_learns_the_texture_and_beats_the_start_and_a_constant_image():
    from fidelis.render.splat_render import render

    train_poses, heldout_poses = poses(0, 40), poses(1, 6)
    images = torch.from_numpy(np.stack([ray_cast(p, K) for p in train_poses]))
    truth = [ray_cast(p, K) for p in heldout_poses]

    start = plane_gaussians()
    with torch.no_grad():
        before = [render({k: v.cuda() for k, v in start.items()}, p, K).cpu().numpy() for p in heldout_poses]
    trained = train(plane_gaussians(), images, train_poses, K, TrainSettings(steps=400, log_every=0))
    with torch.no_grad():
        after = [render(trained, p, K).cpu().numpy() for p in heldout_poses]

    gate = QualityGate(psnr_db_min=0.0, ssim_min=0.0)
    report_before, report_after = quality_report(before, truth, gate), quality_report(after, truth, gate)
    assert report_after["median"]["psnr_db"] > report_before["median"]["psnr_db"] + 5
    assert report_after["worst"]["psnr_db"] > report_after["constant_baseline"]["worst"]["psnr_db"] + 5
    assert report_after["worst"]["ssim"] > 0.8
