"""Dust haze between the descent camera and the ground (#37).

The airlight (Koschmieder) model: each pixel's radiance is attenuated along its line of
sight and replaced by scattered light, the veil,

    seen = scene x t + veil x (1 - t),    t = exp(-tau_path / cos(zenith angle of the ray)).

tau_path is the dust optical depth between the camera and the ground, not the whole
column: with dust thinning as exp(-z/H), the share of the column below height h is
1 - exp(-h/H). H comes from the scene config (`haze.dust_scale_height_m`), sourced in
docs/research/dust-vertical-distribution.md.

Two pessimistic choices, both from the prediction on #37:
- All the light the dust removes counts as veil, though Mars dust scatters mostly forward.
- The veil is as bright as the brightest ground (the 99.9th percentile the exposure is set
  from), so exposure doesn't change and contrast falls by exactly t.

Not modelled: dust dimming the direct sun and filling shadows with sky light. That lowers
shading contrast, the minor part of the scene's (#19).
"""

import math

import numpy as np

from fidelis.render.camera import CameraPose, Intrinsics
from fidelis.render.degrade import EXPOSURE_PERCENTILE
from fidelis.scenario import Scenario


def path_optical_depth(tau: float, altitude_m: float, scale_height_m: float) -> float:
    """Dust optical depth between a camera `altitude_m` up and the ground beneath it."""
    return tau * (1 - math.exp(-altitude_m / scale_height_m))


def ray_cos_zenith(pose: CameraPose, intrinsics: Intrinsics) -> np.ndarray:
    """Per pixel, the cosine of the angle between its line of sight and straight down."""
    u, v = np.meshgrid(np.arange(intrinsics.width), np.arange(intrinsics.height))
    rays = np.stack(
        [(u - intrinsics.cx) / intrinsics.fx, (v - intrinsics.cy) / intrinsics.fy, np.ones(u.shape)], axis=-1
    )
    world = rays @ pose.rotation.T
    return -world[..., 2] / np.linalg.norm(world, axis=-1)


def transmittance(path_tau: float, cos_zenith: np.ndarray) -> np.ndarray:
    """The fraction of the ground's light that survives the slant path through the dust."""
    return np.exp(-path_tau / cos_zenith)


def add_haze(radiance: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Radiance blended towards a veil as bright as its brightest ground, by transmittance t."""
    veil = float(np.percentile(radiance, EXPOSURE_PERCENTILE))
    return (radiance * t + veil * (1 - t)).astype(np.float32)


def scenario_transmittance(
    scenario: Scenario, pose: CameraPose, intrinsics: Intrinsics, scale_height_m: float | None
) -> np.ndarray | float:
    """Per-pixel transmittance for this scenario's dust, or 1.0 when it has none."""
    if scenario.dust_tau == 0:
        return 1.0
    if scale_height_m is None:
        raise ValueError("a scenario with dust needs the dust scale height (haze.dust_scale_height_m)")
    path_tau = path_optical_depth(scenario.dust_tau, scenario.altitude_m, scale_height_m)
    cos = ray_cos_zenith(pose, intrinsics)
    if cos.min() <= 0:
        raise ValueError("a line of sight reaches the horizon, so the slant path through the dust is unbounded")
    return transmittance(path_tau, cos)
