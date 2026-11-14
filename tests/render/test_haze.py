import math
import zlib

import numpy as np
import pytest

from fidelis.render.camera import Intrinsics, look_at
from fidelis.render.degrade import add_noise, camera_image, expose, motion_blur
from fidelis.render.haze import add_haze, path_optical_depth, ray_cos_zenith, scenario_transmittance, transmittance
from fidelis.scenario import Scenario
from fidelis.simulation import scenario_pose

K = Intrinsics.from_fov(65, 65, 90.0)  # odd size, so one pixel sits on the optical axis


def scenario(**overrides):
    values = dict(
        seed=7,
        offset_east_m=0.0,
        offset_north_m=0.0,
        altitude_m=2300.0,
        tilt_deg=0.0,
        tilt_azimuth_deg=0.0,
        heading_deg=0.0,
        noise_sd=0.0,
        blur_px=0.0,
        blur_angle_deg=0.0,
        exposure_gain=1.0,
        sun_index=0,
        dust_tau=0.0,
    )
    return Scenario(**(values | overrides))


def radiance(size=65, seed=0):
    return np.random.default_rng(seed).uniform(0.05, 0.4, (size, size)).astype(np.float32)


# --- how much dust lies between the camera and the ground -----------------------------


def test_no_dust_means_no_optical_depth_below_the_camera():
    assert path_optical_depth(0.0, altitude_m=2300.0, scale_height_m=7400.0) == 0.0


def test_the_optical_depth_below_the_camera_is_the_exponential_profiles_share_of_the_column():
    # Dust thinning as exp(-z/H): the share of the column below h is 1 - exp(-h/H).
    expected = 0.8 * (1 - math.exp(-2300.0 / 7400.0))
    assert path_optical_depth(0.8, altitude_m=2300.0, scale_height_m=7400.0) == pytest.approx(expected, rel=1e-12)


def test_a_lower_dust_top_puts_more_of_the_column_below_the_camera():
    shallow = path_optical_depth(1.0, altitude_m=2300.0, scale_height_m=5700.0)
    deep = path_optical_depth(1.0, altitude_m=2300.0, scale_height_m=11000.0)
    assert deep < shallow < 1.0


# --- the slant path through it ---------------------------------------------------------


def nadir_pose(heading_deg=0.0):
    h = math.radians(heading_deg)
    return look_at([0, 0, 2300.0], [0, 0, 0], [math.sin(h), math.cos(h), 0])


def test_a_nadir_cameras_optical_axis_looks_straight_down():
    cos = ray_cos_zenith(nadir_pose(), K)
    assert cos.shape == (65, 65)
    assert cos[32, 32] == pytest.approx(1.0, abs=1e-12)


def test_a_nadir_cameras_corner_ray_is_as_far_off_vertical_as_the_fov_allows():
    # A 90 deg square view: the corner ray is atan(sqrt(2)) off the axis, cos = 1/sqrt(3).
    # Pixel centres sit half a pixel inside the edge, so allow for that.
    cos = ray_cos_zenith(nadir_pose(), K)
    assert cos[0, 0] == pytest.approx(1 / math.sqrt(3), abs=0.01)
    assert cos.min() == cos[0, 0]


def test_tilting_the_camera_tilts_its_optical_axis_off_vertical():
    pose = scenario_pose(scenario(tilt_deg=18.0, tilt_azimuth_deg=70.0), ground_z=0.0)
    assert ray_cos_zenith(pose, K)[32, 32] == pytest.approx(math.cos(math.radians(18.0)), abs=1e-9)


def test_transmittance_falls_with_the_slant_path():
    cos = np.array([1.0, 0.5])
    np.testing.assert_allclose(transmittance(0.2, cos), [math.exp(-0.2), math.exp(-0.4)])


# --- the airlight (Koschmieder) model --------------------------------------------------


def test_haze_blends_each_pixel_towards_the_veil_by_its_transmittance():
    r = radiance()
    hazed = add_haze(r, 0.6)
    veil = float(np.percentile(r, 99.9))
    np.testing.assert_allclose(hazed, 0.6 * r + 0.4 * veil, rtol=1e-6)


def test_full_transmittance_leaves_the_radiance_unchanged():
    r = radiance()
    np.testing.assert_array_equal(add_haze(r, 1.0), r)


def test_haze_scales_the_contrast_by_the_transmittance():
    r = radiance()
    assert np.std(add_haze(r, 0.5)) == pytest.approx(0.5 * np.std(r), rel=1e-5)


def test_contrast_in_the_camera_image_falls_as_the_dust_thickens():
    r = radiance()
    pose = nadir_pose()
    contrasts = []
    for tau in (0.0, 0.5, 1.0, 2.0):
        s = scenario(dust_tau=tau)
        t = scenario_transmittance(s, pose, K, scale_height_m=7400.0)
        contrasts.append(float(np.std(camera_image(r, s, t))))
    assert all(a > b for a, b in zip(contrasts[:-1], contrasts[1:], strict=True)), contrasts


def test_no_dust_reproduces_todays_camera_image_exactly():
    # Today's camera model written out step by step, so a change to camera_image itself
    # can't move both sides of the comparison at once.
    r = radiance()
    s = scenario(dust_tau=0.0, noise_sd=0.005, blur_px=0.4, blur_angle_deg=30.0)
    rng = np.random.default_rng([s.seed, zlib.crc32(b"camera_noise")])
    blurred = motion_blur(r, s.blur_px, s.blur_angle_deg)
    today = np.clip(add_noise(expose(blurred, s.exposure_gain), s.noise_sd, rng), 0, 1)
    t = scenario_transmittance(s, nadir_pose(), K, scale_height_m=7400.0)
    np.testing.assert_array_equal(camera_image(r, s, t), today)


def test_dust_without_a_scale_height_is_refused():
    # Otherwise a simulator built without the config would quietly drop the haze.
    with pytest.raises(ValueError, match="scale height"):
        scenario_transmittance(scenario(dust_tau=0.7), nadir_pose(), K, scale_height_m=None)


def test_no_dust_needs_no_scale_height():
    assert scenario_transmittance(scenario(dust_tau=0.0), nadir_pose(), K, scale_height_m=None) == 1.0


def test_a_scenario_without_dust_tau_has_none():
    # Records from before #37 carry no dust_tau; they had no haze.
    values = {k: v for k, v in scenario().__dict__.items() if k != "dust_tau"}
    assert Scenario(**values).dust_tau == 0.0
