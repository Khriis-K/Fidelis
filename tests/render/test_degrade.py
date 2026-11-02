import numpy as np
import pytest

from fidelis.render.degrade import add_noise, camera_image, expose, motion_blur
from fidelis.scenario import Scenario


def scenario(**overrides):
    values = dict(
        seed=7,
        offset_east_m=0.0,
        offset_north_m=0.0,
        altitude_m=2300.0,
        tilt_deg=0.0,
        tilt_azimuth_deg=0.0,
        heading_deg=0.0,
        noise_sd=0.005,
        blur_px=0.4,
        blur_angle_deg=30.0,
        exposure_gain=1.0,
        sun_index=0,
    )
    return Scenario(**(values | overrides))


def radiance(size=64, seed=0):
    return np.random.default_rng(seed).uniform(0.05, 0.4, (size, size)).astype(np.float32)


def test_nominal_exposure_puts_the_bright_end_of_the_image_at_full_scale():
    # The 99.9th percentile, not the maximum: a few hot pixels mustn't set the exposure.
    image = expose(radiance(), gain=1.0)
    assert np.percentile(image, 99.9) == pytest.approx(1.0)


def test_exposure_gain_scales_the_nominal_exposure():
    r = radiance()
    np.testing.assert_allclose(expose(r, gain=0.6), 0.6 * expose(r, gain=1.0), rtol=1e-6)


def test_a_black_render_stays_black_instead_of_dividing_by_zero():
    image = expose(np.zeros((16, 16), np.float32), gain=1.0)
    assert np.all(image == 0)


def test_zero_blur_leaves_the_image_unchanged():
    r = radiance()
    np.testing.assert_array_equal(motion_blur(r, 0.0, 45.0), r)


def stripes(size=64):
    """Brightness varying along x (columns) only."""
    return np.tile(np.sin(np.arange(size) * 1.3).astype(np.float32), (size, 1)) * 0.4 + 0.5


@pytest.mark.parametrize("angle", [90.0, 270.0])
def test_blur_along_y_leaves_stripes_that_vary_along_x_unchanged(angle):
    # 0 degrees blurs along image x (columns), 90 along image y (rows, downward).
    image = stripes()
    blurred = motion_blur(image, 0.5, angle)
    np.testing.assert_allclose(blurred[8:-8, 8:-8], image[8:-8, 8:-8], atol=1e-5)


def test_blur_along_x_smooths_stripes_that_vary_along_x():
    image = stripes()
    blurred = motion_blur(image, 0.5, 0.0)
    assert np.abs(blurred - image)[8:-8, 8:-8].max() > 0.01
    # A blur only redistributes light: the mean is kept away from the borders.
    assert blurred[8:-8, 8:-8].mean() == pytest.approx(image[8:-8, 8:-8].mean(), abs=1e-3)


def test_noise_has_the_requested_standard_deviation_and_repeats_with_its_generator():
    grey = np.full((256, 256), 0.5, np.float32)
    noisy = add_noise(grey, 0.01, np.random.default_rng(3))
    assert np.std(noisy - grey) == pytest.approx(0.01, rel=0.05)
    np.testing.assert_array_equal(noisy, add_noise(grey, 0.01, np.random.default_rng(3)))


def test_the_camera_image_is_float32_in_zero_to_one():
    image = camera_image(radiance(), scenario(exposure_gain=1.5))
    assert image.dtype == np.float32
    assert image.min() >= 0 and image.max() <= 1
    assert (image == 1).any()  # 1.5 x the nominal exposure clips the brightest pixels


def test_the_same_scenario_gives_the_same_camera_image():
    r = radiance()
    np.testing.assert_array_equal(camera_image(r, scenario()), camera_image(r, scenario()))


def test_another_seed_draws_other_noise():
    r = radiance()
    assert not np.array_equal(camera_image(r, scenario(seed=7)), camera_image(r, scenario(seed=8)))


def test_perfect_conditions_add_no_noise_or_blur():
    r = radiance()
    perfect = scenario(noise_sd=0.0, blur_px=0.0, exposure_gain=1.0)
    np.testing.assert_allclose(camera_image(r, perfect), np.clip(expose(r, 1.0), 0, 1), atol=1e-7)
