import math

import cv2
import numpy as np
import pytest

from fidelis.nav.pnp import MIN_CORRESPONDENCES, solve_pose
from fidelis.render.camera import Intrinsics, look_at

K = Intrinsics.from_fov(1024, 1024, 90)
# Tilted, rolled, 2.3 km above rough ground: not a pose a sign or transpose slip could reproduce.
TRUE_POSE = look_at(eye=[120.0, -340.0, -250.0], target=[300.0, -200.0, -2550.0], up=[0.3, 1.0, 0.0])


def terrain_points(n, seed=0):
    """Ground points the camera sees: a 4 km patch with 250 m of relief, like the Jezero tile."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1800, 2200, size=(n, 2)) + [300.0, -200.0]
    z = -2550.0 + 125.0 * np.sin(xy[:, 0] / 700.0) * np.cos(xy[:, 1] / 900.0)
    world = np.column_stack([xy, z])
    w2c = TRUE_POSE.world_to_camera()
    image, _ = cv2.projectPoints(world, cv2.Rodrigues(w2c[:3, :3])[0], w2c[:3, 3], K.matrix(), None)
    image = image.reshape(-1, 2)
    seen = (image >= 0).all(axis=1) & (image < 1024).all(axis=1)
    return image[seen], world[seen]


def test_exact_correspondences_recover_the_camera_pose():
    image, world = terrain_points(200)
    solution = solve_pose(image, world, K)
    assert solution.status == "ok"
    np.testing.assert_allclose(solution.pose.position, TRUE_POSE.position, atol=1e-3)
    np.testing.assert_allclose(solution.pose.rotation, TRUE_POSE.rotation, atol=1e-6)
    assert solution.inliers.all()


def test_gross_outliers_are_rejected_and_flagged_in_the_inlier_mask():
    image, world = terrain_points(300)
    rng = np.random.default_rng(1)
    outlier = rng.random(len(image)) < 0.4
    # A wrong match lands anywhere in the image, far from where the point projects.
    image[outlier] = rng.uniform(0, 1024, size=(outlier.sum(), 2))
    far = np.linalg.norm(image - terrain_points(300)[0], axis=1) > 50

    solution = solve_pose(image, world, K)

    assert solution.status == "ok"
    np.testing.assert_allclose(solution.pose.position, TRUE_POSE.position, atol=1e-2)
    assert not solution.inliers[far].any()
    assert solution.inliers[~outlier].all()


def test_the_same_correspondences_give_the_same_pose_every_time():
    image, world = terrain_points(300)
    rng = np.random.default_rng(2)
    image[::3] = rng.uniform(0, 1024, size=(len(image[::3]), 2))
    first, second = solve_pose(image, world, K), solve_pose(image, world, K)
    np.testing.assert_array_equal(first.pose.position, second.pose.position)
    np.testing.assert_array_equal(first.inliers, second.inliers)


@pytest.mark.parametrize("n", [0, 1, MIN_CORRESPONDENCES - 1])
def test_too_few_correspondences_give_a_failure_record_not_an_exception(n):
    image, world = terrain_points(200)
    solution = solve_pose(image[:n], world[:n], K)
    assert solution.status == "too_few_matches"
    assert solution.pose is None
    assert solution.inliers.shape == (n,)
    assert not solution.inliers.any()


def test_correspondences_with_no_consistent_pose_give_a_failure_record():
    # Every match wrong: whatever RANSAC fits explains too few points to count as a fix.
    _, world = terrain_points(200)
    image = np.random.default_rng(3).uniform(0, 1024, size=(len(world), 2))
    solution = solve_pose(image, world, K)
    assert solution.status == "no_solution"
    assert solution.pose is None
    assert solution.inliers.shape == (len(world),)


def test_mismatched_point_counts_are_a_caller_error():
    image, world = terrain_points(50)
    with pytest.raises(ValueError, match="same number"):
        solve_pose(image[:-1], world, K)


# --- a miscalibrated focal length (#38): rendered with K, solved with K's focal length off by e

ALTITUDE = 2300.0


def flat_ground_view(tilt_deg, azimuth_deg):
    """A camera ALTITUDE above flat ground (z = 0), leaning tilt_deg off nadir towards the
    compass bearing azimuth_deg, with a grid over its whole image and the ground points it sees."""
    t, a = math.radians(tilt_deg), math.radians(azimuth_deg)
    eye = np.array([0.0, 0.0, ALTITUDE])
    axis = np.array([math.sin(t) * math.sin(a), math.sin(t) * math.cos(a), -math.cos(t)])
    pose = look_at(eye, eye + axis, up=[0.3, 1.0, 0.0])
    u, v = np.meshgrid(np.linspace(0, 1023, 40), np.linspace(0, 1023, 40))
    pixels = np.column_stack([u.ravel(), v.ravel()])
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(K.matrix()).T @ pose.rotation.T
    world = eye + (-ALTITUDE / rays[:, 2])[:, None] * rays
    return pose, pixels, world


def test_at_nadir_a_focal_error_puts_the_camera_that_fraction_higher_and_nowhere_else():
    # Over flat ground, seen square on, a longer focal length looks exactly like a camera
    # farther away: the solve is off by e * h straight up, exactly, not just to first order.
    pose, pixels, world = flat_ground_view(0.0, 0.0)
    shift = solve_pose(pixels, world, K.with_focal_error(0.01)).pose.position - pose.position

    np.testing.assert_allclose(shift, [0.0, 0.0, 0.01 * ALTITUDE], atol=1e-3)


@pytest.mark.parametrize("focal_error", [0.01, -0.01])
@pytest.mark.parametrize("azimuth_deg", [0.0, 135.0])
@pytest.mark.parametrize("tilt_deg", [6.0, 12.0, 18.0])
def test_off_nadir_a_focal_error_moves_the_fix_e_h_sin_2tilt_away_from_the_tilt(tilt_deg, azimuth_deg, focal_error):
    # First order in e, in the vertical plane of the tilt: fitting ground points x = h tan(tilt + phi)
    # with every ray's angle shrunk by tan(phi') = tan(phi) / (1 + e) is solved exactly by moving the
    # camera e h sin(2 tilt) back from the tilt (and e h cos(2 tilt) up). The rest of the image, out
    # of that plane, bends it by under 2% at 18 deg. This is the off-nadir term JPL found: +-1% gave
    # up to 16.9 m horizontally (Johnson et al. 2017, docs/research/lvs-error-sources.md).
    pose, pixels, world = flat_ground_view(tilt_deg, azimuth_deg)
    shift = solve_pose(pixels, world, K.with_focal_error(focal_error)).pose.position - pose.position

    a = math.radians(azimuth_deg)
    towards_tilt = np.array([math.sin(a), math.cos(a)])
    along = shift[:2] @ towards_tilt
    expected = -focal_error * ALTITUDE * math.sin(math.radians(2 * tilt_deg))
    assert along == pytest.approx(expected, rel=0.03)
    across = shift[:2] - along * towards_tilt
    assert np.linalg.norm(across) < 0.02 * abs(expected)
