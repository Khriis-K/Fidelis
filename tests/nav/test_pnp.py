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
