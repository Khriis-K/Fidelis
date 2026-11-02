import numpy as np
import pytest

from fidelis.render.camera import CameraPose, Intrinsics, look_at


def test_a_nadir_camera_with_north_up_sees_a_north_up_map():
    # Image right is east and image down is south, like a north-up map.
    pose = look_at(eye=[10.0, 20.0, 500.0], target=[10.0, 20.0, 0.0], up=[0.0, 1.0, 0.0])

    np.testing.assert_allclose(pose.rotation[:, 0], [1, 0, 0], atol=1e-12)  # camera x (right)
    np.testing.assert_allclose(pose.rotation[:, 1], [0, -1, 0], atol=1e-12)  # camera y (down)
    np.testing.assert_allclose(pose.rotation[:, 2], [0, 0, -1], atol=1e-12)  # camera z (forward)
    np.testing.assert_allclose(pose.position, [10, 20, 500])


def test_a_level_camera_looking_north_has_east_on_its_right_and_the_sky_up():
    pose = look_at(eye=[0.0, 0.0, 0.0], target=[0.0, 50.0, 0.0], up=[0.0, 0.0, 1.0])

    np.testing.assert_allclose(pose.rotation, [[1, 0, 0], [0, 0, 1], [0, -1, 0]], atol=1e-12)


def test_look_at_gives_a_proper_rotation_for_an_oblique_view():
    pose = look_at(eye=[30.0, -40.0, 120.0], target=[5.0, 10.0, -3.0], up=[0.0, 1.0, 0.0])

    np.testing.assert_allclose(pose.rotation.T @ pose.rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(pose.rotation) == pytest.approx(1.0)
    forward = np.array([5.0, 10.0, -3.0]) - [30.0, -40.0, 120.0]
    np.testing.assert_allclose(pose.rotation[:, 2], forward / np.linalg.norm(forward), atol=1e-12)


def test_look_at_refuses_an_up_vector_along_the_view_direction():
    # Straight down with up = +z leaves the image's roll undefined.
    with pytest.raises(ValueError, match="parallel"):
        look_at(eye=[0.0, 0.0, 100.0], target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0])


def test_world_to_camera_maps_the_camera_centre_to_the_origin_and_the_target_onto_the_optical_axis():
    pose = look_at(eye=[30.0, -40.0, 120.0], target=[5.0, 10.0, -3.0], up=[0.0, 1.0, 0.0])
    w2c = pose.world_to_camera()

    np.testing.assert_allclose(w2c @ [30.0, -40.0, 120.0, 1.0], [0, 0, 0, 1], atol=1e-9)
    target_cam = w2c @ [5.0, 10.0, -3.0, 1.0]
    np.testing.assert_allclose(target_cam[:2], [0, 0], atol=1e-9)
    assert target_cam[2] > 0  # in front of the camera


def test_a_pose_rejects_a_matrix_that_is_not_a_rotation():
    with pytest.raises(ValueError, match="rotation"):
        CameraPose(rotation=np.diag([1.0, 1.0, -1.0]), position=np.zeros(3))


def test_a_pose_accepts_a_rotation_stored_as_float32():
    # gsplat keeps viewmats in float32, and poses saved to JSON may pass through it;
    # float32 rounding (about 6e-8) must not make a valid rotation unusable.
    exact = look_at(eye=[30.0, -40.0, 120.0], target=[5.0, 10.0, -3.0], up=[0.0, 1.0, 0.0])
    pose = CameraPose(rotation=exact.rotation.astype(np.float32), position=exact.position.astype(np.float32))

    np.testing.assert_allclose(pose.rotation, exact.rotation, atol=1e-6)


def test_a_pose_rejects_a_matrix_that_is_slightly_scaled():
    # 0.1% scale is far beyond float32 rounding, so it's a bug upstream, not noise.
    rotation = look_at(eye=[30.0, -40.0, 120.0], target=[5.0, 10.0, -3.0], up=[0.0, 1.0, 0.0]).rotation
    with pytest.raises(ValueError, match="rotation"):
        CameraPose(rotation=1.001 * rotation, position=np.zeros(3))


def test_look_at_refuses_a_target_at_the_eye():
    with pytest.raises(ValueError, match="target"):
        look_at(eye=[1.0, 2.0, 3.0], target=[1.0, 2.0, 3.0], up=[0.0, 1.0, 0.0])


def test_intrinsics_from_field_of_view_centre_the_principal_point_on_the_image():
    # OpenCV pixel convention: pixel (0, 0)'s centre is at (0, 0), so the middle of a
    # 1024-pixel row is 511.5, not 512.
    k = Intrinsics.from_fov(width=1024, height=1024, fov_x_deg=90.0)

    assert k.fx == pytest.approx(512.0)
    assert k.fy == pytest.approx(512.0)
    assert (k.cx, k.cy) == pytest.approx((511.5, 511.5))
    np.testing.assert_allclose(k.matrix(), [[512, 0, 511.5], [0, 512, 511.5], [0, 0, 1]])
