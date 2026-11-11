import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from rasterio.transform import from_origin

from fidelis.nav.navigator import Navigator, nadir_homography
from fidelis.nav.orbital_map import OrbitalMap, load_orbital_map
from fidelis.render.camera import CameraPose, Intrinsics, look_at

REPO = Path(__file__).resolve().parents[2]
LEFT, TOP = 4589000.0, 1094000.0
MAP_RES = 2.0
MAP_SIZE = 512
EAST_SCALE = 0.95
ORIGIN = (LEFT + MAP_RES * MAP_SIZE / 2, TOP - MAP_RES * MAP_SIZE / 2)  # world (0, 0) at the map's centre
GROUND_Z = -2500.0


def texture(size, seed=0):
    """Multi-scale smoothed noise: corners at many scales for the detector to find."""
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), np.float32)
    for sigma in (1, 2, 4, 8):
        image += cv2.GaussianBlur(rng.random((size, size), np.float32), (0, 0), sigma)
    return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)


@pytest.fixture(scope="module")
def flat_map():
    """A textured map of flat ground, 1 km across at 2 m/pixel (1.9 m east-west on the ground)."""
    return OrbitalMap(
        image=texture(MAP_SIZE),
        transform=from_origin(LEFT, TOP, MAP_RES, MAP_RES),
        dtm=np.full((8, 8), GROUND_Z, np.float32),
        dtm_transform=from_origin(LEFT, TOP, MAP_RES * MAP_SIZE / 8, MAP_RES * MAP_SIZE / 8),
        origin=ORIGIN,
        east_scale=EAST_SCALE,
    )


@pytest.fixture(scope="module")
def navigator(flat_map):
    return Navigator(flat_map, device="cpu")


K = Intrinsics.from_fov(256, 256, 90)
ALTITUDE = 192.0  # 1.5 m descent pixels: finer than the map, as in flight
TRUE_POSITION = np.array([40.0, -25.0, GROUND_Z + ALTITUDE])


def attitude(heading_deg):
    """A nadir camera's camera-to-world rotation, its image top towards `heading_deg` (0 north, 90 east)."""
    h = np.radians(heading_deg)
    return look_at(TRUE_POSITION, TRUE_POSITION - [0, 0, 1], [np.sin(h), np.cos(h), 0]).rotation


def nadir_view(orbital_map, heading_deg=0.0):
    """What a nadir pinhole camera at TRUE_POSITION, its top towards `heading_deg`, sees of the flat map."""
    u, v = np.meshgrid(np.arange(K.width, dtype=np.float64), np.arange(K.height, dtype=np.float64))
    right = (u - K.cx) * ALTITUDE / K.fx  # ground metres along the image's right and down directions
    down = (v - K.cy) * ALTITUDE / K.fy
    h = np.radians(heading_deg)
    east = TRUE_POSITION[0] + right * np.cos(h) - down * np.sin(h)
    north = TRUE_POSITION[1] - right * np.sin(h) - down * np.cos(h)
    col = ((east / EAST_SCALE + ORIGIN[0]) - LEFT) / MAP_RES - 0.5
    row = (TOP - (north + ORIGIN[1])) / MAP_RES - 0.5
    return cv2.remap(orbital_map.image, col.astype(np.float32), row.astype(np.float32), cv2.INTER_CUBIC)


def test_a_nadir_view_of_the_map_locates_the_camera_within_a_quarter_map_pixel(navigator, flat_map):
    # A half-pixel slip anywhere (keypoints, map georeferencing, intrinsics) moves the fix
    # by about 1 m here; a quarter of the 1.9 m ground pixel catches it with margin.
    result = navigator.locate(np.clip(nadir_view(flat_map), 0, 1), K)
    assert result.status == "ok"
    assert np.linalg.norm(result.position - TRUE_POSITION) < 0.25 * MAP_RES * EAST_SCALE


def test_the_result_reports_matches_inliers_and_the_matched_pixel_pairs(navigator, flat_map):
    result = navigator.locate(np.clip(nadir_view(flat_map), 0, 1), K)
    n = result.match_count
    assert n > 50
    assert result.image_points.shape == (n, 2)
    assert result.map_points.shape == (n, 2)
    assert result.inliers.shape == (n,) and result.inliers.dtype == bool
    assert result.inlier_count == result.inliers.sum()
    assert 0 < result.inlier_count <= n
    # Every pair replays: the inlier map pixel, put through the fix, projects onto its image pixel.
    w2c = result.pose.world_to_camera()
    world = flat_map.pixel_to_world(result.map_points[result.inliers])
    projected, _ = cv2.projectPoints(world, cv2.Rodrigues(w2c[:3, :3])[0], w2c[:3, 3], K.matrix(), None)
    assert np.median(np.linalg.norm(projected.reshape(-1, 2) - result.image_points[result.inliers], axis=1)) < 1.0


def test_the_nadir_homography_maps_a_tilted_turned_cameras_pixels_onto_a_nadir_north_up_cameras():
    # Same centre, same intrinsics; only the rotation differs, so a homography is exact.
    eye = np.array([10.0, 20.0, 300.0])
    tilted = look_at(eye, eye + [40, -25, -300], [np.sin(2.0), np.cos(2.0), 0])  # about 9 deg off nadir, turned 115 deg
    nadir = look_at(eye, eye - [0, 0, 1], [0, 1, 0])
    ground = np.array([[x, y, 0.0] for x in (-60, 0, 90) for y in (-80, 30, 70)])

    def project(pose):
        w2c = pose.world_to_camera()
        pixels, _ = cv2.projectPoints(ground, cv2.Rodrigues(w2c[:3, :3])[0], w2c[:3, 3], K.matrix(), None)
        return pixels.reshape(-1, 2)

    h = nadir_homography(tilted.rotation, K)
    mapped = cv2.perspectiveTransform(project(tilted).reshape(-1, 1, 2), h).reshape(-1, 2)
    np.testing.assert_allclose(mapped, project(nadir), atol=1e-6)


def test_an_attitude_whose_view_reaches_the_horizon_is_a_caller_error(navigator, flat_map):
    # A nadir view of that camera's corners would lie behind it: there's no image to warp to.
    eye = TRUE_POSITION
    sideways = look_at(eye, eye + [1, 0, -0.3], [0, 0, 1]).rotation  # about 17 deg below the horizon
    with pytest.raises(ValueError, match="horizon"):
        navigator.locate(np.clip(nadir_view(flat_map), 0, 1), K, sideways)


@pytest.mark.parametrize("heading", [90.0, 180.0])
def test_a_turned_view_located_with_its_attitude_is_as_accurate_as_north_up(navigator, flat_map, heading):
    result = navigator.locate(np.clip(nadir_view(flat_map, heading), 0, 1), K, attitude(heading))
    assert result.status == "ok"
    assert np.linalg.norm(result.position - TRUE_POSITION) < 0.25 * MAP_RES * EAST_SCALE


@pytest.mark.parametrize("heading", [15.0, 45.0])
def test_an_off_axis_turn_located_with_its_attitude_lands_within_half_a_map_pixel(navigator, flat_map, heading):
    # Turns off the pixel grid resample the image, which costs accuracy: here 0.795 m at 15
    # deg and 0.496 m at 45, against 0.174 m north-up (scripts/check_attitude_warp.py,
    # docs/benchmarks/navigator.md). Half a map pixel, 0.95 m and the GPU test's tolerance,
    # still catches a warp that's wrong, not just blurred.
    result = navigator.locate(np.clip(nadir_view(flat_map, heading), 0, 1), K, attitude(heading))
    assert result.status == "ok"
    assert np.linalg.norm(result.position - TRUE_POSITION) < 0.5 * MAP_RES * EAST_SCALE


@pytest.mark.parametrize("prior", [None, 0.0])
def test_a_turned_view_without_its_true_attitude_does_not_locate(navigator, flat_map, prior):
    # The warp is what makes turned views work: no prior, or a north-up one, leaves it lost.
    view = np.clip(nadir_view(flat_map, 180.0), 0, 1)
    result = navigator.locate(view, K, None if prior is None else attitude(prior))
    assert result.status != "ok" or np.linalg.norm(result.position - TRUE_POSITION) > 0.25 * MAP_RES * EAST_SCALE


def test_a_turned_views_matched_pixels_are_in_the_cameras_own_frame(navigator, flat_map):
    # Checked against the true camera, not the fix: pixels mapped back the wrong way round
    # (turned 180 deg) would still agree with a fix that turned the camera 180 deg to suit them.
    result = navigator.locate(np.clip(nadir_view(flat_map, 90.0), 0, 1), K, attitude(90.0))
    w2c = CameraPose(attitude(90.0), TRUE_POSITION).world_to_camera()
    world = flat_map.pixel_to_world(result.map_points[result.inliers])
    projected, _ = cv2.projectPoints(world, cv2.Rodrigues(w2c[:3, :3])[0], w2c[:3, 3], K.matrix(), None)
    assert np.median(np.linalg.norm(projected.reshape(-1, 2) - result.image_points[result.inliers], axis=1)) < 1.0
    # And the fix itself faces the way the camera does.
    assert np.allclose(result.pose.rotation, attitude(90.0), atol=0.01)


def test_the_result_keeps_every_keypoint_matched_or_not_in_the_cameras_own_frame(navigator, flat_map):
    # The match picture draws unmatched keypoints too; on a turned view they must come back
    # through the warp like the matched ones, so each matched pixel is one of them.
    result = navigator.locate(np.clip(nadir_view(flat_map, 90.0), 0, 1), K, attitude(90.0))
    assert len(result.keypoints) > result.match_count
    distance = np.linalg.norm(result.image_points[:, None] - result.keypoints[None], axis=2).min(axis=1)
    assert distance.max() < 1e-9
    inside = (result.keypoints >= -0.5) & (result.keypoints <= [K.width - 0.5, K.height - 0.5])
    assert inside.all()


def test_a_featureless_image_gives_a_failure_record_not_an_exception(navigator):
    result = navigator.locate(np.zeros((K.height, K.width), np.float32), K)
    assert result.status != "ok"
    assert result.pose is None and result.position is None
    assert result.image_points.shape == (result.match_count, 2)
    assert result.inliers.shape == (result.match_count,)


def test_an_image_of_unrelated_terrain_gives_a_failure_record_not_an_exception(navigator):
    result = navigator.locate(texture(K.width, seed=99), K)
    assert result.status != "ok"
    assert result.position is None


def test_an_image_that_does_not_match_the_intrinsics_is_a_caller_error(navigator):
    with pytest.raises(ValueError, match="intrinsics"):
        navigator.locate(np.zeros((100, 100), np.float32), K)


def test_no_superpoint_anywhere_in_the_package():
    # SuperPoint's weights are non-commercial; ALIKED (BSD-3) and LightGlue (Apache-2.0) are not.
    sources = (REPO / "src" / "fidelis").rglob("*.py")
    assert not [p for p in sources if "superpoint" in p.read_text(encoding="utf-8").lower()]


@pytest.mark.gpu
@pytest.mark.parametrize("altitude", [2300.0, 4100.0])
def test_a_perfect_conditions_render_of_jezero_locates_within_half_a_ctx_map_pixel(altitude):
    # Perfect conditions (spec): nadir, north up, no noise or blur, and a zenith sun, so the
    # render's shading adds nothing the map's orthoimage doesn't already carry. Both
    # altitudes, above the mean ground, sit inside LVS's 2.2-4.2 km band, near its ends.
    # The map is the 6 m CTX-like one. Half its ground pixel is the tolerance: thousands of
    # inliers average well below one map pixel, so missing it means something systematic.
    # Both ends, because one isn't enough: near 2.2 km the fix already carries 1.88 m
    # of matching bias, pointing so that a half-pixel slip in the map's georeferencing
    # (about 4 m) still lands inside the tolerance. At 4.1 km the bias is small and the
    # slip isn't (docs/benchmarks/navigator.md).
    import drjit as dr
    import mitsuba as mi

    from fidelis.render.mesh_render import Sun, render, require_no_blank_lines, terrain_scene

    data = REPO / "data" / "jezero"
    assert (data / "mesh" / "mesh.json").exists(), (
        "build the Jezero scene first: snakemake --profile profiles/local jezero"
    )
    orbital_map = load_orbital_map(
        data / "matching_map" / "map.tif", data / "tile" / "dtm.tif", data / "mesh" / "mesh.json"
    )
    ground_px = json.loads((data / "matching_map" / "map.json").read_text())["ground_resolution_m"]
    tolerance = 0.5 * min(ground_px.values())

    mi.set_variant("cuda_ad_rgb")
    truth = np.array([150.0, -230.0, float(orbital_map.dtm.mean()) + altitude])
    k = Intrinsics.from_fov(1024, 1024, 90)
    scene = terrain_scene(data / "mesh", Sun(azimuth_deg=0.0, elevation_deg=90.0))
    image = render(scene, look_at(truth, truth - [0, 0, 1], [0, 1, 0]), k, spp=1024)
    require_no_blank_lines(image)
    del scene
    dr.flush_malloc_cache()

    # A zenith sun of irradiance pi renders flat ground at its albedo, already in [0, 1].
    result = Navigator(orbital_map).locate(np.clip(image, 0, 1), k)

    assert result.status == "ok"
    assert np.linalg.norm(result.position - truth) < tolerance


# --- map thinning: the knob behind the cheaper fidelity levels (docs/adr/0002-navigator-fidelity-ladder.md)


def test_thinning_keeps_the_k_highest_scoring_map_keypoints(navigator):
    full = navigator.map_features
    thin = navigator.thinned(500).map_features
    scores = full["keypoint_scores"][0].numpy()
    kept = thin["keypoint_scores"][0].numpy()
    assert len(kept) == 500
    np.testing.assert_array_equal(np.sort(kept), np.sort(scores)[-500:])
    # Each kept keypoint keeps its own position and descriptor, not another's.
    rows = {tuple(p): i for i, p in enumerate(full["keypoints"][0].numpy())}
    for point, descriptor in zip(thin["keypoints"][0].numpy(), thin["descriptors"][0].numpy(), strict=True):
        np.testing.assert_array_equal(descriptor, full["descriptors"][0].numpy()[rows[tuple(point)]])
    np.testing.assert_array_equal(thin["image_size"].numpy(), full["image_size"].numpy())


def test_thinning_leaves_the_real_navigators_map_alone(navigator):
    before = len(navigator.map_keypoints)
    navigator.thinned(100)
    assert len(navigator.map_keypoints) == before


def test_thinning_to_every_keypoint_locates_exactly_as_the_real_navigator(navigator, flat_map):
    # K = all must be the real navigator, not an approximation of it: the top fidelity level
    # is the system under test.
    count = len(navigator.map_keypoints)
    for k in (count, 10 * count):
        # The very same map features, in the same order: one weak keypoint dropped or
        # reordered rarely moves an easy fix, so comparing fixes alone wouldn't notice.
        thin = navigator.thinned(k).map_features
        for key in ("keypoints", "descriptors", "keypoint_scores"):
            np.testing.assert_array_equal(thin[key].numpy(), navigator.map_features[key].numpy())
    image = np.clip(nadir_view(flat_map, heading_deg=30.0), 0, 1)
    real = navigator.locate(image, K, attitude(30.0))
    every = navigator.thinned(len(navigator.map_keypoints)).locate(image, K, attitude(30.0))
    more = navigator.thinned(10 * len(navigator.map_keypoints)).locate(image, K, attitude(30.0))
    for result in (every, more):
        assert result.status == real.status
        np.testing.assert_array_equal(result.position, real.position)
        np.testing.assert_array_equal(result.map_points, real.map_points)


def test_a_thinned_navigator_still_locates_on_a_well_textured_map(navigator, flat_map):
    # Half the flat map's 2,884 keypoints: cheaper, and on easy ground still within a map pixel.
    result = navigator.thinned(1400).locate(np.clip(nadir_view(flat_map), 0, 1), K)
    assert result.status == "ok"
    assert np.linalg.norm(result.position - TRUE_POSITION) < MAP_RES * EAST_SCALE
    kept = {tuple(p) for p in navigator.thinned(1400).map_keypoints}
    assert all(tuple(p) in kept for p in result.map_points)  # it matched only the map it kept


@pytest.mark.parametrize("k", [0, -5])
def test_thinning_to_no_keypoints_is_a_caller_error(navigator, k):
    with pytest.raises(ValueError, match="at least one"):
        navigator.thinned(k)
