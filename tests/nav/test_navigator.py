import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from rasterio.transform import from_origin

from fidelis.nav.navigator import Navigator
from fidelis.nav.orbital_map import OrbitalMap, load_orbital_map
from fidelis.render.camera import Intrinsics, look_at

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


def nadir_view(orbital_map):
    """What a north-up nadir pinhole camera at TRUE_POSITION sees of the flat map, built analytically."""
    u, v = np.meshgrid(np.arange(K.width, dtype=np.float64), np.arange(K.height, dtype=np.float64))
    east = TRUE_POSITION[0] + (u - K.cx) * ALTITUDE / K.fx
    north = TRUE_POSITION[1] - (v - K.cy) * ALTITUDE / K.fy
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
    # Both ends, because one isn't enough: at 2.3 km the fix already carries about 1.8 m
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
