"""Which elevation the LVS map's keypoints get (#36): the CTX DTM's, as LVS's onboard map did."""

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fidelis.nav.orbital_map import load_lvs_map, load_terrain
from fidelis.terrain.scene import curvature_drop

MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"
MARS_RADIUS_M = 3396190.0
LEFT, TOP = 4589000.75, 1094003.14
MAP_RES, CTX_DTM_RES, HIRISE_DTM_RES = 6.0, 20.0, 4.0
ORIGIN = (LEFT + 300.0, TOP - 300.0)
EAST_SCALE = 0.95


def ctx_plane(mx, my):
    return -2500.0 + 0.01 * (mx - LEFT) - 0.02 * (TOP - my)


def hirise_plane(mx, my):
    """The scene's terrain: 5 m above CTX's everywhere, so a lookup shows which DTM answered."""
    return ctx_plane(mx, my) + 5.0


def write_raster(path, data, res, left=LEFT, top=TOP):
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": data.dtype,
        "crs": MARS_EQC,
        "transform": from_origin(left, top, res, res),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def plane_raster(plane, res, size_m=600.0, left=LEFT, top=TOP):
    n = round(size_m / res)
    cols, rows = np.meshgrid(np.arange(n), np.arange(n))
    return plane(left + (cols + 0.5) * res, top - (rows + 0.5) * res).astype(np.float32)


@pytest.fixture
def data(tmp_path):
    """The workflow's layout: the CTX map, the CTX DTM over the whole scene, the scene's own DTM."""
    (tmp_path / "scene").mkdir()
    (tmp_path / "mesh").mkdir()
    write_raster(tmp_path / "map.tif", np.full((100, 100), 128, np.uint8), MAP_RES)
    write_raster(tmp_path / "scene" / "ring_dtm.tif", plane_raster(ctx_plane, CTX_DTM_RES), CTX_DTM_RES)
    # The scene's (HiRISE) DTM covers only the middle 200 m.
    inner = plane_raster(hirise_plane, HIRISE_DTM_RES, 200.0, LEFT + 200.0, TOP - 200.0)
    write_raster(tmp_path / "scene" / "dtm.tif", inner, HIRISE_DTM_RES, LEFT + 200.0, TOP - 200.0)
    frame = {
        "origin": list(ORIGIN),
        "east_scale": EAST_SCALE,
        "curvature_radius_m": MARS_RADIUS_M,
        "ring": {"dtm": "../scene/ring_dtm.tif"},
    }
    (tmp_path / "mesh" / "mesh.json").write_text(json.dumps(frame))
    return tmp_path


def lvs_map(data):
    return load_lvs_map(data / "map.tif", data / "scene" / "ring_dtm.tif", data / "mesh" / "mesh.json")


def test_lvs_map_keypoints_get_ctx_elevation_even_where_the_scene_has_hirise(data):
    # Pixel (50, 50) is in the middle of the map, where the scene's own DTM is HiRISE (+5 m).
    pixels = np.array([[50.0, 50.0], [5.0, 90.0]])
    world = lvs_map(data).pixel_to_world(pixels)

    mx, my = LEFT + (pixels[:, 0] + 0.5) * MAP_RES, TOP - (pixels[:, 1] + 0.5) * MAP_RES
    x, y = (mx - ORIGIN[0]) * EAST_SCALE, my - ORIGIN[1]
    expected = ctx_plane(mx, my) - curvature_drop(x, y, MARS_RADIUS_M)
    np.testing.assert_allclose(world[:, 2], expected, atol=1e-3)
    np.testing.assert_allclose(world[:, 0], x, atol=1e-6)
    np.testing.assert_allclose(world[:, 1], y, atol=1e-6)


def test_the_lvs_map_is_in_the_meshs_world_frame_and_scaled_to_0_1(data):
    loaded = lvs_map(data)

    assert loaded.origin == ORIGIN and loaded.east_scale == EAST_SCALE
    assert loaded.curvature_radius_m == MARS_RADIUS_M
    assert loaded.ring_dtm is None  # one elevation source, CTX's, with nothing layered over it
    np.testing.assert_allclose(loaded.image, 128 / 255, atol=1e-7)


def test_the_scenes_own_terrain_still_answers_with_hirise_where_it_has_it(data):
    # The truth: where the rendered ground is. HiRISE in the middle, CTX's ring beyond.
    terrain = load_terrain(data / "scene" / "dtm.tif", data / "mesh" / "mesh.json")

    assert terrain.ground_height(0.0, 0.0) == pytest.approx(hirise_plane(*ORIGIN), abs=1e-3)
    east, north = 250.0 * EAST_SCALE, 250.0
    outside = ctx_plane(ORIGIN[0] + 250.0, ORIGIN[1] + 250.0) - curvature_drop(east, north, MARS_RADIUS_M)
    assert terrain.ground_height(east, north) == pytest.approx(outside, abs=1e-3)


def test_the_map_and_the_scene_disagree_on_height_where_the_two_dtms_do(data):
    # The map's elevation error is real and stays in: here the fixture's 5 m.
    terrain = load_terrain(data / "scene" / "dtm.tif", data / "mesh" / "mesh.json")

    assert terrain.ground_height(0.0, 0.0) - lvs_map(data).ground_height(0.0, 0.0) == pytest.approx(5.0, abs=1e-3)
