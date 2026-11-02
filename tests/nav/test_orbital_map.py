import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fidelis.nav.orbital_map import OrbitalMap, load_orbital_map

MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"
LEFT, TOP = 4589000.0, 1094000.0
ORIGIN = (4589100.0, 1093900.0)  # world (0, 0), in map metres
EAST_SCALE = 0.95


def plane(mx, my):
    """A tilted plane of elevation over map metres: bilinear sampling reproduces it exactly."""
    return -2500.0 + 0.03 * (mx - LEFT) - 0.05 * (TOP - my)


def plane_dtm(res=8.0, size=25):
    """The plane sampled at the DTM's pixel centres, on a coarser grid than the map's."""
    cols, rows = np.meshgrid(np.arange(size), np.arange(size))
    heights = plane(LEFT + (cols + 0.5) * res, TOP - (rows + 0.5) * res).astype(np.float32)
    return heights, from_origin(LEFT, TOP, res, res)


def make_map(image=None, map_res=2.0, map_size=100):
    dtm, dtm_transform = plane_dtm()
    if image is None:
        image = np.zeros((map_size, map_size), np.float32)
    return OrbitalMap(
        image=image,
        transform=from_origin(LEFT, TOP, map_res, map_res),
        dtm=dtm,
        dtm_transform=dtm_transform,
        origin=ORIGIN,
        east_scale=EAST_SCALE,
    )


def test_a_map_pixel_centre_lands_at_its_ground_position_in_the_world_frame():
    # Pixel (col 0, row 0)'s centre is half a 2 m pixel in from the map's top-left corner,
    # and the world frame is ground metres from ORIGIN: east shrinks by east_scale.
    world = make_map().pixel_to_world(np.array([[0.0, 0.0], [10.0, 20.0]]))
    mx = np.array([LEFT + 1.0, LEFT + 21.0])
    my = np.array([TOP - 1.0, TOP - 41.0])
    np.testing.assert_allclose(world[:, 0], (mx - ORIGIN[0]) * EAST_SCALE, atol=1e-9)
    np.testing.assert_allclose(world[:, 1], my - ORIGIN[1], atol=1e-9)


def test_elevation_comes_from_the_dtm_beneath_the_map_pixel_not_the_dtm_pixel_with_the_same_index():
    # The DTM is on an 8 m grid, the map on 2 m: elevation must be looked up by ground
    # position. A plane is reproduced exactly by bilinear interpolation, so any
    # half-pixel slip in either grid shows up as a metre-scale error here.
    pixels = np.array([[0.0, 0.0], [37.3, 61.8], [99.0, 99.0], [12.5, 80.25]])
    world = make_map().pixel_to_world(pixels)
    mx = LEFT + (pixels[:, 0] + 0.5) * 2.0
    my = TOP - (pixels[:, 1] + 0.5) * 2.0
    np.testing.assert_allclose(world[:, 2], plane(mx, my), atol=1e-3)


def test_ground_height_is_the_dtm_beneath_a_world_position():
    # World east is ground metres: back to map metres it grows by 1 / east_scale. A slip
    # in that (or in the origin) moves the lookup across the tilted plane by metres.
    for east, north in [(0.0, 0.0), (-60.0, 35.0), (40.0, -25.0)]:
        mx, my = east / EAST_SCALE + ORIGIN[0], north + ORIGIN[1]
        assert make_map().ground_height(east, north) == pytest.approx(plane(mx, my), abs=1e-3)


def test_an_empty_set_of_pixels_gives_an_empty_set_of_points():
    assert make_map().pixel_to_world(np.zeros((0, 2))).shape == (0, 3)


def test_a_map_reaching_past_the_dtm_is_rejected():
    # 8 m x 25 = 200 m of DTM; a 120 px map at 2 m reaches 240 m, where elevation is unknown.
    with pytest.raises(ValueError, match="DTM"):
        make_map(map_size=120)


def test_a_map_image_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        make_map(image=np.full((100, 100), 255.0, np.float32))


def write_raster(path, data, res):
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": data.dtype,
        "crs": MARS_EQC,
        "transform": from_origin(LEFT, TOP, res, res),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_the_map_loads_from_the_workflow_files_in_the_mesh_world_frame(tmp_path):
    image = np.arange(100 * 100, dtype=np.uint64).reshape(100, 100) % 256
    write_raster(tmp_path / "map.tif", image.astype(np.uint8), 2.0)
    dtm, _ = plane_dtm()
    write_raster(tmp_path / "dtm.tif", dtm, 8.0)
    (tmp_path / "mesh.json").write_text(json.dumps({"origin": list(ORIGIN), "east_scale": EAST_SCALE}))

    loaded = load_orbital_map(tmp_path / "map.tif", tmp_path / "dtm.tif", tmp_path / "mesh.json")

    np.testing.assert_allclose(loaded.image, image / 255.0, atol=1e-7)
    np.testing.assert_allclose(
        loaded.pixel_to_world(np.array([[10.0, 20.0]])),
        make_map().pixel_to_world(np.array([[10.0, 20.0]])),
        atol=1e-6,
    )


# --- the wide scene: a ring DTM around the inner one, and Mars's curvature (#33) ----------

MARS_RADIUS_M = 3396190.0
RING_RES = 40.0
RING_LEFT, RING_TOP = LEFT - 400.0, TOP + 400.0


def ring_plane(mx, my):
    """The ring's terrain: the inner plane, 3 m higher, so a lookup shows which DTM answered."""
    return plane(mx, my) + 3.0


def ring_dtm():
    cols, rows = np.meshgrid(np.arange(25), np.arange(25))
    heights = ring_plane(RING_LEFT + (cols + 0.5) * RING_RES, RING_TOP - (rows + 0.5) * RING_RES)
    return heights.astype(np.float32), from_origin(RING_LEFT, RING_TOP, RING_RES, RING_RES)


def make_wide_map(curvature_radius_m=None):
    dtm, dtm_transform = plane_dtm()
    ring, ring_transform = ring_dtm()
    # 600 m of map around the 200 m inner DTM, all inside the 1000 m ring.
    return OrbitalMap(
        image=np.zeros((300, 300), np.float32),
        transform=from_origin(LEFT - 200.0, TOP + 200.0, 2.0, 2.0),
        dtm=dtm,
        dtm_transform=dtm_transform,
        origin=ORIGIN,
        east_scale=EAST_SCALE,
        ring_dtm=ring,
        ring_dtm_transform=ring_transform,
        curvature_radius_m=curvature_radius_m,
    )


def test_a_map_wider_than_the_inner_dtm_is_accepted_when_the_ring_covers_it():
    make_wide_map()


def test_elevation_comes_from_the_inner_dtm_inside_it_and_from_the_ring_outside():
    wide = make_wide_map()
    inside = (LEFT + 100.0, TOP - 100.0)
    outside = (LEFT + 350.0, TOP - 100.0)
    for (mx, my), expected in ((inside, plane), (outside, ring_plane)):
        east, north = (mx - ORIGIN[0]) * EAST_SCALE, my - ORIGIN[1]
        assert wide.ground_height(east, north) == pytest.approx(expected(mx, my), abs=1e-3)


def test_map_pixels_over_the_ring_get_ring_elevations():
    wide = make_wide_map()
    # Map column 10 is 179 m west of the inner DTM's edge.
    world = wide.pixel_to_world(np.array([[10.0, 150.0]]))
    mx, my = LEFT - 200.0 + 21.0, TOP + 200.0 - 301.0
    assert world[0, 2] == pytest.approx(ring_plane(mx, my), abs=1e-3)


def test_curvature_lowers_the_ground_by_d_squared_over_2r_from_the_world_origin():
    flat, curved = make_wide_map(), make_wide_map(curvature_radius_m=MARS_RADIUS_M)
    for east, north in [(0.0, 0.0), (250.0, -120.0), (-180.0, 300.0)]:
        drop = (east**2 + north**2) / (2 * MARS_RADIUS_M)
        assert curved.ground_height(east, north) == pytest.approx(flat.ground_height(east, north) - drop, abs=1e-9)
    pixels = np.array([[3.0, 290.0], [250.0, 20.0]])
    f, c = flat.pixel_to_world(pixels), curved.pixel_to_world(pixels)
    np.testing.assert_allclose(c[:, 2], f[:, 2] - (f[:, 0] ** 2 + f[:, 1] ** 2) / (2 * MARS_RADIUS_M), atol=1e-9)
    assert (f[:, 2] - c[:, 2]).max() > 0.01  # far enough out for the drop to show


def test_the_map_loads_the_ring_and_curvature_named_in_mesh_json(tmp_path):
    write_raster(tmp_path / "map.tif", np.zeros((100, 100), np.uint8), 2.0)
    dtm, _ = plane_dtm()
    write_raster(tmp_path / "dtm.tif", dtm, 8.0)
    ring, _ = ring_dtm()
    profile = {
        "driver": "GTiff",
        "width": 25,
        "height": 25,
        "count": 1,
        "dtype": ring.dtype,
        "crs": MARS_EQC,
        "transform": from_origin(RING_LEFT, RING_TOP, RING_RES, RING_RES),
    }
    (tmp_path / "scene").mkdir()
    with rasterio.open(tmp_path / "scene" / "ring_dtm.tif", "w", **profile) as dst:
        dst.write(ring, 1)
    (tmp_path / "mesh").mkdir()
    (tmp_path / "mesh" / "mesh.json").write_text(
        json.dumps(
            {
                "origin": list(ORIGIN),
                "east_scale": EAST_SCALE,
                "curvature_radius_m": MARS_RADIUS_M,
                "ring": {"dtm": "../scene/ring_dtm.tif"},
            }
        )
    )

    loaded = load_orbital_map(tmp_path / "map.tif", tmp_path / "dtm.tif", tmp_path / "mesh" / "mesh.json")

    assert loaded.curvature_radius_m == MARS_RADIUS_M
    np.testing.assert_array_equal(loaded.ring_dtm, ring)
