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
