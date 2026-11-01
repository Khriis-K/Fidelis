import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from fidelis.terrain.tile import cut_tile, lonlat_to_eqc, tile_bounds

REPO = Path(__file__).resolve().parents[2]
MARS_RADIUS_M = 3396190.0
MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"


def test_lonlat_to_eqc_is_arc_length_on_the_sphere():
    x, y = lonlat_to_eqc(90.0, -45.0, MARS_RADIUS_M)

    assert x == pytest.approx(MARS_RADIUS_M * math.pi / 2)
    assert y == pytest.approx(-MARS_RADIUS_M * math.pi / 4)


def test_tile_bounds_cover_the_requested_box_and_snap_to_the_grid():
    origin = (1000.5, 9000.25)

    left, bottom, right, top = tile_bounds(5000.0, 6000.0, 2000.0, origin, step_m=6.0)

    assert left <= 4000.0 and right >= 6000.0
    assert bottom <= 5000.0 and top >= 7000.0
    for edge, o in [(left, origin[0]), (right, origin[0]), (bottom, origin[1]), (top, origin[1])]:
        assert (edge - o) / 6.0 == pytest.approx(round((edge - o) / 6.0)), f"{edge} is off the 6 m grid"
    # Snapping adds less than one step on each side.
    assert right - left < 2000.0 + 2 * 6.0
    assert top - bottom < 2000.0 + 2 * 6.0


def test_tile_bounds_already_on_the_grid_are_kept_exactly():
    assert tile_bounds(3000.0, 3000.0, 1200.0, (0.0, 0.0), step_m=6.0) == (2400.0, 2400.0, 3600.0, 3600.0)


def write_raster(path, image, resolution, left, top, dtype):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=image.shape[1],
        height=image.shape[0],
        count=1,
        dtype=dtype,
        crs=MARS_EQC,
        transform=from_origin(left, top, resolution, resolution),
    ) as dst:
        dst.write(image.astype(dtype), 1)


@pytest.fixture
def mosaics(tmp_path):
    """A 1 m DTM and a 0.25 m orthoimage over the same 40 m box, like the USGS pair.

    Each value encodes its 1 m cell (row * 1000 + col), so a misaligned cut shows up
    as a mismatch between the DTM tile and the striding-down of the ortho tile.
    """
    rows, cols = np.indices((40, 40))
    dtm = rows * 1000 + cols
    ortho = np.repeat(np.repeat(dtm, 4, axis=0), 4, axis=1)
    write_raster(tmp_path / "dtm.tif", dtm, 1.0, 1000.0, 2000.0, "float32")
    write_raster(tmp_path / "ortho.tif", ortho, 0.25, 1000.0, 2000.0, "float32")
    return tmp_path / "dtm.tif", tmp_path / "ortho.tif"


def test_cut_tile_cuts_matching_dtm_and_ortho_windows(mosaics, tmp_path):
    dtm, ortho = mosaics
    bounds = (1006.0, 1970.0, 1018.0, 1988.0)

    dtm_info = cut_tile(dtm, bounds, tmp_path / "dtm_tile.tif")
    ortho_info = cut_tile(ortho, bounds, tmp_path / "ortho_tile.tif")

    with rasterio.open(tmp_path / "dtm_tile.tif") as d, rasterio.open(tmp_path / "ortho_tile.tif") as o:
        assert tuple(d.bounds) == bounds
        assert tuple(o.bounds) == bounds
        assert d.shape == (18, 12)
        assert o.shape == (72, 48)
        np.testing.assert_array_equal(o.read(1)[::4, ::4], d.read(1))
        # Top-left cell of the tile is row 12, col 6 of the mosaic.
        assert d.read(1)[0, 0] == 12 * 1000 + 6
    assert dtm_info["bounds"] == list(bounds)
    assert dtm_info["resolution_m"] == 1.0
    assert ortho_info["resolution_m"] == 0.25
    assert ortho_info["shape"] == [72, 48]


def test_cut_tile_refuses_bounds_off_the_pixel_grid(mosaics, tmp_path):
    dtm, _ = mosaics

    with pytest.raises(ValueError, match="grid"):
        cut_tile(dtm, (1006.5, 1970.0, 1018.0, 1988.0), tmp_path / "t.tif")


def test_cut_tile_refuses_bounds_outside_the_mosaic(mosaics, tmp_path):
    dtm, _ = mosaics

    with pytest.raises(ValueError, match="outside"):
        cut_tile(dtm, (990.0, 1970.0, 1018.0, 1988.0), tmp_path / "t.tif")


def test_the_configured_tile_sits_inside_the_published_mosaic_footprint():
    # Footprint from the USGS product pages; a tile hanging off the edge would be part nodata.
    config = yaml.safe_load((REPO / "config/jezero.yaml").read_text())
    tile, footprint = config["tile"], config["mosaic_footprint"]
    radius = config["mars_radius_m"]

    left, bottom, right, top = tile_bounds(
        *lonlat_to_eqc(tile["center_lon"], tile["center_lat"], radius),
        tile["size_m"],
        lonlat_to_eqc(footprint["lon_min"], footprint["lat_max"], radius),
        step_m=config["matching_map"]["lvs_map_resolution"]["value_m"],
    )

    min_x, max_y = lonlat_to_eqc(footprint["lon_min"], footprint["lat_max"], radius)
    max_x, min_y = lonlat_to_eqc(footprint["lon_max"], footprint["lat_min"], radius)
    assert min_x <= left < right <= max_x
    assert min_y <= bottom < top <= max_y
