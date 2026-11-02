import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

import fidelis.terrain.tile as tile_module
from fidelis.terrain.tile import cut_tile, east_west_ground_scale, grid_step, lonlat_to_eqc, tile_bounds

REPO = Path(__file__).resolve().parents[2]
MARS_RADIUS_M = 3396190.0
MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"


def test_lonlat_to_eqc_is_arc_length_on_the_sphere():
    x, y = lonlat_to_eqc(90.0, -45.0, MARS_RADIUS_M)

    assert x == pytest.approx(MARS_RADIUS_M * math.pi / 2)
    assert y == pytest.approx(-MARS_RADIUS_M * math.pi / 4)


def test_east_west_ground_scale_is_cos_latitude():
    # lat_ts=0: one map metre east-west is cos(latitude) of a ground metre.
    _, y = lonlat_to_eqc(0.0, 60.0, MARS_RADIUS_M)

    assert east_west_ground_scale(y, MARS_RADIUS_M) == pytest.approx(0.5)
    assert east_west_ground_scale(0.0, MARS_RADIUS_M) == 1.0


def test_east_west_ground_scale_at_the_landing_site():
    # GDAL's inverse projection there: 1000 map metres east-west = 948.6 ground metres.
    _, y = lonlat_to_eqc(77.4508, 18.4447, MARS_RADIUS_M)

    assert east_west_ground_scale(y, MARS_RADIUS_M) == pytest.approx(0.9486, abs=1e-4)


@pytest.mark.parametrize(
    ("resolutions", "step"),
    [((4.0, 2.0, 6.0), 12.0), ((1.0, 0.25, 6.0), 6.0), ((0.25,), 0.25), ((1.5, 2.0), 6.0)],
)
def test_grid_step_is_the_smallest_length_every_resolution_divides(resolutions, step):
    assert grid_step(resolutions) == pytest.approx(step)


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


def block_means(image, factor):
    rows, cols = image.shape
    return image.reshape(rows // factor, factor, cols // factor, factor).mean(axis=(1, 3))


# 12 m wide, 24 m tall, inside the 40 m mosaics; whole 2 m and 4 m pixels.
COARSE_BOUNDS = (1004.0, 1964.0, 1016.0, 1988.0)


def test_cut_tile_area_averages_down_to_the_requested_resolution(mosaics, tmp_path):
    dtm, ortho = mosaics
    with rasterio.open(dtm) as src:
        full = src.read(1).astype(np.float64)
    expected = block_means(full[12:36, 4:16], 2)  # mosaic rows 12..35, cols 4..15

    dtm_info = cut_tile(dtm, COARSE_BOUNDS, tmp_path / "dtm_tile.tif", resolution_m=2.0)
    ortho_info = cut_tile(ortho, COARSE_BOUNDS, tmp_path / "ortho_tile.tif", resolution_m=2.0)

    with rasterio.open(tmp_path / "dtm_tile.tif") as d, rasterio.open(tmp_path / "ortho_tile.tif") as o:
        assert tuple(d.bounds) == COARSE_BOUNDS
        assert tuple(o.bounds) == COARSE_BOUNDS
        assert d.res == o.res == (2.0, 2.0)
        assert d.shape == o.shape == (12, 6)
        np.testing.assert_allclose(d.read(1), expected)
        # 8 x 8 blocks of 0.25 m pixels average to the same as 2 x 2 blocks of 1 m pixels.
        np.testing.assert_allclose(o.read(1), expected)
    assert dtm_info["resolution_m"] == ortho_info["resolution_m"] == 2.0
    assert dtm_info["source_resolution_m"] == 1.0
    assert ortho_info["source_resolution_m"] == 0.25
    assert dtm_info["shape"] == [12, 6]


def test_cut_tile_gives_the_same_tile_however_it_strips_the_read(mosaics, tmp_path, monkeypatch):
    # The real 10 km window is 40,000 x 40,000 ortho pixels, so it's read in strips.
    dtm, _ = mosaics
    cut_tile(dtm, COARSE_BOUNDS, tmp_path / "whole.tif", resolution_m=4.0)
    monkeypatch.setattr(tile_module, "STRIP_PIXELS", 50)  # a few rows of 4 m blocks per strip
    cut_tile(dtm, COARSE_BOUNDS, tmp_path / "strips.tif", resolution_m=4.0)

    with rasterio.open(tmp_path / "whole.tif") as a, rasterio.open(tmp_path / "strips.tif") as b:
        assert a.shape == (6, 3)
        np.testing.assert_array_equal(a.read(1), b.read(1))


def test_cut_tile_rounds_an_integer_image_back_to_its_own_type(tmp_path):
    # The orthoimage is 8-bit; its texture and matching map stay 8-bit.
    rows, cols = np.indices((8, 8))
    write_raster(tmp_path / "u8.tif", (rows + cols) * 10, 0.25, 1000.0, 2000.0, "uint8")

    cut_tile(tmp_path / "u8.tif", (1000.0, 1998.0, 1002.0, 2000.0), tmp_path / "t.tif", resolution_m=1.0)

    with rasterio.open(tmp_path / "t.tif") as t:
        assert t.dtypes == ("uint8",)
        np.testing.assert_array_equal(t.read(1), [[30, 70], [70, 110]])


def test_cut_tile_refuses_a_resolution_that_isnt_whole_source_pixels(mosaics, tmp_path):
    dtm, _ = mosaics

    with pytest.raises(ValueError, match="whole multiple"):
        cut_tile(dtm, COARSE_BOUNDS, tmp_path / "t.tif", resolution_m=1.5)


def test_cut_tile_refuses_bounds_that_arent_whole_target_pixels(mosaics, tmp_path):
    dtm, _ = mosaics

    with pytest.raises(ValueError, match="whole"):
        cut_tile(dtm, (1004.0, 1964.0, 1018.0, 1988.0), tmp_path / "t.tif", resolution_m=4.0)  # 14 m wide


def test_cut_tile_counts_nodata_pixels_and_leaves_them_out_of_the_average(tmp_path):
    # Averaging would otherwise hide a DTM hole inside a valid-looking coarse pixel.
    image = np.full((4, 4), 10.0, dtype=np.float32)
    image[0, 0] = -3.4e38
    with rasterio.open(
        tmp_path / "holes.tif",
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs=MARS_EQC,
        transform=from_origin(1000.0, 2000.0, 1.0, 1.0),
        nodata=-3.4e38,
    ) as dst:
        dst.write(image, 1)

    info = cut_tile(tmp_path / "holes.tif", (1000.0, 1996.0, 1004.0, 2000.0), tmp_path / "t.tif", resolution_m=2.0)

    assert info["nodata_pixels"] == 1
    with rasterio.open(tmp_path / "t.tif") as t:
        np.testing.assert_allclose(t.read(1), [[10.0, 10.0], [10.0, 10.0]])


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


# LVS imaged from 4.2 km down to 2.2 km above the ground (NASA/JPL landing press kit,
# pp. 21-22); a nadir 90 deg LCAM view covers 2 x altitude, at altitude / 512 per pixel.
LVS_TOP_M, LVS_BOTTOM_M = 4200.0, 2200.0


def test_the_configured_tile_holds_a_nadir_view_from_the_top_of_the_lvs_band():
    config = yaml.safe_load((REPO / "config/jezero.yaml").read_text())
    tile, radius = config["tile"], config["mars_radius_m"]
    _, y = lonlat_to_eqc(tile["center_lon"], tile["center_lat"], radius)

    # East-west is the short side in ground metres.
    assert tile["size_m"] * east_west_ground_scale(y, radius) > 2 * LVS_TOP_M


def test_the_configured_mesh_and_texture_are_no_coarser_than_an_lcam_pixel_at_the_bottom_of_the_band():
    tile = yaml.safe_load((REPO / "config/jezero.yaml").read_text())["tile"]

    assert tile["texture_resolution_m"] <= tile["mesh_resolution_m"] <= LVS_BOTTOM_M / 512
