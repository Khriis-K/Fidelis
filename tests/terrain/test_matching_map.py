import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fidelis.terrain.matching_map import (
    block_mean,
    build_matching_map,
    downsample_factor,
    main,
    require_same_resolution,
)

REPO = Path(__file__).resolve().parents[2]
MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"


def test_the_factor_from_hirise_ortho_to_the_lvs_map_is_24():
    # 0.25 m/pixel HiRISE orthomosaic to the 6 m/pixel CTX-derived LVS map.
    assert downsample_factor(0.25, 6.0) == 24


def test_a_target_that_is_not_a_whole_multiple_is_rejected():
    # A fractional factor would need resampling that shifts the pixel grid; refuse instead.
    with pytest.raises(ValueError, match="whole multiple"):
        downsample_factor(0.25, 6.1)


def test_a_target_finer_than_the_source_is_rejected():
    with pytest.raises(ValueError):
        downsample_factor(6.0, 0.25)


def test_matching_ctx_and_lvs_resolutions_pass():
    # The LVS map was built at CTX's acquisition resolution, so the two facts agree.
    require_same_resolution(lvs_m=6.0, ctx_m=6.0)


def test_disagreeing_ctx_and_lvs_resolutions_are_rejected():
    # Editing one fact but not the other would silently change what the map mirrors.
    with pytest.raises(ValueError, match="CTX"):
        require_same_resolution(lvs_m=6.0, ctx_m=5.0)


def test_a_zero_target_resolution_is_rejected():
    # A config typo; a factor of 0 would divide the image into nothing.
    with pytest.raises(ValueError):
        downsample_factor(0.25, 0.0)


def test_integer_images_round_to_the_nearest_value():
    # Truncating would darken the whole map by up to one grey level.
    image = np.array([[1, 2], [2, 2]], dtype=np.uint8)  # mean 1.75

    assert block_mean(image, 2)[0, 0] == 2


def test_block_mean_averages_each_block():
    image = np.array(
        [
            [0, 2, 10, 10],
            [4, 6, 10, 30],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(block_mean(image, 2), [[3, 15]])


def test_a_constant_image_stays_constant():
    # Null case: averaging must not invent structure or shift brightness.
    image = np.full((48, 48), 117, dtype=np.uint8)

    result = block_mean(image, 24)

    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, np.full((2, 2), 117))


def test_fine_detail_averages_away():
    # A 1-pixel checkerboard is HiRISE detail CTX can't resolve: it should become flat grey.
    image = (np.indices((24, 24)).sum(axis=0) % 2 * 200).astype(np.uint8)

    np.testing.assert_array_equal(block_mean(image, 24), [[100]])


def test_nodata_pixels_are_left_out_of_the_average():
    image = np.array([[0, 0, 50, 70]], dtype=np.uint8).reshape(2, 2)

    assert block_mean(image, 2, nodata=0)[0, 0] == 60


def test_a_float32_nodata_is_recognised_even_when_float64_cant_hold_it_exactly():
    # -3.4e38 rounds when stored as float32, so comparing after a float64 cast misses it
    # and averages a DTM hole into its neighbours.
    image = np.array([[-3.4e38, 10.0], [10.0, 10.0]], dtype=np.float32)

    assert block_mean(image, 2, nodata=-3.4e38)[0, 0] == 10.0


def test_a_block_with_no_valid_pixels_stays_nodata():
    image = np.zeros((2, 4), dtype=np.uint8)
    image[:, 2:] = 90

    np.testing.assert_array_equal(block_mean(image, 2, nodata=0), [[0, 90]])


def test_a_shape_that_does_not_divide_by_the_factor_is_rejected():
    # Silently dropping edge pixels would shrink the map's footprint without telling anyone.
    with pytest.raises(ValueError, match="divide"):
        block_mean(np.zeros((25, 24)), 24)


def write_ortho(path, image, resolution, left=1200.0, top=3600.0, nodata=0):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=image.shape[1],
        height=image.shape[0],
        count=1,
        dtype=image.dtype,
        crs=MARS_EQC,
        transform=from_origin(left, top, resolution, resolution),
        nodata=nodata,
    ) as dst:
        dst.write(image, 1)


def test_build_matching_map_keeps_the_footprint_and_coarsens_the_pixels(tmp_path):
    ortho = tmp_path / "ortho.tif"
    image = np.arange(48 * 72, dtype=np.uint16).reshape(48, 72).astype(np.uint8)
    write_ortho(ortho, image, 0.25)
    out = tmp_path / "map.tif"

    info = build_matching_map(ortho, 6.0, out)

    with rasterio.open(ortho) as src, rasterio.open(out) as m:
        assert m.bounds == src.bounds
        assert m.res == (6.0, 6.0)
        assert m.shape == (2, 3)
        assert m.crs == src.crs
        assert m.nodata == src.nodata
        np.testing.assert_array_equal(m.read(1), block_mean(image, 24, nodata=0))
    assert info["factor"] == 24
    assert info["resolution_m"] == 6.0
    assert info["shape"] == [2, 3]


def test_the_map_record_says_how_wide_its_pixels_are_on_the_ground(tmp_path, monkeypatch):
    # Map pixels are 6 map metres; east-west that's 6 x cos(latitude) ground metres, and a
    # matcher comparing against LCAM's ground metres needs to know it.
    write_ortho(tmp_path / "ortho.tif", np.full((48, 72), 100, dtype=np.uint8), 0.25)
    (tmp_path / "tile.json").write_text(json.dumps({"east_west_ground_scale": 0.9}))
    out = tmp_path / "map" / "map.tif"
    config = REPO / "config/jezero.yaml"
    monkeypatch.setattr(sys, "argv", ["fidelis-matching-map", str(config), str(tmp_path), str(out)])

    assert main() == 0

    record = json.loads(out.with_suffix(".json").read_text())
    assert record["resolution_m"] == 6.0
    assert record["east_west_ground_scale"] == 0.9
    assert record["ground_resolution_m"] == {"east_west": pytest.approx(5.4), "north_south": 6.0}
