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
    build_scene_map,
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


# --- the wide scene: one map over the inner tile and the CTX ring (#33) -------------------

# Inner texture: 2 m pixels over 120 m from (1200, 3600). Ring: 6 m pixels over 480 m,
# on a grid 0.75 m east and 3.14 m north of the inner one's, like the real CTX mosaic.
RING_LEFT, RING_TOP = 1200.0 - 180.0 + 0.75, 3600.0 + 180.0 + 3.14


def ring_ramp(mx, my):
    """Ring brightness that bilinear resampling reproduces exactly."""
    return 60.0 + 0.2 * (mx - RING_LEFT) + 0.1 * (RING_TOP - my)


@pytest.fixture
def scene_textures(tmp_path):
    inner = (np.arange(60 * 60).reshape(60, 60) % 200 + 20).astype(np.uint8)
    write_ortho(tmp_path / "ortho.tif", inner, 2.0)
    cols, rows = np.meshgrid(np.arange(80), np.arange(80))
    ring = ring_ramp(RING_LEFT + (cols + 0.5) * 6.0, RING_TOP - (rows + 0.5) * 6.0)
    write_ortho(tmp_path / "ring_ortho.tif", np.rint(ring).astype(np.uint8), 6.0, left=RING_LEFT, top=RING_TOP)
    return inner, tmp_path / "ortho.tif", tmp_path / "ring_ortho.tif"


def test_the_scene_map_sits_on_the_inner_grid_inside_the_ring(scene_textures, tmp_path):
    _, inner, ring = scene_textures
    build_scene_map(inner, ring, 6.0, tmp_path / "map.tif")

    with rasterio.open(tmp_path / "map.tif") as m, rasterio.open(ring) as r:
        assert m.res == (6.0, 6.0)
        # On the inner tile's 6 m lattice (1200, 3600 + k * 6), and inside the ring's data.
        for edge, origin in ((m.bounds.left, 1200.0), (m.bounds.top, 3600.0)):
            assert (edge - origin) / 6.0 == pytest.approx(round((edge - origin) / 6.0), abs=1e-9)
        assert r.bounds.left <= m.bounds.left and m.bounds.right <= r.bounds.right
        assert r.bounds.bottom <= m.bounds.bottom and m.bounds.top <= r.bounds.top
        # Shrinking to the lattice loses less than a pixel per side.
        assert m.bounds.left - r.bounds.left < 6.0 and r.bounds.top - m.bounds.top < 6.0


def test_the_scene_map_is_the_inner_texture_averaged_inside_and_the_ring_resampled_outside(scene_textures, tmp_path):
    image, inner, ring = scene_textures
    build_scene_map(inner, ring, 6.0, tmp_path / "map.tif")

    with rasterio.open(tmp_path / "map.tif") as m:
        data, t = m.read(1), m.transform
    col0, row0 = ~t @ (1200.0, 3600.0)
    col0, row0 = round(col0), round(row0)
    np.testing.assert_array_equal(data[row0 : row0 + 20, col0 : col0 + 20], block_mean(image, 3, nodata=0))

    outside = np.ones(data.shape, bool)
    outside[row0 : row0 + 20, col0 : col0 + 20] = False
    cols, rows = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    mx, my = t @ (cols + 0.5, rows + 0.5)
    np.testing.assert_allclose(data[outside].astype(float), ring_ramp(mx, my)[outside], atol=1.0)


def test_the_map_cli_builds_the_scene_map_when_there_is_a_ring(scene_textures, tmp_path, monkeypatch):
    (tmp_path / "tile.json").write_text(json.dumps({"east_west_ground_scale": 0.9}))
    (tmp_path / "ring.json").write_text(json.dumps({"blend_band_m": 40.0}))
    out = tmp_path / "map" / "map.tif"
    config = REPO / "config/jezero.yaml"
    monkeypatch.setattr(sys, "argv", ["fidelis-matching-map", str(config), str(tmp_path), str(out)])

    assert main() == 0

    with rasterio.open(out) as m, rasterio.open(tmp_path / "ortho.tif") as inner:
        assert m.bounds.left < inner.bounds.left and m.bounds.top > inner.bounds.top
    assert json.loads(out.with_suffix(".json").read_text())["ring_source"].endswith("ring_ortho.tif")
