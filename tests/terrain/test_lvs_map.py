"""The real LVS map (#36): the USGS CTX orthomosaic, cut to the scene, on CTX's own pixels."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fidelis.terrain.lvs_map import build_lvs_map, main

REPO = Path(__file__).resolve().parents[2]
MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"
# CTX's grid sits a fraction of a pixel off HiRISE's, like the real mosaic (0.749 m, 3.140 m).
CTX_LEFT, CTX_TOP = 1000.75, 4003.14
CTX_RES = 6.0
# A window of whole CTX pixels: 10 columns and 8 rows in from the mosaic's corner, 40 x 30 pixels.
BOUNDS = (CTX_LEFT + 60.0, CTX_TOP - 228.0, CTX_LEFT + 300.0, CTX_TOP - 48.0)
WINDOW = (slice(8, 38), slice(10, 50))
# No CTX pixel in the fixture has this value; every HiRISE-derived pixel does.
HIRISE_SENTINEL = 250


def write_tif(path, image, resolution, left, top, nodata=0):
    profile = {
        "driver": "GTiff",
        "width": image.shape[1],
        "height": image.shape[0],
        "count": 1,
        "dtype": image.dtype,
        "crs": MARS_EQC,
        "transform": from_origin(left, top, resolution, resolution),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(image, 1)


def ctx_image():
    """An 80 x 100 CTX mosaic of values 1-199: never nodata, never the HiRISE sentinel."""
    return (np.arange(80 * 100).reshape(80, 100) * 7 % 199 + 1).astype(np.uint8)


@pytest.fixture
def ctx_ortho(tmp_path):
    path = tmp_path / "ctx_ortho.tif"
    write_tif(path, ctx_image(), CTX_RES, CTX_LEFT, CTX_TOP)
    return path


def test_the_lvs_map_is_the_ctx_mosaics_own_pixels_under_the_bounds(ctx_ortho, tmp_path):
    info = build_lvs_map(ctx_ortho, BOUNDS, tmp_path / "map.tif")

    with rasterio.open(tmp_path / "map.tif") as m:
        np.testing.assert_array_equal(m.read(1), ctx_image()[WINDOW])
        assert tuple(m.bounds) == pytest.approx(BOUNDS)
        assert m.res == (CTX_RES, CTX_RES)
        assert m.dtypes[0] == "uint8"
    assert info["shape"] == [30, 40]
    assert info["resolution_m"] == CTX_RES


def test_the_lvs_map_stays_on_ctxs_pixel_grid_not_the_scenes(ctx_ortho, tmp_path):
    # Resampling onto the HiRISE lattice would smooth the map; LVS's was "not ... smoothed".
    build_lvs_map(ctx_ortho, BOUNDS, tmp_path / "map.tif")

    with rasterio.open(tmp_path / "map.tif") as m:
        for edge, origin in ((m.bounds.left, CTX_LEFT), (m.bounds.top, CTX_TOP)):
            assert (edge - origin) / CTX_RES == pytest.approx(round((edge - origin) / CTX_RES), abs=1e-9)


def test_bounds_off_the_ctx_pixel_grid_are_refused(ctx_ortho, tmp_path):
    # Whole HiRISE-lattice bounds would need the resampling the map must not have.
    off_grid = tuple(v + 1.5 for v in BOUNDS)
    with pytest.raises(ValueError, match="grid"):
        build_lvs_map(ctx_ortho, off_grid, tmp_path / "map.tif")


def test_a_ctx_window_with_nodata_is_refused(tmp_path):
    # A hole would be a black patch in the map that is not on Mars.
    image = ctx_image()
    image[20, 30] = 0
    write_tif(tmp_path / "ctx_ortho.tif", image, CTX_RES, CTX_LEFT, CTX_TOP)
    with pytest.raises(ValueError, match="nodata"):
        build_lvs_map(tmp_path / "ctx_ortho.tif", BOUNDS, tmp_path / "map.tif")
    # No holed map is left behind for a later step to pick up.
    assert not (tmp_path / "map.tif").exists()


# --- the command the workflow runs ---------------------------------------------------------


@pytest.fixture
def scene_dir(tmp_path):
    """A scene folder whose every HiRISE-derived raster is the sentinel value.

    ortho.tif is the HiRISE texture the scene renders; ring_ortho.tif is CTX after brightness
    matching, whose gain and offset come from HiRISE's statistics. Neither may reach the map.
    """
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "ring.json").write_text(json.dumps({"bounds": list(BOUNDS)}))
    (scene / "tile.json").write_text(json.dumps({"east_west_ground_scale": 0.9}))
    write_tif(scene / "ortho.tif", np.full((90, 120), HIRISE_SENTINEL, np.uint8), 2.0, BOUNDS[0], BOUNDS[3])
    write_tif(scene / "ring_ortho.tif", np.full((30, 40), HIRISE_SENTINEL, np.uint8), CTX_RES, BOUNDS[0], BOUNDS[3])
    return scene


def run_cli(monkeypatch, scene_dir, ctx_ortho, out):
    config = REPO / "config/jezero.yaml"
    monkeypatch.setattr(sys, "argv", ["fidelis-lvs-map", str(config), str(scene_dir), str(ctx_ortho), str(out)])
    return main()


def test_no_hirise_pixels_reach_the_lvs_map(scene_dir, ctx_ortho, tmp_path, monkeypatch):
    out = tmp_path / "lvs_map" / "map.tif"

    assert run_cli(monkeypatch, scene_dir, ctx_ortho, out) == 0

    with rasterio.open(out) as m:
        data = m.read(1)
    assert not (data == HIRISE_SENTINEL).any()
    np.testing.assert_array_equal(data, ctx_image()[WINDOW])


def test_the_lvs_map_builds_with_no_hirise_raster_in_reach(scene_dir, ctx_ortho, tmp_path, monkeypatch):
    # The builder can't read HiRISE pixels it is never given: only the ring's bounds and the scale.
    for name in ("ortho.tif", "ring_ortho.tif"):
        (scene_dir / name).unlink()
    out = tmp_path / "lvs_map" / "map.tif"

    assert run_cli(monkeypatch, scene_dir, ctx_ortho, out) == 0

    with rasterio.open(out) as m:
        np.testing.assert_array_equal(m.read(1), ctx_image()[WINDOW])


def test_the_map_record_names_its_one_source_and_its_ground_pixel_size(scene_dir, ctx_ortho, tmp_path, monkeypatch):
    out = tmp_path / "lvs_map" / "map.tif"
    run_cli(monkeypatch, scene_dir, ctx_ortho, out)

    record = json.loads(out.with_suffix(".json").read_text())
    assert record["source"] == str(ctx_ortho)
    assert "ring_source" not in record
    assert record["bounds"] == pytest.approx(list(BOUNDS))
    assert record["resolution_m"] == CTX_RES
    assert record["ground_resolution_m"] == {"east_west": pytest.approx(5.4), "north_south": 6.0}


def test_a_ctx_mosaic_at_another_resolution_than_the_lvs_maps_is_refused(scene_dir, tmp_path, monkeypatch):
    # The config says the LVS map is 6 m/pixel; a mosaic that isn't is the wrong product.
    write_tif(tmp_path / "coarse.tif", np.full((40, 50), 90, np.uint8), 12.0, CTX_LEFT, CTX_TOP)
    with pytest.raises(ValueError, match="6"):
        run_cli(monkeypatch, scene_dir, tmp_path / "coarse.tif", tmp_path / "map.tif")
