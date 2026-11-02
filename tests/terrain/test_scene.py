import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from fidelis.terrain.scene import (
    HEIGHT_STEP_TOLERANCE_M,
    band_weight,
    bilinear,
    blend_scene,
    brightness_match,
    curvature_drop,
    main,
    ring_bounds,
    to_local,
)
from fidelis.terrain.tile import lonlat_to_eqc

REPO = Path(__file__).resolve().parents[2]
MARS_RADIUS_M = 3396190.0
MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"

# --- curvature and the local frame -------------------------------------------------------


@pytest.mark.parametrize("d", [0.0, 1000.0, 5000.0, 13000.0])
def test_curvature_drop_matches_the_sphere_falling_away_from_its_tangent_plane(d):
    # Exact: a sphere of radius R sits R - sqrt(R^2 - d^2) below the plane touching it, d away.
    exact = MARS_RADIUS_M - math.sqrt(MARS_RADIUS_M**2 - d**2)
    assert curvature_drop(d, 0.0, MARS_RADIUS_M) == pytest.approx(exact, abs=1e-3)
    assert curvature_drop(d, 0.0, MARS_RADIUS_M) == pytest.approx(d**2 / (2 * MARS_RADIUS_M))


def test_curvature_drop_at_the_distances_the_issue_quotes():
    assert curvature_drop(5000.0, 0.0, MARS_RADIUS_M) == pytest.approx(3.68, abs=0.01)
    assert curvature_drop(13000.0, 0.0, MARS_RADIUS_M) == pytest.approx(24.88, abs=0.01)


def test_curvature_drop_depends_on_horizontal_distance_in_any_direction():
    assert curvature_drop(3000.0, 4000.0, MARS_RADIUS_M) == pytest.approx(curvature_drop(5000.0, 0.0, MARS_RADIUS_M))


def test_the_local_frame_scales_east_shifts_to_the_origin_and_drops_with_curvature():
    origin = (4590000.0, 1093000.0)
    mx = np.array([origin[0], origin[0] + 10000.0])
    my = np.array([origin[1], origin[1] - 5000.0])
    h = np.array([-2500.0, -2400.0])

    local = to_local(mx, my, h, origin, east_scale=0.9, radius_m=MARS_RADIUS_M)

    assert local.dtype == np.float32
    np.testing.assert_allclose(local[:, 0], [0.0, 9000.0])
    np.testing.assert_allclose(local[:, 1], [0.0, -5000.0])
    expected_drop = (9000.0**2 + 5000.0**2) / (2 * MARS_RADIUS_M)
    np.testing.assert_allclose(local[:, 2], [-2500.0, -2400.0 - expected_drop], atol=1e-3)


def test_without_a_radius_the_local_frame_is_flat():
    local = to_local(np.array([10000.0]), np.array([0.0]), np.array([-2500.0]), (0.0, 0.0), 1.0, None)
    assert local[0, 2] == -2500.0


# --- sampling and blending primitives -----------------------------------------------------

LEFT, TOP = 4589000.0, 1094000.0


def plane(mx, my):
    return -2500.0 + 0.03 * (mx - LEFT) - 0.05 * (TOP - my)


def test_bilinear_reproduces_a_plane_sampled_at_pixel_centres():
    res = 20.0
    cols, rows = np.meshgrid(np.arange(12), np.arange(9))
    z = plane(LEFT + (cols + 0.5) * res, TOP - (rows + 0.5) * res)
    mx = LEFT + np.array([10.0, 37.3, 201.9, 230.0])
    my = TOP - np.array([10.0, 55.5, 99.1, 170.0])

    np.testing.assert_allclose(bilinear(z, from_origin(LEFT, TOP, res, res), mx, my), plane(mx, my), atol=1e-9)


def test_band_weight_is_zero_on_the_edge_and_rises_linearly_to_one_a_band_inside():
    edge = (0.0, 0.0, 100.0, 60.0)  # left, bottom, right, top
    mx = np.array([0.0, 5.0, 10.0, 20.0, 50.0, 95.0, -3.0])
    my = np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0])

    np.testing.assert_allclose(band_weight(mx, my, edge, band_m=10.0), [0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0])


def test_band_weight_follows_the_nearest_edge():
    edge = (0.0, 0.0, 100.0, 60.0)
    # 4 m from the top edge, 50 from the sides.
    assert band_weight(np.array([50.0]), np.array([56.0]), edge, band_m=10.0)[0] == pytest.approx(0.4)


def test_brightness_match_maps_the_source_onto_the_reference_mean_and_spread():
    rng = np.random.default_rng(0)
    reference = rng.uniform(40, 200, 1000)
    source = 1.2 * reference - 10  # brighter, more contrast

    gain, offset = brightness_match(reference, source)

    np.testing.assert_allclose(gain * source + offset, reference, atol=1e-9)


# --- blend_scene end to end on synthetic mosaics ------------------------------------------

# The inner (HiRISE-like) tile: 120 m square at 4 m DTM / 2 m ortho.
IN_LEFT, IN_TOP, IN_SIZE = 4585862.0, 1098314.0, 120.0
# The CTX-like mosaics are off the inner grid by a fraction of a pixel, like the real ones
# (CTX's origin sits 0.749 m east, 3.140 m north of a 6 m HiRISE pixel corner).
CTX_LEFT, CTX_TOP = IN_LEFT - 360.0 + 0.749, IN_TOP + 360.0 + 3.14
CTX_OFFSET_M = 0.54  # CTX floats above HiRISE, as measured over the real overlap
SCENE_SIZE, BAND = 400.0, 40.0


def ramp(mx, my):
    """Ortho brightness that both resampling and block averaging reproduce exactly."""
    return 100.0 + 0.1 * (mx - IN_LEFT) - 0.1 * (my - IN_TOP)


def ctx_brightness(value):
    return 1.2 * value - 10.0


def write_raster(path, data, left, top, res, nodata=None):
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": data.dtype,
        "crs": MARS_EQC,
        "transform": from_origin(left, top, res, res),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def centres(left, top, res, shape):
    cols, rows = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    return left + (cols + 0.5) * res, top - (rows + 0.5) * res


def in_plane(mx, my):
    return -2500.0 + 0.01 * (mx - IN_LEFT) + 0.02 * (my - IN_TOP)


@pytest.fixture
def mosaics(tmp_path):
    tile = tmp_path / "tile"
    tile.mkdir()
    n_dtm, n_ortho = int(IN_SIZE / 4), int(IN_SIZE / 2)
    mx, my = centres(IN_LEFT, IN_TOP, 4.0, (n_dtm, n_dtm))
    write_raster(
        tile / "dtm.tif", in_plane(mx, my).astype(np.float32), IN_LEFT, IN_TOP, 4.0, nodata=-3.4028226550889045e38
    )
    mx, my = centres(IN_LEFT, IN_TOP, 2.0, (n_ortho, n_ortho))
    write_raster(tile / "ortho.tif", np.rint(ramp(mx, my)).astype(np.uint8), IN_LEFT, IN_TOP, 2.0, nodata=0)
    (tile / "tile.json").write_text(
        json.dumps({"bounds": [IN_LEFT, IN_TOP - IN_SIZE, IN_LEFT + IN_SIZE, IN_TOP], "east_west_ground_scale": 0.95})
    )

    mx, my = centres(CTX_LEFT, CTX_TOP, 20.0, (40, 40))
    ctx_dtm = tmp_path / "ctx_dtm.tif"
    write_raster(ctx_dtm, (in_plane(mx, my) + CTX_OFFSET_M).astype(np.float32), CTX_LEFT, CTX_TOP, 20.0)
    mx, my = centres(CTX_LEFT, CTX_TOP, 6.0, (134, 134))
    ctx_ortho = tmp_path / "ctx_ortho.tif"
    write_raster(ctx_ortho, np.rint(ctx_brightness(ramp(mx, my))).astype(np.uint8), CTX_LEFT, CTX_TOP, 6.0, nodata=0)
    return tile, ctx_dtm, ctx_ortho


@pytest.fixture
def blended(mosaics, tmp_path):
    tile, ctx_dtm, ctx_ortho = mosaics
    record = blend_scene(tile, ctx_dtm, ctx_ortho, tmp_path / "scene", SCENE_SIZE, BAND, MARS_RADIUS_M)
    return record, tmp_path / "scene"


def read(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.bounds


def test_the_inner_boundary_heights_meet_the_ctx_surface_within_the_stated_tolerance(blended):
    _, scene = blended
    heights, transform, _ = read(scene / "dtm.tif")
    ctx, ctx_transform, _ = read(scene / "ring_dtm.tif")
    mx, my = centres(transform.c, transform.f, 4.0, heights.shape)
    edge = np.zeros(heights.shape, bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True

    step = heights[edge] - bilinear(ctx, ctx_transform, mx[edge], my[edge])

    assert np.abs(step).max() <= HEIGHT_STEP_TOLERANCE_M
    # Unblended, the step is CTX's offset: the test can see a missing blend.
    assert np.abs(in_plane(mx[edge], my[edge]) - bilinear(ctx, ctx_transform, mx[edge], my[edge])).min() > 0.5


def test_heights_a_band_inside_the_edge_are_hirise_untouched(blended):
    _, scene = blended
    heights, transform, _ = read(scene / "dtm.tif")
    mx, my = centres(transform.c, transform.f, 4.0, heights.shape)
    edge = (IN_LEFT + 2.0, IN_TOP - IN_SIZE + 2.0, IN_LEFT + IN_SIZE - 2.0, IN_TOP - 2.0)
    inside = band_weight(mx, my, edge, BAND) == 1.0

    assert inside.any()
    np.testing.assert_array_equal(heights[inside], in_plane(mx[inside], my[inside]).astype(np.float32))


def test_across_the_band_heights_ease_from_ctx_to_hirise(blended):
    _, scene = blended
    heights, transform, _ = read(scene / "dtm.tif")
    mx, my = centres(transform.c, transform.f, 4.0, heights.shape)
    edge = (IN_LEFT + 2.0, IN_TOP - IN_SIZE + 2.0, IN_LEFT + IN_SIZE - 2.0, IN_TOP - 2.0)
    w = band_weight(mx, my, edge, BAND)

    expected = in_plane(mx, my) + (1 - w) * CTX_OFFSET_M
    np.testing.assert_allclose(heights, expected, atol=1e-3)


def test_the_ctx_ring_is_brightness_matched_to_hirise(blended):
    record, scene = blended
    ring, transform, _ = read(scene / "ring_ortho.tif")
    mx, my = centres(transform.c, transform.f, 6.0, ring.shape)

    # Unmatched, CTX is 10-30 DN brighter here; matched, only uint8 rounding remains.
    np.testing.assert_allclose(ring.astype(float), ramp(mx, my), atol=1.5)
    assert record["ring"]["brightness_gain"] == pytest.approx(1 / 1.2, rel=0.02)


def test_the_inner_texture_fades_into_the_matched_ctx_at_its_edge_and_is_untouched_inside(blended, mosaics):
    _, scene = blended
    tile, _, _ = mosaics
    texture, transform, _ = read(scene / "ortho.tif")
    original, _, _ = read(tile / "ortho.tif")
    mx, my = centres(transform.c, transform.f, 2.0, texture.shape)
    edge = (IN_LEFT + 2.0, IN_TOP - IN_SIZE + 2.0, IN_LEFT + IN_SIZE - 2.0, IN_TOP - 2.0)
    w = band_weight(mx, my, edge, BAND)

    np.testing.assert_array_equal(texture[w == 1.0], original[w == 1.0])
    # Where the weight is zero the texture is CTX's, matched: the ramp to within rounding.
    np.testing.assert_allclose(texture[w == 0.0].astype(float), ramp(mx[w == 0.0], my[w == 0.0]), atol=1.5)


def test_the_ring_covers_the_scene_size_around_the_inner_tile_centre(blended):
    _, scene = blended
    heights, transform, bounds = read(scene / "ring_dtm.tif")
    cx, cy = IN_LEFT + IN_SIZE / 2, IN_TOP - IN_SIZE / 2
    # Pixel centres, where the mesh's outermost vertices go, reach size/2 every way.
    half = SCENE_SIZE / 2
    assert bounds.left + 10.0 <= cx - half and bounds.right - 10.0 >= cx + half
    assert bounds.bottom + 10.0 <= cy - half and bounds.top - 10.0 >= cy + half
    np.testing.assert_allclose(
        heights, in_plane(*centres(transform.c, transform.f, 20.0, heights.shape)) + 0.54, atol=1e-3
    )


def test_the_scene_record_carries_the_frame_and_curvature_for_the_mesh(blended):
    record, scene = blended
    assert json.loads((scene / "tile.json").read_text())["curvature_radius_m"] == MARS_RADIUS_M
    ring = json.loads((scene / "ring.json").read_text())
    assert ring == record["ring"]
    assert ring["blend_band_m"] == BAND


def test_the_record_gives_the_drop_from_the_tiles_highest_ground_to_the_scenes_lowest(blended):
    # The sampler's bound on how much lower a view's edge can be than the ground under a
    # camera over the tile; curvature counts, since the mesh has it.
    record, scene = blended
    centre = (IN_LEFT + IN_SIZE / 2, IN_TOP - IN_SIZE / 2)

    def local_z(path):
        heights, transform, _ = read(path)
        return to_local(
            *centres(transform.c, transform.f, transform.a, heights.shape), heights, centre, 0.95, MARS_RADIUS_M
        )[..., 2]

    inner, ring = local_z(scene / "dtm.tif"), local_z(scene / "ring_dtm.tif")
    expected = inner.max() - min(inner.min(), ring.min())
    assert record["ring"]["drop_below_tile_m"] == pytest.approx(expected, abs=1e-3)
    # The ring's lowest corner is lower than anything on the tile (the plane slopes down south-west).
    assert ring.min() < inner.min()


def test_ring_bounds_sit_on_the_ctx_grid_and_reach_the_size_from_the_centre():
    origin = (CTX_LEFT, CTX_TOP)
    left, bottom, right, top = ring_bounds((IN_LEFT + 60, IN_TOP - 60), 400.0, 20.0, 6.0, origin)
    for edge, o in ((left, origin[0]), (right, origin[0]), (bottom, origin[1]), (top, origin[1])):
        assert (edge - o) / 60.0 == pytest.approx(round((edge - o) / 60.0), abs=1e-9)
    assert left + 10 <= IN_LEFT + 60 - 200 and right - 10 >= IN_LEFT + 60 + 200


def test_the_cli_reads_the_scene_section_of_the_config(mosaics, tmp_path, monkeypatch):
    tile, ctx_dtm, ctx_ortho = mosaics
    config = tmp_path / "scene.yaml"
    config.write_text(
        yaml.safe_dump({"mars_radius_m": MARS_RADIUS_M, "scene": {"size_m": SCENE_SIZE, "blend_band_m": BAND}})
    )
    out = tmp_path / "cli"
    monkeypatch.setattr(sys, "argv", ["fidelis-scene", str(config), str(tile), str(ctx_dtm), str(ctx_ortho), str(out)])

    assert main() == 0
    assert json.loads((out / "ring.json").read_text())["blend_band_m"] == BAND


# --- the configured Jezero scene ----------------------------------------------------------


def configured_scene_lonlat():
    """The configured scene's footprint in degrees, grown by a CTX pixel's rounding."""
    config = yaml.safe_load((REPO / "config/jezero.yaml").read_text())
    radius, tile, scene = config["mars_radius_m"], config["tile"], config["scene"]
    cx, cy = lonlat_to_eqc(tile["center_lon"], tile["center_lat"], radius)
    # Outward snapping adds at most one 60 m step per side, and the inner tile centre sits
    # within a 12 m step of the landing site.
    half = scene["size_m"] / 2 + 20 + 60 + 12
    to_deg = 180 / (math.pi * radius)
    return config, ((cx - half) * to_deg, (cy - half) * to_deg, (cx + half) * to_deg, (cy + half) * to_deg)


def test_the_jezero_exclusion_covers_the_whole_wide_scene():
    # No Jezero leakage: matcher training must keep out of everything the test scene shows.
    config, (lon_min, lat_min, lon_max, lat_max) = configured_scene_lonlat()
    box = config["jezero_exclusion"]

    assert box["lon_min"] <= lon_min and lon_max <= box["lon_max"]
    assert box["lat_min"] <= lat_min and lat_max <= box["lat_max"]


def test_the_jezero_exclusion_covers_both_mosaics_it_could_leak_from():
    config = yaml.safe_load((REPO / "config/jezero.yaml").read_text())
    box = config["jezero_exclusion"]
    for footprint in (config["mosaic_footprint"], config["ctx_footprint"]):
        assert box["lon_min"] <= footprint["lon_min"] and footprint["lon_max"] <= box["lon_max"]
        assert box["lat_min"] <= footprint["lat_min"] and footprint["lat_max"] <= box["lat_max"]


def test_the_configured_scene_fits_inside_the_ctx_mosaic():
    config, (lon_min, lat_min, lon_max, lat_max) = configured_scene_lonlat()
    ctx = config["ctx_footprint"]
    assert ctx["lon_min"] <= lon_min and lon_max <= ctx["lon_max"]
    assert ctx["lat_min"] <= lat_min and lat_max <= ctx["lat_max"]
