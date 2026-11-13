"""Shadow reduction on the orthoimage (#19): divide out the shading the terrain baked into the photo."""

import json
import math
import sys

import cv2
import mitsuba as mi
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fidelis.render.camera import Intrinsics, look_at
from fidelis.render.mesh_render import Sun, render, terrain_scene
from fidelis.terrain.deshade import (
    build_deshaded_texture,
    deshade,
    fitted_sun,
    flat_detail_kept,
    main,
    slope_brightness_ratio,
    slopes,
)
from fidelis.terrain.mesh import build_terrain_mesh, terrain_shapes

MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"
LEFT, TOP = 4589870.0, 1094306.0
DTM_RES, K = 4.0, 2  # a 4 m DTM under a 2 m orthoimage, as in the Jezero scene
N = 512  # DTM pixels a side: about 2 km, so the fit has blocks of the size the Jezero config uses
EAST_SCALE = 0.9486  # ground metres per map metre east-west
DX, DY = DTM_RES * EAST_SCALE, DTM_RES
FIT = {"block_m": 1000.0, "highpass_m": 100.0, "shade_floor": 0.3}
PARAMS = FIT | {"detail_m": 8.0}  # the scene config's `deshade` section, less its check sun
WEST_SUN, EAST_SUN = (267.0, 42.0), (219.0, 37.0)  # bearing, elevation: two HiRISE images' suns
# How close a block's fitted sun bearing comes to the true one here. The ground's own patches
# line up with its slopes a little by chance, and the fit can't tell that from shading.
BEARING_TOLERANCE_DEG = 12

# Rolling terrain as a sum of waves (amplitude m, wavelength m, heading deg, phase), so its
# slopes are known exactly at any point. About 7 degrees rms, so shading swings brightness by
# about 10%: a little rougher than the Jezero tile at 4 m (9%, the #19 comment).
WAVES = [(1.0, 61.0, 20.0, 0.3), (1.3, 97.0, 110.0, 1.1), (2.0, 173.0, 250.0, 2.0), (1.7, 131.0, 330.0, 4.0)]


def ground(shape, res):
    """Ground metres east and north of every pixel centre; row 0 is the northmost."""
    rows, cols = shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    return (c + 0.5) * res * EAST_SCALE, -(r + 0.5) * res


def terrain(shape, res):
    """Heights and their true east and north slopes at every pixel centre."""
    x, y = ground(shape, res)
    z, zx, zy = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for amplitude, wavelength, heading_deg, phase in WAVES:
        k = 2 * math.pi / wavelength
        kx, ky = k * math.sin(math.radians(heading_deg)), k * math.cos(math.radians(heading_deg))
        z += amplitude * np.sin(kx * x + ky * y + phase)
        zx += amplitude * kx * np.cos(kx * x + ky * y + phase)
        zy += amplitude * ky * np.cos(kx * x + ky * y + phase)
    return z, zx, zy


def lambert(zx, zy, bearing_deg, elevation_deg):
    """A diffuse surface's brightness under the sun, as a fraction of flat ground's."""
    az, el = math.radians(bearing_deg), math.radians(elevation_deg)
    sun = (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))
    cosine = (-zx * sun[0] - zy * sun[1] + sun[2]) / np.sqrt(zx**2 + zy**2 + 1)
    return np.clip(cosine, 0, None) / sun[2]


def albedo(shape):
    """The ground's own light and dark: broad patches plus fine grain, unrelated to the slopes."""
    rng = np.random.default_rng(19)
    patches = cv2.GaussianBlur(rng.standard_normal(shape), (0, 0), 12)
    grain = cv2.GaussianBlur(rng.standard_normal(shape), (0, 0), 1.5)
    return np.clip(110 + 25 * patches / patches.std() + 12 * grain / grain.std(), 40, None)


@pytest.fixture(scope="module")
def world():
    """A DTM, the ground's true albedo at orthoimage resolution, and the true slopes there."""
    heights, _, _ = terrain((N, N), DTM_RES)
    _, zx, zy = terrain((N * K, N * K), DTM_RES / K)
    return heights, albedo((N * K, N * K)), zx, zy


def photo(world, sun=WEST_SUN):
    _, true_albedo, zx, zy = world
    return true_albedo * lambert(zx, zy, *sun)


def relative_error(image, truth):
    return float(np.sqrt(np.mean((image / truth - 1) ** 2)))


def test_dividing_out_the_fitted_shading_recovers_the_grounds_own_brightness(world):
    heights, true_albedo, _, _ = world
    shaded = photo(world)

    recovered, _ = deshade(shaded, heights, DX, DY, **FIT)

    assert relative_error(shaded, true_albedo) > 0.08  # the photo's shading is worth removing
    assert relative_error(recovered, true_albedo) < 0.02


def test_the_fit_finds_where_the_photos_sun_was(world):
    heights = world[0]

    _, fit = deshade(photo(world), heights, DX, DY, **FIT)

    bearing, elevation = fitted_sun(fit.p, fit.q)
    assert np.all(np.abs(bearing - WEST_SUN[0]) < BEARING_TOLERANCE_DEG), bearing
    assert np.all(np.abs(elevation - WEST_SUN[1]) < 5), elevation


def test_a_mosaic_of_two_photos_gets_each_ones_own_sun(world):
    # The Jezero texture is a patchwork of HiRISE images taken under different suns, and
    # its seams aren't published. One sun for the whole tile would be wrong on part of it.
    heights, true_albedo, zx, zy = world
    mosaic = true_albedo * lambert(zx, zy, *WEST_SUN)
    east = np.s_[:, N * K // 2 :]
    mosaic[east] = (true_albedo * lambert(zx, zy, *EAST_SUN))[east]

    recovered, fit = deshade(mosaic, heights, DX, DY, **FIT)

    bearing, _ = fitted_sun(fit.p, fit.q)
    assert bearing.shape == (2, 2)
    assert np.all(np.abs(bearing[:, 0] - WEST_SUN[0]) < BEARING_TOLERANCE_DEG), bearing
    assert np.all(np.abs(bearing[:, -1] - EAST_SUN[0]) < BEARING_TOLERANCE_DEG), bearing
    assert relative_error(recovered, true_albedo) < 0.03


def test_ground_with_no_baked_shading_is_left_alone(world):
    # The null case: brightness unrelated to the slopes must not be "corrected". What is left
    # is the chance fit above: a fifth of what the same step takes out of a shaded photo.
    heights, true_albedo, _, _ = world

    recovered, _ = deshade(true_albedo, heights, DX, DY, **FIT)

    assert relative_error(recovered, true_albedo) < 0.02
    assert relative_error(photo(world), true_albedo) > 5 * relative_error(recovered, true_albedo)

    # Steeper ground gives a made-up correction more to work with, and must come back as clean.
    on_steep_ground, _ = deshade(true_albedo, 3 * heights, DX, DY, **FIT)
    assert relative_error(on_steep_ground, true_albedo) < 0.02


def test_flat_ground_comes_back_unchanged(world):
    true_albedo = world[1]
    flat = np.full((N, N), -2600.0)

    recovered, _ = deshade(true_albedo, flat, DX, DY, **FIT)

    np.testing.assert_allclose(recovered, true_albedo)


def test_a_slope_the_fit_would_turn_black_is_floored_instead_of_blowing_up(world):
    # A cliff facing east, away from the photo's western sun: the fitted shade there is far
    # below zero, and dividing by it would flip the pixel's sign or blow it up.
    heights = world[0].copy()
    heights[200:280, 240:] -= 60.0
    shaded = photo(world)

    recovered, _ = deshade(shaded, heights, DX, DY, **FIT)

    assert recovered.min() >= 0
    assert recovered.max() <= shaded.max() / FIT["shade_floor"] + 1e-9


def test_slopes_are_per_ground_metre_east_and_north():
    # 1 m up per pixel east, 2 m down per pixel south (so 2 m up per pixel north).
    r, c = np.mgrid[:6, :8]
    heights = 1.0 * c - 2.0 * r

    zx, zy = slopes(heights, DX, DY)

    np.testing.assert_allclose(zx, 1.0 / DX)
    np.testing.assert_allclose(zy, 2.0 / DY)


def test_a_sun_in_the_west_brightens_slopes_that_face_west():
    # Brightness 1 + p zx + q zy: ground rising to the east (zx > 0) faces west.
    bearing, elevation = fitted_sun(np.array([[1.0]]), np.array([[0.0]]))

    assert bearing[0, 0] == pytest.approx(270.0)
    assert elevation[0, 0] == pytest.approx(45.0)


def test_sun_facing_slopes_are_brighter_in_the_photo_and_not_after(world):
    heights, _, _, _ = world
    zx, zy = slopes(heights, DX, DY)
    shaded = photo(world)
    recovered, _ = deshade(shaded, heights, DX, DY, **FIT)

    def at_dtm(image):
        return image.reshape(N, K, N, K).mean(axis=(1, 3))

    before = slope_brightness_ratio(at_dtm(shaded), zx, zy, WEST_SUN[0], PARAMS["highpass_m"] / DTM_RES)
    after = slope_brightness_ratio(at_dtm(recovered), zx, zy, WEST_SUN[0], PARAMS["highpass_m"] / DTM_RES)

    assert before > 1.15
    assert after == pytest.approx(1.0, abs=0.02)


def test_fine_detail_on_flat_ground_survives_and_a_blur_is_caught(world):
    # The null case for the metric: blurring the photo flat also "reduces shadow contrast".
    heights, _, _, _ = world
    zx, zy = slopes(heights, DX, DY)
    shaded = photo(world)
    recovered, _ = deshade(shaded, heights, DX, DY, **FIT)
    blurred = cv2.GaussianBlur(shaded, (0, 0), 8)
    detail_px = PARAMS["detail_m"] / (DTM_RES / K)

    assert flat_detail_kept(shaded, recovered, zx, zy, detail_px) == pytest.approx(1.0, abs=0.05)
    assert flat_detail_kept(shaded, blurred, zx, zy, detail_px) < 0.5


def write_raster(path, data, res, nodata=None):
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": data.dtype,
        "crs": MARS_EQC,
        "transform": from_origin(LEFT, TOP, res, res),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


CHECK_SUN = {"azimuth_deg": WEST_SUN[0], "half_width_m": 800.0, "source": "the fixture"}


@pytest.fixture
def scene_dir(world, tmp_path):
    heights = world[0]
    write_raster(tmp_path / "dtm.tif", (heights - 2600).astype(np.float32), DTM_RES, nodata=-3.4e38)
    write_raster(tmp_path / "ortho.tif", np.clip(np.round(photo(world)), 1, 255).astype(np.uint8), DTM_RES / K, 0)
    (tmp_path / "tile.json").write_text(json.dumps({"east_west_ground_scale": EAST_SCALE}))
    return tmp_path


def test_the_texture_is_written_on_the_orthoimages_own_pixels(scene_dir, world, tmp_path):
    out = tmp_path / "mesh" / "texture_deshaded.png"

    record = build_deshaded_texture(
        scene_dir / "dtm.tif", scene_dir / "ortho.tif", out, EAST_SCALE, PARAMS | {"check_sun": CHECK_SUN}
    )

    texture = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert texture.shape == (N * K, N * K)
    assert texture.dtype == np.uint8
    assert relative_error(texture.astype(float), world[1]) < 0.025
    assert not list(out.parent.glob("*.tmp*"))  # written to a temp file, then renamed
    assert record == json.loads(out.with_suffix(".json").read_text())


def test_the_record_carries_the_metric_and_the_fitted_suns(scene_dir, tmp_path):
    record = build_deshaded_texture(
        scene_dir / "dtm.tif",
        scene_dir / "ortho.tif",
        tmp_path / "texture_deshaded.png",
        EAST_SCALE,
        PARAMS | {"check_sun": CHECK_SUN},
    )

    ratio = record["slope_brightness_ratio"]
    assert ratio["before"] > 1.15
    assert ratio["after"] == pytest.approx(1.0, abs=0.03)
    assert ratio["sun"] == CHECK_SUN
    assert record["flat_detail_kept"] == pytest.approx(1.0, abs=0.05)
    assert np.all(np.abs(np.array(record["fitted_sun"]["bearing_deg"]) - WEST_SUN[0]) < BEARING_TOLERANCE_DEG)
    assert record["clipped_fraction"] < 0.01
    assert {k: record[k] for k in PARAMS} == PARAMS
    # East-west, a map metre is less than a ground metre: slopes must be per ground metre.
    assert record["dtm_pixel_ground_m"] == pytest.approx({"east": DTM_RES * EAST_SCALE, "north": DTM_RES})


def test_an_orthoimage_that_is_not_a_whole_multiple_of_the_dtm_is_rejected(scene_dir, world, tmp_path):
    write_raster(scene_dir / "ortho.tif", np.ones((N * K - 1, N * K), np.uint8), DTM_RES / K, 0)

    with pytest.raises(ValueError, match="whole number"):
        build_deshaded_texture(
            scene_dir / "dtm.tif",
            scene_dir / "ortho.tif",
            tmp_path / "t.png",
            EAST_SCALE,
            PARAMS | {"check_sun": CHECK_SUN},
        )


def test_a_dtm_with_nodata_is_rejected(scene_dir, world, tmp_path):
    heights = (world[0] - 2600).astype(np.float32)
    heights[5, 7] = -3.4e38
    write_raster(scene_dir / "dtm.tif", heights, DTM_RES, nodata=-3.4e38)

    with pytest.raises(ValueError, match="nodata"):
        build_deshaded_texture(
            scene_dir / "dtm.tif",
            scene_dir / "ortho.tif",
            tmp_path / "t.png",
            EAST_SCALE,
            PARAMS | {"check_sun": CHECK_SUN},
        )
    assert not (tmp_path / "t.png").exists()


def test_the_cli_reads_its_settings_from_the_config(scene_dir, tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"deshade": PARAMS | {"check_sun": CHECK_SUN}}))  # JSON is YAML
    out = tmp_path / "mesh" / "texture_deshaded.png"
    monkeypatch.setattr(sys, "argv", ["fidelis-deshade", str(config), str(scene_dir), str(out)])

    assert main() == 0

    assert out.exists()
    assert json.loads(out.with_suffix(".json").read_text())["block_m"] == PARAMS["block_m"]


def test_the_renderer_can_be_pointed_at_the_deshaded_texture(scene_dir, tmp_path):
    mesh_dir = tmp_path / "mesh"
    build_terrain_mesh(scene_dir / "dtm.tif", scene_dir / "ortho.tif", mesh_dir, east_scale=EAST_SCALE)
    mi.set_variant("scalar_rgb")

    def texture_file(shapes):
        return shapes["terrain"]["bsdf"]["reflectance"]["filename"]

    assert texture_file(terrain_shapes(mesh_dir)).endswith("texture.png")
    assert texture_file(terrain_shapes(mesh_dir, texture="texture_deshaded.png")).endswith("texture_deshaded.png")


def test_a_render_uses_the_texture_it_was_asked_for(scene_dir, tmp_path):
    mesh_dir = tmp_path / "mesh"
    build_terrain_mesh(scene_dir / "dtm.tif", scene_dir / "ortho.tif", mesh_dir, east_scale=EAST_SCALE)
    cv2.imwrite(str(mesh_dir / "texture_deshaded.png"), np.full((N * K, N * K), 255, np.uint8))  # far brighter
    mi.set_variant("scalar_rgb")
    ground = float(world_ground(scene_dir))
    pose = look_at(eye=[0.0, 0.0, ground + 300.0], target=[0.0, 0.0, ground], up=[0.0, 1.0, 0.0])
    k = Intrinsics.from_fov(width=8, height=8, fov_x_deg=60.0)
    sun = Sun(180.0, 60.0)

    photo_render = render(terrain_scene(mesh_dir, sun), pose, k, spp=4)
    other_render = render(terrain_scene(mesh_dir, sun, "texture_deshaded.png"), pose, k, spp=4)

    assert other_render.mean() > 1.5 * photo_render.mean()


def world_ground(scene_dir):
    with rasterio.open(scene_dir / "dtm.tif") as src:
        return src.read(1).mean()
