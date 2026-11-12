"""The whole-map offset between the LVS map (CTX) and the rendered scene (HiRISE), #40.

The synthetic scenes are a sum of sinusoids evaluated analytically at every pixel
centre, so a shifted or warped copy has no interpolation error of its own: whatever
the registration gets wrong is the registration's.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from fidelis.map_frame import load_map_offset
from fidelis.terrain.map_offset import measure_offset, shift_between, summarise

REPO = Path(__file__).resolve().parents[2]
MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"
RNG = np.random.default_rng(40)
# A broadband texture, as real terrain is: 1500 waves in random directions, wavelengths
# 15 m (above the 6 m map's 12 m Nyquist) to 1 km, amplitude falling as 1/frequency.
# A handful of waves would not do: phase correlation whitens the spectrum, and with few
# tones the window's own edges dominate and pull every shift towards zero.
_K = 2 * np.pi / np.exp(RNG.uniform(np.log(15), np.log(1000), 1500))
_THETA = RNG.uniform(0, 2 * np.pi, 1500)
_KX, _KY = _K * np.cos(_THETA), _K * np.sin(_THETA)
_COEF = np.exp(1j * RNG.uniform(0, 2 * np.pi, 1500)) / _K
_SCALE = 3 * np.sqrt(np.sum(np.abs(_COEF) ** 2) / 2)
LEFT, TOP, SIZE_M = 4_585_862.0, 1_098_314.0, 3840.0


def texture(x, y):
    """The texture at map x (east), y (north), clipped to 0-1. x and y are a grid: x varies along rows."""
    x0, y0 = x[0, :], y[:, 0]
    # Separable: e^{i(kx x + ky y)} = e^{i kx x} e^{i ky y}, so the sum is a matrix product.
    # That needs x to depend on the column only and y on the row only (warps included).
    assert np.allclose(x, x0[None, :]) and np.allclose(y, y0[:, None])
    z = np.real((np.exp(1j * np.outer(y0, _KY)) * _COEF) @ np.exp(1j * np.outer(x0, _KX)).T)
    return np.clip(0.5 + z / (2 * _SCALE), 0, 1)


def grid(left, top, res, shape):
    rows, cols = shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    return left + (c + 0.5) * res, top - (r + 0.5) * res


def write(path, image, left, top, res):
    profile = dict(
        driver="GTiff",
        width=image.shape[1],
        height=image.shape[0],
        count=1,
        dtype="uint8",
        crs=MARS_EQC,
        transform=from_origin(left, top, res, res),
        nodata=0,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(image, 1)


def scene_pair(tmp_path, displacement, gain=1.0, bias=0.0):
    """A 2 m HiRISE-like tile and a 6 m CTX-like ortho, off the tile's grid as the real one is.

    displacement(x, y) -> (de, dn): where the CTX image shows the ground the tile shows at (x, y).
    """
    n = int(SIZE_M / 2)
    x, y = grid(LEFT, TOP, 2.0, (n, n))
    write(tmp_path / "tile.tif", np.clip(np.rint(20 + 215 * texture(x, y)), 1, 255).astype(np.uint8), LEFT, TOP, 2.0)
    # The real CTX origin sits 0.749 m east and 3.140 m north of a 6 m tile corner; reach past the tile.
    c_left, c_top = LEFT - 600 + 0.749, TOP + 600 + 3.140
    m = int((SIZE_M + 1200) / 6)
    x, y = grid(c_left, c_top, 6.0, (m, m))
    de, dn = displacement(x, y)
    ctx = gain * (20 + 215 * texture(x - de, y - dn)) + bias
    write(tmp_path / "ctx.tif", np.clip(np.rint(ctx), 1, 255).astype(np.uint8), c_left, c_top, 6.0)
    return tmp_path / "tile.tif", tmp_path / "ctx.tif"


# --- shift_between: one window ------------------------------------------------------------


def test_shift_between_recovers_a_subpixel_shift_and_its_sign():
    x, y = grid(0.0, 0.0, 6.0, (128, 128))
    ref = texture(x, y)
    # The moved image shows each feature 2.3 px east and 1.6 px south (rows grow southward).
    mov = texture(x - 2.3 * 6.0, y + 1.6 * 6.0)
    dx, dy, response = shift_between(ref, mov)
    assert dx == pytest.approx(2.3, abs=0.05)
    assert dy == pytest.approx(1.6, abs=0.05)
    assert response > 0.3


def test_shift_between_ignores_brightness_gain_and_offset():
    # CTX and HiRISE differ in brightness (gain 1.25, offset -68 DN over the tile, ring.json).
    # Neither the shift nor the response, which decides whether a window counts, may move.
    x, y = grid(0.0, 0.0, 6.0, (128, 128))
    plain = shift_between(texture(x, y), texture(x + 6.0, y))
    dx, dy, response = shift_between(texture(x, y), 3.0 * texture(x + 6.0, y) + 50)
    assert (dx, dy) == (pytest.approx(-1.0, abs=0.05), pytest.approx(0.0, abs=0.05))
    assert (dx, dy, response) == pytest.approx(plain, abs=1e-6)


def test_a_featureless_window_has_no_correlation_peak():
    # The null case: nothing to register gives a response near zero, never a confident shift.
    x, y = grid(0.0, 0.0, 6.0, (128, 128))
    _, _, response = shift_between(texture(x, y), np.full((128, 128), 0.5))
    assert response < 0.05


# --- summarise: the whole-map shift and what's left over ----------------------------------


def window_table(de, dn, east_km=None, north_km=None):
    """Windows on a 6 x 6 grid over a 6 km square, centred on 0."""
    e, n = np.meshgrid(np.arange(6) - 2.5, np.arange(6) - 2.5)
    e, n = e.ravel(), n.ravel()
    de = np.broadcast_to(de(e, n) if callable(de) else de, e.shape).astype(float)
    dn = np.broadcast_to(dn(e, n) if callable(dn) else dn, e.shape).astype(float)
    return {"east_km": e, "north_km": n, "de_m": de.copy(), "dn_m": dn.copy(), "response": np.ones(e.shape)}


def test_a_pure_shift_gives_that_shift_no_residual_and_no_trend():
    s = summarise(window_table(14.0, -9.0), min_response=0.1)
    assert (s["east_m"], s["north_m"]) == (pytest.approx(14.0), pytest.approx(-9.0))
    assert s["residual_rms_m"] == pytest.approx(0.0, abs=1e-9)
    for slope in s["trend_m_per_km"].values():
        assert slope["value"] == pytest.approx(0.0, abs=1e-9)
    assert s["windows_used"] == 36


def test_a_shift_that_grows_across_the_map_shows_up_as_a_trend():
    # 2 m of extra eastward offset per km east: local distortion, which one shift can't remove.
    table = window_table(lambda e, n: 10.0 + 2.0 * e, 0.0)
    s = summarise(table, min_response=0.1)
    assert s["east_m"] == pytest.approx(10.0)
    assert s["trend_m_per_km"]["de_per_east"]["value"] == pytest.approx(2.0)
    assert s["trend_m_per_km"]["de_per_north"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert s["residual_rms_m"] > 3.0


def test_the_uncertainty_is_the_standard_error_of_the_window_shifts():
    table = window_table(5.0, 0.0)
    table["de_m"] += RNG.normal(0, 3.0, 36)
    s = summarise(table, min_response=0.1)
    used = table["de_m"]
    assert s["east_sd_m"] == pytest.approx(np.std(used, ddof=1))
    assert s["east_se_m"] == pytest.approx(np.std(used, ddof=1) / 6)


def test_weak_windows_and_wild_outliers_are_left_out():
    table = window_table(5.0, 2.0)
    table["response"][0] = 0.01  # too weak to trust
    table["de_m"][1] = 400.0  # a wrong peak
    s = summarise(table, min_response=0.1)
    assert s["windows_total"] == 36
    assert s["windows_weak"] == 1
    assert s["windows_outlier"] == 1
    assert s["windows_used"] == 34
    assert s["east_m"] == pytest.approx(5.0)
    assert s["residual_rms_m"] == pytest.approx(0.0, abs=1e-9)


def test_no_usable_window_is_an_error_not_a_zero_offset():
    table = window_table(5.0, 2.0)
    table["response"][:] = 0.0
    with pytest.raises(ValueError, match="no window"):
        summarise(table, min_response=0.1)


# --- measure_offset: the two rasters end to end ------------------------------------------


def test_a_known_whole_map_shift_is_recovered(tmp_path):
    tile, ctx = scene_pair(tmp_path, lambda x, y: (np.full_like(x, 14.0), np.full_like(x, -9.0)), gain=1.25, bias=-30)
    result = measure_offset(tile, ctx, window_px=128)
    assert result["east_m"] == pytest.approx(14.0, abs=0.5)
    assert result["north_m"] == pytest.approx(-9.0, abs=0.5)
    assert result["residual_rms_m"] < 0.5
    assert result["windows_used"] == 16  # 639 whole CTX pixels across the tile: 4 x 4 windows of 128


@pytest.mark.parametrize("east_m", [1.5, 3.0, 4.5])
def test_shifts_between_whole_ctx_pixels_are_not_pulled_to_whole_pixels(tmp_path, east_m):
    # A quarter, half and three quarters of a 6 m pixel. Resampling the coarse image
    # locks shifts towards whole pixels; the measurement must not.
    tile, ctx = scene_pair(tmp_path, lambda x, y: (np.full_like(x, east_m), np.zeros_like(x)))
    result = measure_offset(tile, ctx, window_px=128)
    assert result["east_m"] == pytest.approx(east_m, abs=0.15)
    assert result["north_m"] == pytest.approx(0.0, abs=0.15)


def test_identical_images_give_zero_offset(tmp_path):
    # The CTX grid sits a fraction of a pixel off the tile's; resampling must not invent a shift.
    tile, ctx = scene_pair(tmp_path, lambda x, y: (np.zeros_like(x), np.zeros_like(x)))
    result = measure_offset(tile, ctx, window_px=128)
    assert result["east_m"] == pytest.approx(0.0, abs=0.3)
    assert result["north_m"] == pytest.approx(0.0, abs=0.3)


def test_a_linear_warp_is_reported_as_a_trend_not_absorbed(tmp_path):
    centre_x = LEFT + SIZE_M / 2
    tile, ctx = scene_pair(tmp_path, lambda x, y: (6.0 + 3.0 * (x - centre_x) / 1000, np.zeros_like(x)))
    result = measure_offset(tile, ctx, window_px=128)
    assert result["east_m"] == pytest.approx(6.0, abs=0.7)
    assert result["trend_m_per_km"]["de_per_east"]["value"] == pytest.approx(3.0, abs=0.5)
    assert result["trend_m_per_km"]["de_per_north"]["value"] == pytest.approx(0.0, abs=0.5)


def test_the_result_records_the_area_used(tmp_path):
    tile, ctx = scene_pair(tmp_path, lambda x, y: (np.zeros_like(x), np.zeros_like(x)))
    result = measure_offset(tile, ctx, window_px=128)
    # The windows' extent: 4 x 4 windows of 768 m, inside the tile and centred on it to
    # within a CTX pixel (CTX's grid is off the tile's).
    left, bottom, right, top = result["bounds"]
    assert right - left == pytest.approx(3072) and top - bottom == pytest.approx(3072)
    assert LEFT <= left and right <= LEFT + SIZE_M and TOP - SIZE_M <= bottom and top <= TOP
    assert (left + right) / 2 == pytest.approx(LEFT + SIZE_M / 2, abs=6)
    assert (top + bottom) / 2 == pytest.approx(TOP - SIZE_M / 2, abs=6)
    assert result["window_m"] == 768.0
    assert len(result["windows"]) == 16


# --- the scene config carries it for the map-frame error (#35) ----------------------------


def test_the_config_records_the_offset_and_where_it_came_from():
    # ctx_map_offset waits for the CTX map (#36), which copies it into the live map_offset.
    # It has map_offset's shape, so map_frame.load_map_offset can read it as it is.
    config = yaml.safe_load((REPO / "config" / "jezero.yaml").read_text())
    record = config["ctx_map_offset"]
    offset = load_map_offset({"map_offset": record})
    assert (offset.east_m, offset.north_m) == (record["east_m"], record["north_m"])
    assert record["uncertainty_m"]["east"] > 0 and record["uncertainty_m"]["north"] > 0
    # east_m is ground metres, the frame map_frame_error subtracts in: the map-metre value
    # scaled by the tile's east-west ground scale (tile.json, 0.9486 at the tile centre).
    scale = np.cos(np.radians(config["tile"]["center_lat"]))
    assert record["east_m"] == pytest.approx(record["map_m"]["east"] * scale, abs=0.01)
    assert record["north_m"] == record["map_m"]["north"]
    write_up = REPO / record["write_up"]
    assert write_up.is_file()
    # The write-up must quote the very numbers the config carries, as the config's own keys
    # (a bare number could match some other figure in the text).
    text = write_up.read_text(encoding="utf-8")
    assert f"`east_m: {record['east_m']:+.2f}`" in text and f"`north_m: {record['north_m']:+.2f}`" in text
