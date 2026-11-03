import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import stats

from fidelis.scenario import (
    FIELDS,
    MC_SEED_LIMIT,
    Integers,
    ScenarioConfig,
    TruncatedNormal,
    Uniform,
    check_fits_tile,
    draw,
    load_config,
    max_ground_reach_m,
    parse_distribution,
    pin_sun,
)

REPO = Path(__file__).resolve().parents[1]
N = 3000  # draws per distribution test; with fixed seeds the tests are deterministic


def make_config(**overrides) -> ScenarioConfig:
    dists = {
        "offset_east_m": TruncatedNormal(mean=0.0, sd=300.0, low=-900.0, high=900.0),
        "offset_north_m": TruncatedNormal(mean=50.0, sd=200.0, low=-500.0, high=700.0),
        "altitude_m": Uniform(2200.0, 2600.0),
        "tilt_deg": TruncatedNormal(mean=0.0, sd=3.0, low=0.0, high=10.0),
        "tilt_azimuth_deg": Uniform(0.0, 360.0),
        "heading_deg": Uniform(0.0, 360.0),
        "noise_sd": Uniform(0.0, 0.02),
        "blur_px": Uniform(0.0, 2.0),
        "blur_angle_deg": Uniform(0.0, 180.0),
        "exposure_gain": TruncatedNormal(mean=1.0, sd=0.1, low=0.7, high=1.3),
        "sun_index": Integers(0, 4),
    }
    dists.update(overrides)
    return ScenarioConfig(**dists)


def scipy_dist(d):
    if isinstance(d, Uniform):
        return stats.uniform(loc=d.low, scale=d.high - d.low)
    a, b = (d.low - d.mean) / d.sd, (d.high - d.mean) / d.sd
    return stats.truncnorm(a, b, loc=d.mean, scale=d.sd)


# --- reproducibility ------------------------------------------------------------------


def test_the_same_seed_always_gives_the_same_scenario():
    config = make_config()
    assert draw(config, 12345) == draw(config, 12345)


def test_a_scenario_records_its_seed():
    assert draw(make_config(), 98765).seed == 98765


def test_different_seeds_give_different_scenarios():
    config = make_config()
    a, b = draw(config, 1), draw(config, 2)
    for name in FIELDS:
        if name != "sun_index":  # five values, so two seeds may well agree
            assert getattr(a, name) != getattr(b, name), name


def test_changing_one_inputs_distribution_leaves_the_other_draws_unchanged():
    # Each input has its own random stream, so retuning altitude can't silently
    # reshuffle every other input of an existing seed.
    before = draw(make_config(), 7)
    after = draw(make_config(altitude_m=Uniform(2300.0, 2400.0)), 7)

    assert after.altitude_m != before.altitude_m
    for name in FIELDS:
        if name != "altitude_m":
            assert getattr(after, name) == getattr(before, name), name


def test_a_scenario_round_trips_through_json_as_plain_numbers():
    s = draw(make_config(), 3)
    record = json.loads(json.dumps(asdict(s)))
    assert record == asdict(s)
    assert type(record["sun_index"]) is int and type(record["seed"]) is int


def test_negative_seeds_are_refused():
    with pytest.raises(ValueError, match="seed"):
        draw(make_config(), -1)


# --- distributions ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def many():
    config = make_config()
    return config, [draw(config, seed) for seed in range(N)]


@pytest.mark.parametrize("name", [f for f in FIELDS if f != "sun_index"])
def test_each_continuous_input_follows_its_configured_distribution(many, name):
    config, samples = many
    values = [getattr(s, name) for s in samples]
    result = stats.kstest(values, scipy_dist(getattr(config, name)).cdf)
    assert result.pvalue > 1e-3, f"{name}: KS statistic {result.statistic:.4f}"


def test_a_truncated_normal_is_not_mistaken_for_a_uniform(many):
    # Guards the KS test itself: it must be able to tell two distributions apart.
    config, samples = many
    values = [s.offset_east_m for s in samples]
    d = config.offset_east_m
    assert stats.kstest(values, stats.uniform(loc=d.low, scale=d.high - d.low).cdf).pvalue < 1e-6


def test_the_sun_index_is_uniform_over_its_range(many):
    config, samples = many
    counts = np.bincount([s.sun_index for s in samples], minlength=5)
    assert len(counts) == 5  # never outside 0..4
    assert stats.chisquare(counts).pvalue > 1e-3


def test_every_draw_stays_within_its_bounds(many):
    config, samples = many
    for name in FIELDS:
        low, high = getattr(config, name).bounds()
        values = [getattr(s, name) for s in samples]
        assert low <= min(values) and max(values) <= high, name


def test_inputs_are_drawn_independently_of_each_other(many):
    _, samples = many
    columns = np.array([[getattr(s, name) for name in FIELDS] for s in samples], dtype=float)
    corr = np.corrcoef(columns, rowvar=False)
    off_diagonal = corr[~np.eye(len(FIELDS), dtype=bool)]
    assert np.max(np.abs(off_diagonal)) < 0.1


# --- parsing the config ----------------------------------------------------------------


def test_distributions_parse_from_their_config_form():
    assert parse_distribution({"dist": "uniform", "low": 1, "high": 2}) == Uniform(1.0, 2.0)
    assert parse_distribution({"dist": "truncnorm", "mean": 0, "sd": 1, "low": -2, "high": 3}) == TruncatedNormal(
        0.0, 1.0, -2.0, 3.0
    )
    assert parse_distribution({"dist": "integers", "low": 0, "high": 5}) == Integers(0, 5)


@pytest.mark.parametrize(
    "spec",
    [
        {"dist": "uniform", "low": 2, "high": 1},
        {"dist": "truncnorm", "mean": 0, "sd": 0, "low": -1, "high": 1},
        {"dist": "truncnorm", "mean": 0, "sd": 1, "low": 1, "high": -1},
        {"dist": "integers", "low": 3, "high": 2},
        {"dist": "gamma", "low": 0, "high": 1},
        {"dist": "uniform", "low": 0},
    ],
)
def test_malformed_distributions_are_refused(spec):
    with pytest.raises(ValueError):
        parse_distribution(spec)


def test_a_config_missing_an_input_is_refused():
    section = {name: {"dist": "uniform", "low": 0, "high": 1} for name in FIELDS if name != "blur_px"}
    with pytest.raises(ValueError, match="blur_px"):
        ScenarioConfig.from_dict(section)


def test_a_config_with_an_unknown_input_is_refused():
    section = {name: {"dist": "uniform", "low": 0, "high": 1} for name in FIELDS}
    section["wind_mps"] = {"dist": "uniform", "low": 0, "high": 1}
    with pytest.raises(ValueError, match="wind_mps"):
        ScenarioConfig.from_dict(section)


# --- staying on the tile ---------------------------------------------------------------


def test_a_nadir_90_degree_camera_reaches_its_footprint_corner():
    # Straight down, the farthest ground a 90 x 90 deg view sees is its corner, at
    # altitude * sqrt(2) horizontally.
    assert max_ground_reach_m(altitude_m=2000.0, tilt_deg=0.0, fov_deg=90.0) == pytest.approx(2000 * math.sqrt(2))


def test_tilting_the_camera_reaches_farther():
    level = max_ground_reach_m(altitude_m=2000.0, tilt_deg=0.0, fov_deg=90.0)
    tilted = max_ground_reach_m(altitude_m=2000.0, tilt_deg=10.0, fov_deg=90.0)
    assert tilted == pytest.approx(2000 * math.tan(math.atan(math.sqrt(2)) + math.radians(10)))
    assert tilted > level


def test_a_view_that_reaches_the_horizon_has_no_finite_reach():
    with pytest.raises(ValueError, match="horizon"):
        max_ground_reach_m(altitude_m=2000.0, tilt_deg=40.0, fov_deg=90.0)


def test_a_config_whose_worst_case_fits_the_tile_passes():
    # Worst case: 2600 m + 259 m relief, 10 deg tilt -> corner reach 6058 m; plus a 900 m
    # offset east (6958 m) and 700 m north (6758 m).
    check_fits_tile(make_config(), half_width_east_m=7000.0, half_width_north_m=7000.0, fov_deg=90.0, relief_m=259.0)


def test_a_config_that_can_see_past_the_east_edge_is_refused():
    with pytest.raises(ValueError, match="east"):
        check_fits_tile(
            make_config(), half_width_east_m=6900.0, half_width_north_m=7000.0, fov_deg=90.0, relief_m=259.0
        )


def test_a_large_negative_offset_counts_against_the_tile():
    # 1500 m west plus 6058 m of reach passes the 7000 m west edge.
    config = make_config(offset_east_m=Uniform(-1500.0, 100.0))
    with pytest.raises(ValueError, match="east"):
        check_fits_tile(config, half_width_east_m=7000.0, half_width_north_m=7000.0, fov_deg=90.0, relief_m=259.0)


def test_terrain_relief_counts_against_the_tile():
    # Ground lower than the spot under the camera widens the view; without the relief
    # term this config would fit (east reach 6409 m instead of 6958 m).
    check_fits_tile(make_config(), half_width_east_m=6600.0, half_width_north_m=7000.0, fov_deg=90.0, relief_m=0.0)
    with pytest.raises(ValueError):
        check_fits_tile(
            make_config(), half_width_east_m=6600.0, half_width_north_m=7000.0, fov_deg=90.0, relief_m=259.0
        )


@pytest.fixture(scope="module")
def jezero():
    config = load_config(REPO / "config" / "jezero.yaml")
    return config, [draw(config, seed) for seed in range(N)]


def test_the_jezero_config_loads_and_every_draw_stays_on_the_tile(jezero):
    # Each draw's own reach, not just the config's worst case, against the scene's
    # half-widths (the CTX-ringed scene since #33; east-west scaled to ground by cos(latitude)).
    _, samples = jezero
    scene = yaml.safe_load((REPO / "config" / "jezero.yaml").read_text(encoding="utf-8"))
    half_north = scene["scene"]["size_m"] / 2
    half_east = half_north * math.cos(math.radians(scene["tile"]["center_lat"]))
    relief, fov = scene["scenario"]["terrain_relief_m"], scene["scenario"]["camera_fov_deg"]
    for s in samples:
        reach = max_ground_reach_m(s.altitude_m + relief, s.tilt_deg, fov)
        assert abs(s.offset_east_m) + reach <= half_east
        assert abs(s.offset_north_m) + reach <= half_north


@pytest.mark.parametrize("name", [f for f in FIELDS if f != "sun_index"])
def test_each_jezero_input_follows_its_configured_distribution(jezero, name):
    config, samples = jezero
    values = [getattr(s, name) for s in samples]
    assert stats.kstest(values, scipy_dist(getattr(config, name)).cdf).pvalue > 1e-3, name


def test_a_seed_gives_the_same_scenario_across_library_versions():
    # Pinned values, not just a repeat within one process: a NumPy update that changed
    # its random streams would otherwise quietly change every recorded descent.
    expected = {
        "offset_east_m": 376.63344732215114,
        "offset_north_m": 32.21027351543845,
        "altitude_m": 2521.111338107608,
        "tilt_deg": 7.470293128595389,
        "tilt_azimuth_deg": 301.249454471735,
        "heading_deg": 224.00240995535933,
        "noise_sd": 0.010867478677705513,
        "blur_px": 0.9237440734229438,
        "blur_angle_deg": 155.68170013876778,
        "exposure_gain": 1.0429697251747545,
        "sun_index": 0,
    }
    s = draw(make_config(), 0)
    for name, value in expected.items():
        assert getattr(s, name) == pytest.approx(value, rel=1e-12), name


def test_a_truncated_normal_bounded_deep_in_its_tail_is_refused():
    # cdf(9 sd) rounds to 1, which would crash every draw; refuse it up front.
    with pytest.raises(ValueError, match="tails"):
        TruncatedNormal(mean=0.0, sd=1.0, low=-1.0, high=9.0)


def test_a_non_integer_sun_index_is_refused():
    section = {name: {"dist": "uniform", "low": 0, "high": 1} for name in FIELDS}
    with pytest.raises(ValueError, match="sun_index"):
        ScenarioConfig.from_dict(section)


def test_loading_a_config_that_can_leave_the_tile_is_refused(tmp_path):
    scene = yaml.safe_load((REPO / "config" / "jezero.yaml").read_text(encoding="utf-8"))
    scene["scenario"]["tilt_deg"] = {"dist": "uniform", "low": 0, "high": 20}
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(scene), encoding="utf-8")
    with pytest.raises(ValueError, match="past the tile"):
        load_config(path)


# --- no GPU ----------------------------------------------------------------------------


def test_the_sampler_does_not_import_torch():
    code = "import sys, fidelis.scenario; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_the_jezero_config_reaches_flight_like_tilts_and_offsets():
    # #33: the CTX-ringed scene lifts the tile clip. Flight pitch swung to about -12 deg and
    # pre-flight simulations averaged 7-18 deg; position sigma is ~1.1 km.
    config = load_config(REPO / "config" / "jezero.yaml")
    assert config.tilt_deg.bounds()[1] >= 15
    for offset in (config.offset_east_m, config.offset_north_m):
        assert min(abs(b) for b in offset.bounds()) >= 2 * offset.sd


def test_offsets_that_can_take_the_camera_off_the_hirise_tile_are_refused(tmp_path):
    # terrain_relief_m is measured from the tile's highest ground. At nadir a 5 km offset's
    # views still fit the wide scene, so only the camera-over-the-tile check catches it.
    scene = yaml.safe_load((REPO / "config" / "jezero.yaml").read_text(encoding="utf-8"))
    scene["scenario"]["tilt_deg"] = {"dist": "uniform", "low": 0, "high": 0}
    scene["scenario"]["offset_east_m"] = {"dist": "truncnorm", "mean": 0, "sd": 1100, "low": -5000, "high": 5000}
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(scene), encoding="utf-8")
    with pytest.raises(ValueError, match="HiRISE tile"):
        load_config(path)


# --- pinning the sun: while only some suns have a splat scene (#9 trains one), a run can
# hold sun_index at one value without changing anything else a seed draws


CONFIG = load_config(REPO / "config" / "jezero.yaml")


@pytest.mark.parametrize("sun_index", [0, 3])
def test_every_seed_draws_the_pinned_sun(sun_index):
    pinned = pin_sun(CONFIG, sun_index)
    assert {draw(pinned, seed).sun_index for seed in range(200)} == {sun_index}


def test_pinning_the_sun_leaves_every_other_input_of_a_seed_unchanged():
    pinned = pin_sun(CONFIG, 0)
    for seed in range(50):
        free, held = asdict(draw(CONFIG, seed)), asdict(draw(pinned, seed))
        assert {k: free[k] for k in FIELDS if k != "sun_index"} == {k: held[k] for k in FIELDS if k != "sun_index"}


@pytest.mark.parametrize("sun_index", [-1, 5])
def test_a_sun_the_config_cannot_draw_is_refused(sun_index):
    with pytest.raises(ValueError, match="sun"):
        pin_sun(CONFIG, sun_index)


def test_monte_carlo_seeds_and_splat_view_seeds_meet_at_one_limit():
    from fidelis.splat.views import view_seeds

    assert min(view_seeds("train", 1)) == MC_SEED_LIMIT
