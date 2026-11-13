"""The stress case (#43): suns beyond the realistic set, the texture a run renders from, and the labels that
keep stress records out of realistic estimates (docs/benchmarks/stress-sun-scan.md, ADR 0003 decision 1)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from rasterio.transform import from_origin

from fidelis.nav.navigator import Navigator
from fidelis.nav.orbital_map import OrbitalMap
from fidelis.plain_mc import estimate
from fidelis.render.camera import Intrinsics
from fidelis.render.mesh_render import Sun, config_suns
from fidelis.scenario import Scenario, draw, load_config
from fidelis.simulation import SimulationRecord, Simulator
from fidelis.stress import (
    CUSTOM_SUNS,
    DESHADED_TEXTURE,
    PHOTO_TEXTURE,
    REALISTIC_SUNS,
    STRESS_SUNS,
    check_pinned_sun,
    check_sun_list,
    ctx_map_sun,
    is_stress_case,
    pin_stress_sun,
    pinned_sun,
    pool_key,
    stress_suns,
    sun_difference_deg,
    suns_for,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "config" / "jezero.yaml"
SCENE = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
CONFIG = load_config(CONFIG_PATH)


# --- the angle between two suns


def test_a_sun_is_zero_degrees_from_itself():
    assert sun_difference_deg(Sun(123.0, 37.0), Sun(123.0, 37.0)) == pytest.approx(0.0, abs=1e-6)


def test_suns_on_opposite_horizons_are_180_degrees_apart():
    assert sun_difference_deg(Sun(0.0, 0.0), Sun(180.0, 0.0)) == pytest.approx(180.0)


def test_the_zenith_is_90_degrees_from_any_sun_on_the_horizon():
    assert sun_difference_deg(Sun(0.0, 90.0), Sun(250.0, 0.0)) == pytest.approx(90.0)


def test_bearing_counts_for_less_the_higher_the_suns_are():
    # Half a turn in bearing at the same elevation e is 180 - 2e degrees, not 180.
    assert sun_difference_deg(Sun(10.0, 30.0), Sun(190.0, 30.0)) == pytest.approx(120.0)


def test_it_reproduces_the_research_notes_largest_realistic_difference():
    # docs/research/ctx-and-hirise-sun-angles.md, section 6: the 310/25 sun is 39.0 deg from the map's.
    assert sun_difference_deg(Sun(310.0, 25.0), ctx_map_sun(SCENE)) == pytest.approx(39.0, abs=0.05)


# --- the stress suns in the scene config


def test_the_map_sun_is_the_ctx_image_over_the_tile_from_the_research_note():
    sun = ctx_map_sun(SCENE)
    assert (sun.azimuth_deg, sun.elevation_deg) == (267.37, 42.42)


def test_each_stress_suns_stated_difference_is_its_computed_one():
    for entry, sun in zip(SCENE[STRESS_SUNS], stress_suns(SCENE), strict=True):
        assert entry["sun_difference_deg"] == pytest.approx(sun_difference_deg(sun, ctx_map_sun(SCENE)), abs=0.05)


def test_the_stress_suns_are_the_pre_registered_dose_steps():
    # docs/benchmarks/stress-sun-scan.md: one elevation, and the dose rises across the 100 deg line.
    differences = [e["sun_difference_deg"] for e in SCENE[STRESS_SUNS]]
    assert differences == [73.4, 98.3, 112.6]
    assert {s.elevation_deg for s in stress_suns(SCENE)} == {25.0}


def test_no_stress_sun_is_in_the_realistic_list():
    realistic = [tuple(s.direction()) for s in config_suns(SCENE)]
    for sun in stress_suns(SCENE):
        assert not any(np.allclose(sun.direction(), r) for r in realistic)


def test_the_realistic_sampler_never_draws_past_the_realistic_suns():
    drawn = {draw(CONFIG, seed).sun_index for seed in range(2000)}
    assert drawn == set(range(len(SCENE["suns"])))


def test_a_stress_run_renders_from_the_stress_list_and_a_realistic_run_from_the_realistic_one():
    assert suns_for(SCENE, STRESS_SUNS) == stress_suns(SCENE)
    assert suns_for(SCENE, "suns") == config_suns(SCENE)
    with pytest.raises(ValueError, match="sun list"):
        suns_for(SCENE, "moons")


# --- pinning a stress sun


def test_a_pinned_stress_sun_lights_every_draw_and_leaves_the_seeds_other_inputs_alone():
    pinned = pin_stress_sun(CONFIG, 2, SCENE)
    for seed in (40_000, 40_001, 40_299):
        realistic, stressed = draw(CONFIG, seed), draw(pinned, seed)
        assert stressed.sun_index == 2
        assert replace(stressed, sun_index=realistic.sun_index) == realistic


@pytest.mark.parametrize("index", [-1, 3])
def test_a_stress_sun_past_the_list_is_refused(index):
    with pytest.raises(ValueError, match="stress sun"):
        pin_stress_sun(CONFIG, index, SCENE)


# --- what makes a record a stress case


def record(seed=0, error_m=1.0, sun_index=0, sun_list="suns", texture=PHOTO_TEXTURE, **overrides):
    error = [error_m, 0.0, 0.0]
    values = dict(
        seed=seed,
        fidelity="mesh",
        status="ok",
        true_position=[0.0, 0.0, 0.0],
        estimated_position=error,
        error_vector=error,
        error_m=error_m,
        match_count=100,
        inlier_count=80,
        cost_s=1.0,
        gpu="test GPU",
        scenario={"sun_index": sun_index},
        map_offset_m=[0.0, 0.0],
        map_offset_source="test",
        map_frame_error_vector=error,
        map_frame_error_m=error_m,
        sun_list=sun_list,
        texture=texture,
    )
    return SimulationRecord(**(values | overrides))


def test_a_realistic_sun_on_the_photo_texture_is_not_a_stress_case():
    assert not is_stress_case(record())


def test_a_record_from_before_the_labels_is_not_a_stress_case():
    # Every record before #43 used the realistic suns and the photo texture.
    assert not is_stress_case(record(sun_list=None, texture=None))


def test_a_stress_sun_is_a_stress_case_even_on_the_photo_texture():
    assert is_stress_case(record(sun_list=STRESS_SUNS, texture=PHOTO_TEXTURE))


def test_a_realistic_sun_on_the_deshaded_texture_is_a_stress_case():
    # The plan's anchor: a realistic sun, but not the scene the realistic rate is measured on.
    assert is_stress_case(record(sun_list="suns", texture=DESHADED_TEXTURE))


# --- estimates never pool stress with realistic records


def test_an_estimate_refuses_to_pool_stress_and_realistic_records():
    with pytest.raises(ValueError, match="stress"):
        estimate([record(seed=1), record(seed=2, sun_list=STRESS_SUNS)], r_m=40.0)


def test_an_estimate_refuses_to_pool_two_textures():
    with pytest.raises(ValueError, match="stress"):
        estimate([record(seed=1, texture=DESHADED_TEXTURE), record(seed=2, texture=PHOTO_TEXTURE)], r_m=40.0)


def test_an_estimate_refuses_to_pool_two_stress_suns():
    # Each dose step is its own estimate; pooled, they'd be a rate of nothing.
    stress = dict(sun_list=STRESS_SUNS, texture=DESHADED_TEXTURE)
    with pytest.raises(ValueError, match="stress"):
        estimate([record(seed=1, sun_index=0, **stress), record(seed=2, sun_index=2, **stress)], r_m=40.0)


def test_realistic_records_from_before_and_after_the_labels_still_pool_across_suns():
    pooled = estimate([record(seed=1, sun_list=None, texture=None, sun_index=0), record(seed=2, sun_index=3)], 40.0)
    assert pooled.n == 2
    assert not pooled.stress_case


def test_a_stress_estimate_is_labelled_as_one_in_its_fields_and_its_summary():
    stress = dict(sun_list=STRESS_SUNS, texture=DESHADED_TEXTURE, sun_index=2)
    est = estimate([record(seed=1, **stress), record(seed=2, error_m=50.0, **stress)], r_m=40.0)
    assert est.stress_case
    assert (est.sun_list, est.texture) == (STRESS_SUNS, DESHADED_TEXTURE)
    assert est.summary().startswith("STRESS CASE")
    assert est.failures == 1


def test_a_realistic_summary_carries_no_stress_label():
    assert "STRESS" not in estimate([record()], r_m=40.0).summary()


# --- the simulator labels what it renders


LEFT, TOP, RES, SIZE = 4589000.0, 1094000.0, 2.0, 128
TINY_MAP = OrbitalMap(
    image=np.random.default_rng(0).random((SIZE, SIZE), np.float32),
    transform=from_origin(LEFT, TOP, RES, RES),
    dtm=np.full((4, 4), -2500.0, np.float32),
    dtm_transform=from_origin(LEFT, TOP, RES * SIZE / 4, RES * SIZE / 4),
    origin=(LEFT + RES * SIZE / 2, TOP - RES * SIZE / 2),
    east_scale=1.0,
)
K = Intrinsics.from_fov(64, 64, 90)


def render_black(pose, intrinsics, sun_index, seed):
    return np.zeros((intrinsics.height, intrinsics.width), np.float32)


def scenario(sun_index=0):
    return Scenario(
        seed=40_000,
        offset_east_m=0.0,
        offset_north_m=0.0,
        altitude_m=100.0,
        tilt_deg=0.0,
        tilt_azimuth_deg=0.0,
        heading_deg=0.0,
        noise_sd=0.0,
        blur_px=0.0,
        blur_angle_deg=0.0,
        exposure_gain=1.0,
        sun_index=sun_index,
    )


@pytest.fixture(scope="module")
def navigator():
    return Navigator(TINY_MAP, device="cpu")


def test_a_default_simulator_labels_its_records_realistic(navigator):
    rec = Simulator({"black": render_black}, navigator, K).run(scenario(), "black").record
    assert (rec.sun_list, rec.texture) == ("suns", PHOTO_TEXTURE)
    assert not is_stress_case(rec)


def test_a_stress_simulator_labels_even_a_no_fix_record(navigator):
    sim = Simulator({"black": render_black}, navigator, K, sun_list=STRESS_SUNS, texture=DESHADED_TEXTURE)
    rec = sim.run(scenario(sun_index=2), "black").record
    assert rec.status != "ok"
    assert (rec.sun_list, rec.texture, rec.scenario["sun_index"]) == (STRESS_SUNS, DESHADED_TEXTURE, 2)
    assert is_stress_case(rec)


def test_an_unknown_sun_list_or_texture_is_a_caller_error(navigator):
    with pytest.raises(ValueError, match="sun list"):
        Simulator({"black": render_black}, navigator, K, sun_list="moons")
    with pytest.raises(ValueError, match="texture"):
        Simulator({"black": render_black}, navigator, K, texture="texture_blurred.png")


# --- after the code review (#43): custom suns, a pinned sun for off-photo runs, the pinned sun's angle


def test_custom_suns_can_label_a_simulator_but_never_come_from_the_config():
    check_sun_list(CUSTOM_SUNS)
    with pytest.raises(ValueError, match="sun list"):
        suns_for(SCENE, CUSTOM_SUNS)


def test_custom_suns_are_never_counted_as_realistic():
    # A test's zenith sun, say: not one of the config's realistic suns.
    assert is_stress_case(record(sun_list=CUSTOM_SUNS))


def test_realistic_records_share_a_pool_key_across_suns_and_stress_records_only_within_one():
    assert pool_key(record(sun_index=0)) == pool_key(record(sun_index=3, sun_list=None, texture=None))
    stress = dict(sun_list=STRESS_SUNS, texture=DESHADED_TEXTURE)
    assert pool_key(record(sun_index=0, **stress)) != pool_key(record(sun_index=2, **stress))


def test_a_run_on_the_deshaded_texture_without_a_pinned_sun_is_refused_before_it_starts():
    # Unpinned, its draws span five suns, which estimate() won't pool: it would render everything, then fail.
    with pytest.raises(ValueError, match="pin"):
        check_pinned_sun(DESHADED_TEXTURE, sun_index=None, stress_sun=None)


@pytest.mark.parametrize(("sun_index", "stress_sun"), [(3, None), (None, 2)])
def test_a_run_on_the_deshaded_texture_with_a_pinned_sun_is_allowed(sun_index, stress_sun):
    check_pinned_sun(DESHADED_TEXTURE, sun_index=sun_index, stress_sun=stress_sun)


def test_a_realistic_run_on_the_photo_still_draws_its_own_suns():
    check_pinned_sun(PHOTO_TEXTURE, sun_index=None, stress_sun=None)


def test_the_anchor_records_its_realistic_sun_and_its_39_degrees():
    assert pinned_sun(SCENE, sun_index=3, stress_sun=None) == {
        "sun_list": REALISTIC_SUNS,
        "index": 3,
        "azimuth_deg": 310,
        "elevation_deg": 25,
        "sun_difference_deg": 39.0,
    }


def test_a_stress_sun_records_its_computed_difference():
    sun = pinned_sun(SCENE, sun_index=None, stress_sun=2)
    assert (sun["sun_list"], sun["index"], sun["azimuth_deg"]) == (STRESS_SUNS, 2, 87.37)
    assert sun["sun_difference_deg"] == 112.6


def test_an_unpinned_run_records_no_single_sun():
    assert pinned_sun(SCENE, sun_index=None, stress_sun=None) is None
