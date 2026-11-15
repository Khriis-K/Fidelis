"""The pre-registered multifidelity go/no-go with the sun varying (#48, docs/benchmarks/sun-sweep.md),
on synthetic records whose answer is known."""

import math
from dataclasses import replace

import numpy as np
import pytest
import yaml

from fidelis.fidelity_sweep import variance_ratio
from fidelis.simulation import SimulationRecord
from fidelis.sun_sweep import (
    HOLDOUT,
    SELECTION,
    SUNS,
    Formula,
    apply_rule,
    cv_ratio,
    focal_shift,
    horizontal_error,
    stratified_ratio,
)

REAL_COST = 0.50
MAP = 58866
SUN_OFFSETS = [(1.0, 0.5), (-2.0, 1.0), (0.5, -3.0), (3.0, 2.0), (-1.0, -1.0)]


def scenario(rng, seed):
    return {
        "sun_index": seed % SUNS,
        "dust_tau": rng.uniform(0.5, 1.0),
        "attitude_error_x_deg": rng.uniform(-0.5, 0.5),
        "focal_error": rng.uniform(-0.01, 0.01),
        "altitude_m": rng.uniform(1500, 3000),
        "tilt_deg": rng.uniform(0, 18),
        "tilt_azimuth_deg": rng.uniform(0, 360),
    }


def geometric_error(sc, sun_scale, focal_scale):
    """A seed's east-north error: a per-sun offset plus the focal shift back from the tilt direction."""
    east, north = SUN_OFFSETS[sc["sun_index"]]
    shift = focal_scale * focal_shift(sc)
    az = math.radians(sc["tilt_azimuth_deg"])
    return sun_scale * east + shift * math.sin(az), sun_scale * north + shift * math.cos(az)


def record(seed, fidelity, east, north, cost_s, map_keypoints, sc, status="ok", gpu="NVIDIA A40"):
    ok = status == "ok"
    offset = [1.89, -1.49]
    vector = [east + offset[0], north + offset[1], 5.0] if ok else None
    return SimulationRecord(
        seed=seed,
        fidelity=fidelity,
        status=status,
        true_position=[0.0, 0.0, 0.0],
        estimated_position=vector,
        error_vector=vector,
        error_m=math.hypot(*vector) if ok else None,
        match_count=100,
        inlier_count=50,
        cost_s=cost_s,
        gpu=gpu,
        scenario=sc,
        map_keypoints=map_keypoints,
        map_offset_m=offset,
        map_offset_source="test",
        map_frame_error_vector=[east, north, 5.0] if ok else None,
    )


def sweep(levels, seeds=None, sun_scale=1.0, focal_scale=1.0, nav_sd=1.0, rerun_sd=0.05, base_seed=0, gpus=None):
    """A sweep where every level shares the seed's geometric error plus the real navigator's own scatter.

    `levels`: {name: (extra_sd, cost_s)}, the level's scatter on top of the real one; "mesh" and
    "mesh-rerun" are added. A level's error tracks the real one exactly as far as its extra_sd allows.
    `gpus`: {name: gpu} to put one level on a different GPU.
    """
    seeds = seeds if seeds is not None else list(SELECTION) + list(HOLDOUT)
    rng = np.random.default_rng(base_seed)
    scenarios = {s: scenario(rng, s) for s in seeds}
    real = {}
    for s in seeds:
        east, north = geometric_error(scenarios[s], sun_scale, focal_scale)
        real[s] = (east + rng.normal(0, nav_sd), north + rng.normal(0, nav_sd))
    gpus = gpus or {}
    out = {}
    for name, (sd, cost) in {"mesh": (0.0, REAL_COST), "mesh-rerun": (rerun_sd, REAL_COST), **levels}.items():
        k = MAP if name in ("mesh", "mesh-rerun") else int(name.split("-k")[1])
        fidelity = "mesh" if name == "mesh-rerun" else name
        out[name] = {
            s: record(
                s,
                fidelity,
                real[s][0] + rng.normal(0, sd),
                real[s][1] + rng.normal(0, sd),
                cost * (1 + 0.01 * rng.normal()),
                k,
                scenarios[s],
                gpu=gpus.get(name, "NVIDIA A40"),
            )
            for s in seeds
        }
    return out


def formula_from(seeds, **kw):
    """A formula fitted on its own seeds, as the real one is fitted on the combined scan's."""
    return Formula.from_records(list(sweep({}, seeds=seeds, base_seed=99, **kw)["mesh"].values()))


# The pieces


def test_the_error_is_horizontal_and_in_the_map_frame():
    sc = {"sun_index": 0, "focal_error": 0.0, "altitude_m": 2000.0, "tilt_deg": 0.0, "tilt_azimuth_deg": 0.0}
    r = record(0, "mesh", 3.0, 4.0, REAL_COST, MAP, sc)  # stored with the map offset added, and 5 m up
    assert horizontal_error(r) == pytest.approx(5.0)


def test_the_config_draws_the_suns_this_rule_weights_equally():
    # Stratified MC's weights are 1/SUNS each, which holds only while the config draws the suns uniformly.
    config = yaml.safe_load(open("config/jezero.yaml", encoding="utf-8"))
    assert config["scenario"]["sun_index"] == {"dist": "integers", "low": 0, "high": SUNS - 1}
    assert len(config["suns"]) == SUNS


def test_stratifying_on_a_per_sun_constant_removes_all_variance():
    strata = np.arange(100) % SUNS
    errors = np.array([2.0, 5.0, 1.0, 7.0, 3.0])[strata]
    assert stratified_ratio(errors, strata) == pytest.approx(0.0)


def test_stratifying_on_a_sun_the_error_ignores_saves_nothing():
    # The null case: strata that know nothing about the error leave plain MC's variance.
    rng = np.random.default_rng(1)
    strata = np.arange(20000) % SUNS
    assert stratified_ratio(rng.normal(size=strata.size), strata) == pytest.approx(1.0, abs=0.03)


def test_stratifying_needs_two_descents_in_every_sun():
    strata = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4])
    with pytest.raises(ValueError, match="sun 4"):
        stratified_ratio(np.arange(9.0), strata)


def test_a_control_variate_saves_one_minus_rho_squared():
    rng = np.random.default_rng(2)
    x = rng.normal(size=5000)
    y = 0.8 * x + 0.6 * rng.normal(size=5000)
    rho = np.corrcoef(x, y)[0, 1]
    assert cv_ratio(y, x) == pytest.approx(1 - rho**2)


def test_a_constant_control_variate_saves_nothing():
    # The null case from CLAUDE.md: a predictor that outputs a constant can't reduce variance.
    assert cv_ratio(np.arange(10.0), np.full(10, 3.0)) == 1.0


def test_the_formula_recovers_the_focal_geometry_on_fresh_seeds():
    formula = formula_from(range(300), nav_sd=0.01)
    fresh = sweep({}, nav_sd=0.01)["mesh"]
    predicted = [formula.predict(r) for r in fresh.values()]
    actual = [horizontal_error(r) for r in fresh.values()]
    assert np.corrcoef(predicted, actual)[0, 1] > 0.99


def test_a_formula_fitted_on_unrelated_errors_predicts_nothing_on_fresh_seeds():
    # Null case: when the error has no geometric part, the fitted formula is noise.
    formula = formula_from(range(300), sun_scale=0.0, focal_scale=0.0)
    fresh = sweep({}, sun_scale=0.0, focal_scale=0.0)["mesh"]
    predicted = np.array([formula.predict(r) for r in fresh.values()])
    actual = np.array([horizontal_error(r) for r in fresh.values()])
    assert cv_ratio(actual, predicted) > 0.9


def test_acv_matches_the_two_model_formula():
    rho, w = 0.9, 0.2
    cov = np.array([[1.0, rho], [rho, 1.0]])
    expected = (math.sqrt(1 - rho**2) + math.sqrt(w * rho**2)) ** 2
    assert variance_ratio(np.array([1.0, w]), cov, method="acvmf") == pytest.approx(expected, rel=0.02)


# The rule


def test_a_cheap_level_that_tracks_the_navigators_own_scatter_is_a_go():
    # No geometric part, so the sun and the formula explain nothing; a cheap level that follows
    # the real navigator seed by seed is then the only thing that helps.
    data = sweep({"mesh-k8000": (0.1, 0.05), "mesh-k1000": (3.0, 0.04)}, sun_scale=0.0, focal_scale=0.0)
    report = apply_rule(data, formula_from(range(300), sun_scale=0.0, focal_scale=0.0))
    assert report.gates_passed
    assert report.chosen == ("mesh-k8000",)
    assert report.chosen_ratio < report.best_baseline_ratio
    assert report.go


def test_a_level_that_only_shares_the_geometry_is_a_no_go():
    # The error is mostly focal geometry, which the free formula already knows. A cheap level at
    # half the real cost tracks it too, but can't beat a predictor that costs nothing.
    data = sweep({"mesh-k32000": (0.3, 0.25)}, nav_sd=0.5)
    report = apply_rule(data, formula_from(range(300), nav_sd=0.5))
    assert report.gates_passed
    assert report.cv_ratio < 0.2
    assert report.chosen_ratio > report.cv_ratio
    assert not report.go


def test_the_sun_share_is_reported_next_to_the_levels():
    data = sweep({"mesh-k8000": (0.5, 0.1)}, focal_scale=0.0)
    report = apply_rule(data, formula_from(range(300), focal_scale=0.0))
    assert report.sun_share == pytest.approx(1 - report.stratified_ratio)
    assert report.sun_share > 0.3


def test_a_jittery_real_navigator_stops_the_rule():
    data = sweep({"mesh-k8000": (0.1, 0.05)}, rerun_sd=5.0)
    report = apply_rule(data, formula_from(range(300)))
    assert not report.gates_passed
    assert report.chosen is None and not report.go


def test_levels_on_different_gpus_stop_the_rule():
    # Costs from mixed GPU models are meaningless (CLAUDE.md, "Pin the GPU type").
    data = sweep({"mesh-k8000": (0.1, 0.05)}, gpus={"mesh-k8000": "NVIDIA L40S"})
    report = apply_rule(data, formula_from(range(300)))
    assert any("GPU" in f for f in report.gate_failures)


def test_a_level_run_with_the_sourced_errors_off_stops_the_rule():
    # --exact-camera etc. leave the config SHA unchanged, so the rule checks the draws themselves.
    data = sweep({"mesh-k8000": (0.1, 0.05)})
    data["mesh-k8000"] = {
        s: replace(r, scenario={**r.scenario, "focal_error": 0.0}) for s, r in data["mesh-k8000"].items()
    }
    report = apply_rule(data, formula_from(range(300)))
    assert any("mesh-k8000: focal error is off" in f for f in report.gate_failures)
