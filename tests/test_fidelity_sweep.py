"""The pre-registered rule for choosing the cheaper fidelity levels (docs/benchmarks/fidelity-sweep.md),
on synthetic records whose answer is known."""

import math

import numpy as np
import pytest

from fidelis.fidelity_sweep import SELECTION, apply_rule, variance_ratio
from fidelis.simulation import SimulationRecord

REAL_COST = 0.30


def record(seed, fidelity, error_m, cost_s, map_keypoints, status="ok"):
    ok = status == "ok"
    return SimulationRecord(
        seed=seed,
        fidelity=fidelity,
        status=status,
        true_position=[0.0, 0.0, 0.0],
        estimated_position=[error_m, 0.0, 0.0] if ok else None,
        error_vector=[error_m, 0.0, 0.0] if ok else None,
        error_m=error_m if ok else None,
        match_count=100,
        inlier_count=50,
        cost_s=cost_s,
        gpu="NVIDIA A40",
        scenario={},
        map_keypoints=map_keypoints,
    )


MAP = 70726


def sweep(levels, seeds=range(300), base_seed=0, failing=None):
    """A sweep where each level's error is the real one plus noise of the given SD.

    `levels`: {name: (noise_sd, cost_s)}; "splat" and "splat-rerun" are added. `failing`:
    {name: seeds where that level gets no fix}.
    """
    rng = np.random.default_rng(base_seed)
    truth = {s: 2.0 + rng.gamma(2.0, 1.0) for s in seeds}
    failing = failing or {}
    out = {}
    for name, (sd, cost) in {"splat": (0.0, REAL_COST), "splat-rerun": (0.05, REAL_COST), **levels}.items():
        k = MAP if name in ("splat", "splat-rerun") else int(name.split("-k")[1])
        fidelity = "splat" if name == "splat-rerun" else name
        out[name] = {
            s: record(
                s,
                fidelity,
                truth[s] + rng.normal(0, sd),
                cost * (1 + 0.01 * rng.normal()),
                k,
                "too_few_matches" if s in failing.get(name, ()) else "ok",
            )
            for s in seeds
        }
    return out


# --- MXMCPy's predicted variance, relative to plain MC at the same budget


def test_a_two_model_ratio_matches_the_mfmc_formula():
    # Two-model MFMC at its optimum: (sqrt(1 - rho^2) + sqrt(w rho^2))^2, w the cost ratio.
    rho, w = 0.95, 0.3
    cov = np.array([[1.0, rho], [rho, 1.0]])
    expected = (math.sqrt(1 - rho**2) + math.sqrt(w * rho**2)) ** 2
    assert variance_ratio(np.array([1.0, w]), cov) == pytest.approx(expected, rel=0.02)


def test_an_uncorrelated_cheap_model_predicts_no_saving():
    # The null case: a cheap model that knows nothing about the real one can't help.
    ratio = variance_ratio(np.array([1.0, 0.1]), np.eye(2))
    assert ratio is None or ratio >= 0.99


def test_models_in_an_order_mfmc_cannot_use_give_none():
    # The cheapest model correlates better than the middle one: MFMC's ordering fails.
    cov = np.array([[1, 0.5, 0.9], [0.5, 1, 0.5], [0.9, 0.5, 1.0]])
    assert variance_ratio(np.array([1.0, 0.3, 0.1]), cov) is None


# --- the rule


def test_the_rule_picks_the_cheap_well_correlated_pair_and_drops_the_rest():
    levels = {
        "splat-k1000": (0.2, 0.05),  # cheap and tracks well, but fails often: out on disagreements
        "splat-k4000": (0.1, 0.10),
        "splat-k16000": (0.05, 0.20),
        "splat-k32000": (0.02, 0.27),  # tracks best but barely cheaper: out on cost
    }
    report = apply_rule(sweep(levels, failing={"splat-k1000": range(0, 200, 20)}))
    assert report.gates_passed
    eligible = {s.name for s in report.levels if s.eligible}
    assert eligible == {"splat-k4000", "splat-k16000"}
    by_name = {s.name: s for s in report.levels}
    assert by_name["splat-k1000"].disagreements == 10
    assert "disagreements" in " ".join(by_name["splat-k1000"].reasons)
    assert "cost" in " ".join(by_name["splat-k32000"].reasons)
    assert report.chosen == ("splat-k16000", "splat-k4000")
    assert 0 < report.chosen_ratio < 1
    low, high = report.bootstrap_interval
    assert low <= report.chosen_ratio <= high


def test_a_level_that_tracks_poorly_is_not_eligible():
    levels = {"splat-k4000": (0.1, 0.10), "splat-k8000": (5.0, 0.15)}  # k8000 is mostly noise
    report = apply_rule(sweep(levels))
    by_name = {s.name: s for s in report.levels}
    assert not by_name["splat-k8000"].eligible
    assert "correlation" in " ".join(by_name["splat-k8000"].reasons)
    assert report.chosen == ("splat-k4000",)  # one eligible level: a two-level ladder, reported as such


def test_costs_and_correlations_use_only_the_selection_seeds():
    # Holdout seeds are wrecked here; the selection numbers mustn't move.
    levels = {"splat-k4000": (0.1, 0.10), "splat-k16000": (0.05, 0.20)}
    clean = apply_rule(sweep(levels))
    data = sweep(levels)
    for s in range(200, 300):
        old = data["splat-k4000"][s]
        data["splat-k4000"][s] = record(s, old.fidelity, 9999.0, 50.0, old.map_keypoints)
    wrecked = apply_rule(data)
    assert SELECTION == range(200)
    for a, b in zip(clean.levels, wrecked.levels, strict=True):
        assert (a.cost_s, a.rho) == (b.cost_s, b.rho)
    assert wrecked.chosen == clean.chosen
    assert wrecked.holdout_flagged and not clean.holdout_flagged


def test_a_jittery_real_navigator_stops_the_rule():
    # If the real navigator can't agree with itself, no correlation means anything.
    data = sweep({"splat-k4000": (0.1, 0.10)})
    rng = np.random.default_rng(5)
    for s, r in data["splat-rerun"].items():
        data["splat-rerun"][s] = record(s, r.fidelity, r.error_m + rng.normal(0, 10), r.cost_s, r.map_keypoints)
    report = apply_rule(data)
    assert not report.gates_passed
    assert "ceiling" in " ".join(report.gate_failures)
    assert report.chosen is None


def test_a_level_that_matched_more_keypoints_than_its_name_says_stops_the_rule():
    data = sweep({"splat-k4000": (0.1, 0.10)})
    for s, r in data["splat-k4000"].items():
        data["splat-k4000"][s] = record(s, r.fidelity, r.error_m, r.cost_s, MAP)
    report = apply_rule(data)
    assert not report.gates_passed
    assert "map_keypoints" in " ".join(report.gate_failures)
