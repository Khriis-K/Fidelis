import json
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from scipy.stats import binomtest

from fidelis.plain_mc import estimate, is_failure, read_records, run_samples, wilson_interval, write_records
from fidelis.scenario import draw, load_config
from fidelis.simulation import SimulationRecord

REPO = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO / "config" / "jezero.yaml")


def record(seed=0, status="ok", error_m=1.0):
    ok = status == "ok"
    return SimulationRecord(
        seed=seed,
        fidelity="mesh",
        status=status,
        true_position=[0.0, 0.0, 0.0],
        estimated_position=[error_m, 0.0, 0.0] if ok else None,
        error_vector=[error_m, 0.0, 0.0] if ok else None,
        error_m=error_m if ok else None,
        match_count=100,
        inlier_count=80 if ok else 0,
        cost_s=1.5,
        gpu="test GPU",
        scenario={},
    )


# --- what counts as a failure


def test_an_error_beyond_r_is_a_failure_and_one_within_it_is_not():
    assert is_failure(record(error_m=12.5), r_m=12.0)
    assert not is_failure(record(error_m=11.9), r_m=12.0)


@pytest.mark.parametrize("status", ["too_few_matches", "no_solution"])
def test_no_fix_at_all_is_a_failure_whatever_r_is(status):
    # A navigator that gives up hasn't kept the lander within r: that's a failure, not a skip.
    assert is_failure(record(status=status), r_m=1e9)


# --- the confidence interval


@pytest.mark.parametrize("failures, n", [(0, 10), (1, 20), (3, 50), (7, 7), (40, 400)])
def test_the_interval_is_the_wilson_score_interval(failures, n):
    expected = binomtest(failures, n).proportion_ci(confidence_level=0.95, method="wilson")
    low, high = wilson_interval(failures, n, 0.95)
    assert low == pytest.approx(expected.low, abs=1e-12)
    assert high == pytest.approx(expected.high, abs=1e-12)


def test_no_failures_still_gives_an_upper_bound_above_zero():
    # The null case: a Wald interval would collapse to [0, 0] and claim certainty.
    low, high = wilson_interval(0, 20, 0.95)
    assert low == 0 and high > 0.1


def test_a_wider_confidence_gives_a_wider_interval():
    narrow, wide = wilson_interval(3, 40, 0.9), wilson_interval(3, 40, 0.99)
    assert wide[0] < narrow[0] and wide[1] > narrow[1]


def test_no_samples_is_a_caller_error():
    with pytest.raises(ValueError, match="sample"):
        wilson_interval(0, 0, 0.95)


# --- the estimate


def test_the_estimate_counts_failures_and_carries_its_interval_and_threshold():
    records = [record(0, error_m=2.0), record(1, error_m=30.0), record(2, status="no_solution"), record(3, error_m=5.0)]
    est = estimate(records, r_m=12.0, r_provisional=True)
    assert (est.n, est.failures, est.p) == (4, 2, 0.5)
    assert (est.ci_low, est.ci_high) == pytest.approx(wilson_interval(2, 4, 0.95))
    assert est.confidence == 0.95
    assert est.r_m == 12.0 and est.r_provisional is True


def test_the_estimate_is_json_safe_and_says_r_is_provisional():
    est = estimate([record(0)], r_m=12.0, r_provisional=True)
    written = json.loads(json.dumps(asdict(est)))
    assert written["r_provisional"] is True
    assert "provisional" in est.summary().lower()


def test_a_final_r_is_not_called_provisional():
    est = estimate([record(0)], r_m=12.0, r_provisional=False)
    assert "provisional" not in est.summary().lower()


# --- running and writing samples


class FakeSimulator:
    """Records the scenarios it's asked to run and answers with a record for each."""

    def __init__(self):
        self.calls = []

    def run(self, scenario, fidelity):
        self.calls.append((scenario, fidelity))
        values = {f.name: getattr(scenario, f.name) for f in fields(scenario) if f.name != "seed"}
        return SimpleNamespace(record=SimulationRecord(**(asdict(record(scenario.seed)) | {"scenario": values})))


def test_each_seed_runs_its_drawn_scenario_and_records_come_back_in_seed_order():
    sim = FakeSimulator()
    records = run_samples(sim, CONFIG, range(20, 32), "mesh")
    assert [r.seed for r in records] == list(range(20, 32))
    assert all(fidelity == "mesh" for _, fidelity in sim.calls)
    assert sorted((s for s, _ in sim.calls), key=lambda s: s.seed) == [draw(CONFIG, seed) for seed in range(20, 32)]


def test_samples_run_grouped_by_sun_so_each_scene_loads_once():
    sim = FakeSimulator()
    run_samples(sim, CONFIG, range(40), "mesh")
    suns = [s.sun_index for s, _ in sim.calls]
    assert suns == sorted(suns)
    assert len(set(suns)) > 1  # forty draws over five suns: more than one sun was drawn


def test_written_records_read_back_unchanged(tmp_path):
    records = [record(0), record(1, status="no_solution"), record(2, error_m=40.0)]
    path = tmp_path / "out" / "records.jsonl"
    write_records(records, path)
    assert read_records(path) == records
    assert len(path.read_text().splitlines()) == 3  # one line per sample
    assert [p.name for p in path.parent.iterdir()] == ["records.jsonl"]  # no temp file left behind
