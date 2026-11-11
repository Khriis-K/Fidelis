import json
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.stats import binomtest

from fidelis.map_frame import MapOffset, map_frame_error
from fidelis.plain_mc import estimate, is_failure, read_records, run_samples, wilson_interval, write_records
from fidelis.scenario import draw, load_config
from fidelis.simulation import SimulationRecord

REPO = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO / "config" / "jezero.yaml")
SELF_MADE = MapOffset(0.0, 0.0, "test: map made from the scene's own orthoimage")


def record(seed=0, status="ok", error_m=1.0, offset=SELF_MADE):
    """A descent whose fix is error_m east of the truth (true frame), against a map shifted by offset."""
    ok = status == "ok"
    error = [error_m, 0.0, 0.0] if ok else None
    in_map = map_frame_error(error, offset) if ok else None
    return SimulationRecord(
        seed=seed,
        fidelity="mesh",
        status=status,
        true_position=[0.0, 0.0, 0.0],
        estimated_position=error,
        error_vector=error,
        error_m=abs(error_m) if ok else None,
        match_count=100,
        inlier_count=80 if ok else 0,
        cost_s=1.5,
        gpu="test GPU",
        scenario={},
        map_offset_m=[offset.east_m, offset.north_m],
        map_offset_source=offset.source,
        map_frame_error_vector=in_map,
        map_frame_error_m=float(np.linalg.norm(in_map)) if ok else None,
    )


MAP_FRAME_FIELDS = ("map_offset_m", "map_offset_source", "map_frame_error_vector", "map_frame_error_m")


def older_record(**kwargs):
    """As written before ADR 0003: no map-frame fields at all."""
    values = asdict(record(**kwargs))
    for name in MAP_FRAME_FIELDS:
        del values[name]
    return values


# --- what counts as a failure


def test_an_error_beyond_r_is_a_failure_and_one_within_it_is_not():
    assert is_failure(record(error_m=40.5), r_m=40.0)
    assert not is_failure(record(error_m=39.9), r_m=40.0)


@pytest.mark.parametrize("status", ["too_few_matches", "no_solution"])
def test_no_fix_at_all_is_a_failure_whatever_r_is(status):
    # A navigator that gives up hasn't kept the lander within r: that's a failure, not a skip.
    assert is_failure(record(status=status), r_m=1e9)


def test_a_miss_beyond_r_only_in_the_true_frame_is_not_a_failure():
    # 60 m from truth, but 55 m of it is the whole map's shift: 5 m off in the map frame.
    shifted = MapOffset(55.0, 0.0, "test")
    assert not is_failure(record(error_m=60.0, offset=shifted), r_m=40.0)


def test_a_miss_beyond_r_in_both_frames_is_a_failure():
    # 105 m from truth, 50 m of it the navigator's own: still beyond 40 m once the shift is removed.
    shifted = MapOffset(55.0, 0.0, "test")
    assert is_failure(record(error_m=105.0, offset=shifted), r_m=40.0)


def test_a_miss_beyond_r_only_in_the_map_frame_is_a_failure():
    # 15 m west of truth against a map shifted 30 m east: 45 m off in the map frame.
    shifted = MapOffset(30.0, 0.0, "test")
    assert is_failure(record(error_m=-15.0, offset=shifted), r_m=40.0)


def test_the_jezero_r_is_lvss_40_m_requirement_with_its_source():
    scene = yaml.safe_load((REPO / "config" / "jezero.yaml").read_text(encoding="utf-8"))
    r = scene["plain_mc"]["r_m"]
    assert r["value"] == 40
    assert "Mohan" in r["source"]


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
    records = [record(0, error_m=2.0), record(1, error_m=50.0), record(2, status="no_solution"), record(3, error_m=5.0)]
    est = estimate(records, r_m=40.0)
    assert (est.n, est.failures, est.p) == (4, 2, 0.5)
    assert (est.ci_low, est.ci_high) == pytest.approx(wilson_interval(2, 4, 0.95))
    assert est.confidence == 0.95
    assert est.r_m == 40.0


def test_the_estimate_counts_no_fix_and_beyond_r_failures_separately():
    records = [
        record(0, error_m=2.0),
        record(1, error_m=50.0),
        record(2, status="no_solution"),
        record(3, status="too_few_matches"),
        record(4, error_m=5.0),
    ]
    est = estimate(records, r_m=40.0)
    assert (est.failures, est.no_fix, est.beyond_r) == (3, 2, 1)
    summary = est.summary()
    assert "2 no fix" in summary and "1 beyond r" in summary


def test_the_estimate_reports_the_offset_its_records_were_measured_against():
    shifted = MapOffset(12.5, -3.0, "registered CTX to HiRISE (#40)")
    est = estimate([record(0, offset=shifted), record(1, offset=shifted)], r_m=40.0)
    assert est.map_offset_m == [12.5, -3.0]
    assert est.map_offset_source == "registered CTX to HiRISE (#40)"
    summary = est.summary()
    assert "map frame" in summary and "12.5" in summary and "-3" in summary


def test_records_measured_against_different_offsets_cannot_share_an_estimate():
    a, b = MapOffset(0.0, 0.0, "self-made"), MapOffset(14.0, 0.0, "CTX")
    with pytest.raises(ValueError, match="offset"):
        estimate([record(0, offset=a), record(1, offset=b)], r_m=40.0)


def test_the_estimate_is_json_safe():
    est = estimate([record(0)], r_m=40.0)
    written = json.loads(json.dumps(asdict(est)))
    assert written["r_m"] == 40.0 and written["no_fix"] == 0 and written["beyond_r"] == 0


def test_a_final_r_is_not_called_provisional():
    # ADR 0003 fixed r at LVS's 40 m: nothing the estimate prints may call it provisional.
    est = estimate([record(0)], r_m=40.0)
    assert "provisional" not in est.summary().lower()


# --- records written before ADR 0003


def test_an_older_record_without_map_frame_fields_still_loads(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(older_record(error_m=3.0)) + "\n", encoding="utf-8")
    (loaded,) = read_records(path)
    assert loaded.error_m == 3.0
    assert all(getattr(loaded, name) is None for name in MAP_FRAME_FIELDS)


def test_an_older_record_is_judged_by_its_true_frame_error():
    # Before #36 the map was made from the scene's own orthoimage, so the offset was zero.
    beyond, within = (SimulationRecord(**older_record(error_m=e)) for e in (45.0, 30.0))
    assert is_failure(beyond, r_m=40.0)
    assert not is_failure(within, r_m=40.0)
    est = estimate([beyond, within], r_m=40.0)
    assert (est.beyond_r, est.map_offset_m, est.map_offset_source) == (1, None, None)
    assert "not recorded" in est.summary()


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
