"""The error-vs-sun-difference report (#22, docs/benchmarks/sun-degradation.md), on records whose answer is known."""

import math

import numpy as np
import pytest

from fidelis.degradation import (
    bearing_deg,
    by_sun,
    east_north,
    exceedance,
    record_sun,
    spread,
)
from fidelis.plain_mc import wilson_interval
from fidelis.render.mesh_render import Sun
from fidelis.simulation import SimulationRecord

SCENE = {
    "ctx_map_sun": {"azimuth_deg": 270.0, "elevation_deg": 40.0},
    "suns": [{"azimuth_deg": 270.0, "elevation_deg": 40.0}, {"azimuth_deg": 270.0, "elevation_deg": 25.0}],
    "stress_suns": [{"azimuth_deg": 90.0, "elevation_deg": 40.0}],
}


def record(seed, east, north, sun_index=0, sun_list="suns", status="ok"):
    ok = status == "ok"
    vector = [east, north, 3.0] if ok else None
    return SimulationRecord(
        seed=seed,
        fidelity="mesh",
        status=status,
        true_position=[0.0, 0.0, 0.0],
        estimated_position=vector,
        error_vector=vector,
        error_m=None,
        match_count=100,
        inlier_count=80 if ok else 0,
        cost_s=0.5,
        gpu="NVIDIA A40",
        scenario={"sun_index": sun_index},
        map_frame_error_vector=vector,
        sun_list=sun_list,
    )


@pytest.mark.parametrize(
    ("east", "north", "expected"),
    [(0.0, 1.0, 0.0), (1.0, 0.0, 90.0), (0.0, -1.0, 180.0), (-1.0, 0.0, 270.0), (1.0, 1.0, 45.0)],
)
def test_bearing_is_clockwise_from_north(east, north, expected):
    assert bearing_deg(east, north) == pytest.approx(expected)


def test_record_sun_follows_the_records_sun_list():
    assert record_sun(record(0, 0, 0, sun_index=1), SCENE) == Sun(270.0, 25.0)
    assert record_sun(record(0, 0, 0, sun_index=0, sun_list="stress_suns"), SCENE) == Sun(90.0, 40.0)


def test_record_sun_treats_a_missing_sun_list_as_the_realistic_one():
    assert record_sun(record(0, 0, 0, sun_index=1, sun_list=None), SCENE) == Sun(270.0, 25.0)


def test_east_north_drops_height_and_counts_no_fixes():
    records = [record(0, 3.0, 4.0), record(1, 0, 0, status="too_few_matches"), record(2, -1.0, 2.0)]
    vectors, no_fix = east_north(records)
    np.testing.assert_allclose(vectors, [[3.0, 4.0], [-1.0, 2.0]])
    assert no_fix == 1


def test_by_sun_groups_by_sun_difference_from_the_map():
    records = [record(0, 1, 0, sun_index=0), record(1, 1, 0, sun_index=1), record(2, 1, 0, sun_index=1)]
    groups = by_sun(records, SCENE)
    assert sorted(groups) == [pytest.approx(0.0, abs=1e-6), pytest.approx(15.0)]
    assert [len(groups[k]) for k in sorted(groups)] == [1, 2]


def test_a_constant_error_has_no_scatter_and_zero_width_intervals():
    """The null case: every descent off by the same vector."""
    vectors = np.tile([3.0, 4.0], (50, 1))
    s = spread(vectors, no_fix=0, resamples=200, seed=0)
    assert s.shift == pytest.approx(5.0)
    assert s.shift_bearing == pytest.approx(math.degrees(math.atan2(3.0, 4.0)))
    assert s.scatter_median == pytest.approx(0.0, abs=1e-12)
    assert s.median == pytest.approx(5.0)
    assert s.median_ci == pytest.approx((5.0, 5.0))
    assert s.scatter_ci == pytest.approx((0.0, 0.0), abs=1e-12)


def test_scatter_is_distance_from_the_mean_shift_not_from_zero():
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    vectors = np.column_stack([10.0 + 2.0 * np.cos(angles), 2.0 * np.sin(angles)])
    s = spread(vectors, no_fix=0, resamples=200, seed=0)
    assert s.shift == pytest.approx(10.0)
    assert s.shift_bearing == pytest.approx(90.0)
    assert s.scatter_median == pytest.approx(2.0)


def test_spread_reports_the_sample_statistics_with_intervals_around_them():
    rng = np.random.default_rng(1)
    vectors = rng.normal(0.0, 3.0, size=(500, 2))
    errors = np.hypot(vectors[:, 0], vectors[:, 1])
    s = spread(vectors, no_fix=2, resamples=500, seed=0)
    assert (s.n, s.no_fix) == (500, 2)
    assert s.median == pytest.approx(np.median(errors))
    assert s.p90 == pytest.approx(np.percentile(errors, 90))
    assert s.max == pytest.approx(errors.max())
    for value, (low, high) in [(s.median, s.median_ci), (s.p90, s.p90_ci), (s.shift, s.shift_ci)]:
        assert low <= value <= high
        assert low < high


def test_spread_is_repeatable_for_a_seed():
    vectors = np.random.default_rng(2).normal(size=(100, 2))
    assert spread(vectors, 0, 300, seed=5) == spread(vectors, 0, 300, seed=5)


def test_spread_needs_at_least_one_fix():
    with pytest.raises(ValueError):
        spread(np.empty((0, 2)), no_fix=3, resamples=10, seed=0)


def test_exceedance_counts_no_fixes_beyond_every_line_and_is_strict():
    rows = exceedance([1.0, 5.0, 5.0, 12.0], no_fix=1, xs=[5.0, 40.0])
    assert [(r.x, r.count, r.n) for r in rows] == [(5.0, 2, 5), (40.0, 1, 5)]
    assert rows[0].p == pytest.approx(2 / 5)
    assert rows[1].upper == pytest.approx(wilson_interval(1, 5, 0.95)[1])


def test_report_counts_the_realistic_no_fixes_in_the_exceedance_curve(tmp_path):
    """The stress runs, read after the realistic ones, must not change the realistic exceedance counts."""
    from fidelis.degradation import REALISTIC_RUNS, STRESS_RUNS, report
    from fidelis.plain_mc import write_records

    for i, run in enumerate(REALISTIC_RUNS):
        records = [record(10 * i + j, 1.0, 1.0, sun_index=j % 2) for j in range(4)]
        if i == 0:
            records.append(record(99, 0, 0, status="too_few_matches"))
        write_records(records, tmp_path / run / "records.jsonl")
    for run in STRESS_RUNS:
        write_records(
            [record(j, 2.0, 0.0, sun_list="stress_suns") for j in range(3)],
            tmp_path / "results/stress_sun" / run / "records.jsonl",
        )
    text = report(tmp_path, SCENE, resamples=20)
    assert "| 40 m | 1 of 17 |" in text
