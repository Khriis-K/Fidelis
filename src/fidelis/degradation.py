"""How navigation degrades with sun difference (#22, docs/benchmarks/sun-degradation.md).

From saved records only: per sun, the horizontal map-frame error's median, 90th percentile and
maximum, its mean shift and the scatter about it, with bootstrap intervals; and, for the
realistic series, the exceedance curve P(error > x). No descents are rendered.

The realistic (#47) and stress (#43) series differ in texture and sourced errors as well as
sun, so they are reported side by side and never joined.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from fidelis.plain_mc import is_no_fix, read_records, wilson_interval
from fidelis.render.mesh_render import Sun
from fidelis.simulation import SimulationRecord
from fidelis.stress import REALISTIC_SUNS, ctx_map_sun, sun_difference_deg, suns_for

R_M = 40.0  # LVS's requirement, ADR 0003
EXCEEDANCE_XS = (2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0)
RESAMPLES = 2000
CONFIDENCE = 0.95


def bearing_deg(east: float, north: float) -> float:
    """Compass bearing of an east-north vector: 0 north, 90 east."""
    return math.degrees(math.atan2(east, north)) % 360.0


def record_sun(record: SimulationRecord, scene: dict) -> Sun:
    """The sun that lit a record, from its sun list (None: the realistic one, as before #43)."""
    return suns_for(scene, record.sun_list or REALISTIC_SUNS)[record.scenario["sun_index"]]


def by_sun(records: list[SimulationRecord], scene: dict) -> dict[float, list[SimulationRecord]]:
    """Records grouped by their sun's angle from the map's sun, degrees."""
    map_sun = ctx_map_sun(scene)
    groups: dict[float, list[SimulationRecord]] = {}
    for r in records:
        groups.setdefault(sun_difference_deg(record_sun(r, scene), map_sun), []).append(r)
    return groups


def east_north(records: list[SimulationRecord]) -> tuple[np.ndarray, int]:
    """The map-frame east-north errors of the descents that got a fix, and how many didn't."""
    fixed = [r.map_frame_error_vector[:2] for r in records if not is_no_fix(r)]
    return np.array(fixed, dtype=float).reshape(-1, 2), len(records) - len(fixed)


@dataclass(frozen=True)
class Spread:
    n: int  # descents with a fix
    no_fix: int
    median: float
    median_ci: tuple[float, float]
    p90: float
    p90_ci: tuple[float, float]
    max: float  # no interval: a bootstrap never exceeds the sample maximum
    shift: float  # size of the mean east-north error
    shift_ci: tuple[float, float]
    shift_bearing: float
    scatter_median: float  # median distance from the mean shift
    scatter_ci: tuple[float, float]


def _stats(vectors: np.ndarray) -> np.ndarray:
    errors = np.hypot(vectors[:, 0], vectors[:, 1])
    mean = vectors.mean(axis=0)
    scatter = np.hypot(*(vectors - mean).T)
    return np.array([np.median(errors), np.percentile(errors, 90), math.hypot(*mean), np.median(scatter)])


def spread(vectors: np.ndarray, no_fix: int, resamples: int, seed: int) -> Spread:
    """One group's error statistics, with percentile-bootstrap intervals over descents."""
    if len(vectors) == 0:
        raise ValueError("a spread needs at least one descent with a fix")
    point = _stats(vectors)
    rng = np.random.default_rng(seed)
    boot = np.array([_stats(vectors[rng.integers(0, len(vectors), len(vectors))]) for _ in range(resamples)])
    tail = (1 - CONFIDENCE) / 2 * 100
    low, high = np.percentile(boot, [tail, 100 - tail], axis=0)
    ci = [(float(lo), float(hi)) for lo, hi in zip(low, high, strict=True)]
    mean = vectors.mean(axis=0)
    return Spread(
        n=len(vectors),
        no_fix=no_fix,
        median=float(point[0]),
        median_ci=ci[0],
        p90=float(point[1]),
        p90_ci=ci[1],
        max=float(np.hypot(vectors[:, 0], vectors[:, 1]).max()),
        shift=float(point[2]),
        shift_ci=ci[2],
        shift_bearing=bearing_deg(*mean),
        scatter_median=float(point[3]),
        scatter_ci=ci[3],
    )


@dataclass(frozen=True)
class Exceedance:
    x: float
    count: int
    n: int
    p: float
    upper: float  # 95% Wilson upper bound


def exceedance(errors, no_fix: int, xs) -> list[Exceedance]:
    """P(error > x) for each x; a descent with no fix is beyond every x."""
    errors = np.asarray(errors, dtype=float)
    n = len(errors) + no_fix
    rows = []
    for x in xs:
        count = int((errors > x).sum()) + no_fix
        rows.append(Exceedance(x, count, n, count / n, wilson_interval(count, n, CONFIDENCE)[1]))
    return rows


# --- the report ------------------------------------------------------------------------------

REALISTIC_RUNS = [f"results/certified_bound/batch{i}" for i in range(4)]
STRESS_RUNS = ["anchor", "stress1", "stress2", "stress3", "stress3_photo"]


def _ci(pair: tuple[float, float]) -> str:
    return f"{pair[0]:.2f}–{pair[1]:.2f}"


def _row(label: str, sun: Sun | None, difference: float | None, s: Spread) -> str:
    """One table row; a pooled row has no single sun, so no sun columns and no relative bearing."""
    sun_text = "pooled" if sun is None else f"{sun.azimuth_deg:g}°, {sun.elevation_deg:g}°"
    difference_text = "—" if difference is None else f"{difference:.1f}°"
    bearing = f"{s.shift_bearing:.0f}°"
    if sun is not None:
        bearing += f" ({(s.shift_bearing - sun.azimuth_deg) % 360:.0f}° from sun)"
    return (
        f"| {label} | {sun_text} | {difference_text} | {s.n + s.no_fix:,} | {s.no_fix} "
        f"| {s.median:.2f} ({_ci(s.median_ci)}) | {s.p90:.2f} ({_ci(s.p90_ci)}) | {s.max:.2f} "
        f"| {s.shift:.2f} ({_ci(s.shift_ci)}) | {bearing} | {s.scatter_median:.2f} ({_ci(s.scatter_ci)}) |"
    )


HEADER = (
    "| Group | Sun (az, el) | Sun difference | Descents | No fix | Median (95% CI) | 90th pct (95% CI) | Max "
    "| Mean shift (95% CI) | Shift bearing | Scatter median (95% CI) |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


def report(root: Path, scene: dict, resamples: int = RESAMPLES) -> str:
    lines = ["All errors are horizontal, in the map frame, in metres. Generated by `fidelis-degradation`.", ""]

    realistic = [r for run in REALISTIC_RUNS for r in read_records(root / run / "records.jsonl")]
    lines += ["### Realistic series (#47, photo texture, sourced errors on)", "", HEADER]
    for difference, group in sorted(by_sun(realistic, scene).items()):
        vectors, no_fix = east_north(group)
        lines.append(
            _row(
                f"sun {group[0].scenario['sun_index']}",
                record_sun(group[0], scene),
                difference,
                spread(vectors, no_fix, resamples, seed=0),
            )
        )
    pooled, pooled_no_fix = east_north(realistic)
    lines.append(_row("all", None, None, spread(pooled, pooled_no_fix, resamples, seed=0)))

    lines += [
        "",
        "### Stress series (#43, de-shaded texture, sourced errors off; a diagnostic, not a rate)",
        "",
        HEADER,
    ]
    map_sun = ctx_map_sun(scene)
    for run in STRESS_RUNS:
        group = read_records(root / "results/stress_sun" / run / "records.jsonl")
        sun = record_sun(group[0], scene)
        vectors, no_fix = east_north(group)
        lines.append(_row(run, sun, sun_difference_deg(sun, map_sun), spread(vectors, no_fix, resamples, seed=0)))

    errors = np.hypot(pooled[:, 0], pooled[:, 1])
    lines += [
        "",
        "### Exceedance, realistic series pooled (exploratory; r = 40 m is the official line)",
        "",
        "| x | Descents beyond x | P(error > x) | 95% upper bound |",
        "|---|---|---|---|",
    ]
    for e in exceedance(errors, pooled_no_fix, EXCEEDANCE_XS):
        lines.append(f"| {e.x:g} m | {e.count:,} of {e.n:,} | {e.p:.4%} | {e.upper:.4%} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/jezero.yaml"))
    parser.add_argument("--root", type=Path, default=Path("."), help="the directory holding results/")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scene = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    text = report(args.root, scene)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(args.out)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
