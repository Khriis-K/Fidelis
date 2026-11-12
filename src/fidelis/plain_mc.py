"""Plain Monte Carlo: run N seeded descents, count failures, report P(fail) with a confidence interval.

A descent fails if the navigator gives no fix at all, or its fix is more than r from the
truth in the map frame: after removing the whole-map offset between the matching map and
the scene (fidelis.map_frame). r is LVS's 40 m requirement, from the scene config, and is
not tuned to a target failure rate (docs/adr/0003-failure-is-40m-in-the-map-frame.md).
No-fix and beyond-r failures are counted separately.

The interval is Wilson's score interval: unlike the textbook p +- z sqrt(p(1-p)/n), it
keeps an upper bound above zero when no failures were seen, and stays inside [0, 1].

    uv run fidelis-plain-mc config/jezero.yaml data/jezero results/plain_mc

writes records.jsonl (one SimulationRecord per seed; any of them reruns from its seed
and the config) and estimate.json. On the GPU a rerun repeats the camera image but not
the exact error, which the navigator's sensitivity moves by up to 0.839 m so far: the
record is the result (docs/adr/0001-gpu-reproducibility-is-image-level.md).
"""

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist, median

import yaml

from fidelis.map_frame import MATCHING_MAPS
from fidelis.scenario import MC_SEED_LIMIT, ScenarioConfig, draw, load_config, pin_sun
from fidelis.simulation import SimulationRecord
from fidelis.terrain.fetch import sha256_of


def is_no_fix(record: SimulationRecord) -> bool:
    return record.status != "ok"


def is_beyond_r(record: SimulationRecord, r_m: float) -> bool:
    """A fix more than r off in the map frame. Records from before ADR 0003 have no map-frame
    error; their map was the scene's own, so the offset was zero and the true-frame error is it."""
    if is_no_fix(record):
        return False
    error = record.error_m if record.map_frame_error_m is None else record.map_frame_error_m
    return error > r_m


def is_failure(record: SimulationRecord, r_m: float) -> bool:
    return is_no_fix(record) or is_beyond_r(record, r_m)


def wilson_interval(failures: int, n: int, confidence: float) -> tuple[float, float]:
    if n < 1:
        raise ValueError("an interval needs at least one sample")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = failures / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class Estimate:
    n: int
    failures: int  # no_fix + beyond_r
    no_fix: int
    beyond_r: int  # got a fix, but more than r off in the map frame
    p: float
    ci_low: float
    ci_high: float
    confidence: float
    r_m: float
    # The whole-map offset [east, north] the records' map-frame errors removed, and its
    # source. None for records from before ADR 0003, which didn't record one.
    map_offset_m: list[float] | None
    map_offset_source: str | None

    def summary(self) -> str:
        if self.map_offset_m is None:
            offset = "map offset not recorded (records from before ADR 0003, when it was zero)"
        else:
            east, north = self.map_offset_m
            offset = f"map offset {east:g} m east, {north:g} m north ({self.map_offset_source})"
        return (
            f"P(fail) = {self.p:.3f} ({self.failures}/{self.n}: {self.no_fix} no fix, {self.beyond_r} beyond r),"
            f" {self.confidence:.0%} Wilson interval [{self.ci_low:.3f}, {self.ci_high:.3f}],"
            f" at r = {self.r_m:g} m in the map frame; {offset}"
        )


def estimate(records: list[SimulationRecord], r_m: float, confidence: float = 0.95) -> Estimate:
    """P(fail) over the records, at r in the map frame. All records must share one map offset."""
    offsets = {(None if r.map_offset_m is None else tuple(r.map_offset_m), r.map_offset_source) for r in records}
    if len(offsets) > 1:
        raise ValueError(f"records measured against different map offsets can't share an estimate: {offsets}")
    offset, source = next(iter(offsets), (None, None))
    no_fix = sum(is_no_fix(r) for r in records)
    beyond_r = sum(is_beyond_r(r, r_m) for r in records)
    failures = no_fix + beyond_r
    low, high = wilson_interval(failures, len(records), confidence)
    return Estimate(
        n=len(records),
        failures=failures,
        no_fix=no_fix,
        beyond_r=beyond_r,
        p=failures / len(records),
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        r_m=r_m,
        map_offset_m=None if offset is None else list(offset),
        map_offset_source=source,
    )


def run_samples(simulator, config: ScenarioConfig, seeds: Iterable[int], fidelity: str) -> list[SimulationRecord]:
    """One record per seed, in seed order. Runs them grouped by sun, so each sun's scene loads once."""
    scenarios = sorted((draw(config, seed) for seed in seeds), key=lambda s: (s.sun_index, s.seed))
    records = [simulator.run(s, fidelity).record for s in scenarios]
    return sorted(records, key=lambda r: r.seed)


def write_records(records: list[SimulationRecord], path: Path) -> None:
    """JSON Lines, one record per line, written to a temp file and renamed into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(asdict(r)) + "\n" for r in records), encoding="utf-8")
    tmp.replace(path)


def read_records(path: Path) -> list[SimulationRecord]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [SimulationRecord(**json.loads(line)) for line in lines if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("data_dir", type=Path, help="the workflow's scene data, e.g. data/jezero")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--samples", type=int, help="default: the config's plain_mc.samples")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--fidelity", default="mesh")
    parser.add_argument(
        "--sun-index", type=int, help="draw every descent under this sun (while only some suns have a splat scene)"
    )
    parser.add_argument(
        "--matching-map",
        choices=MATCHING_MAPS,
        help="default: the config's matching_map.source; 'scene' reruns earlier results on the map they used",
    )
    args = parser.parse_args()

    from fidelis.simulation import build_simulator

    scene = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = scene["plain_mc"]
    n = args.samples or settings["samples"]
    seeds = range(args.first_seed, args.first_seed + n)
    if seeds.stop > MC_SEED_LIMIT:
        raise ValueError(f"Monte Carlo seeds must stay below {MC_SEED_LIMIT}: splat training views use those above")
    config = load_config(args.config)
    if args.sun_index is not None:
        config = pin_sun(config, args.sun_index)
    simulator = build_simulator(args.config, args.data_dir, (args.fidelity,), matching_map=args.matching_map)
    records = run_samples(simulator, config, seeds, args.fidelity)
    write_records(records, args.out_dir / "records.jsonl")

    est = estimate(records, settings["r_m"]["value"])
    out = asdict(est) | {
        "fidelity": args.fidelity,
        "matching_map": simulator.matching_map,
        "seeds": [seeds.start, seeds.stop - 1],
        # None: each seed drew its own sun. Reruns of a pinned run need the same --sun-index.
        "sun_index_pinned": args.sun_index,
        "gpu": sorted({r.gpu for r in records}),
        "median_cost_s": median(r.cost_s for r in records),
        # Reruns need the same config: its hash says whether a later one still is.
        "config": str(args.config),
        "config_sha256": sha256_of(args.config),
        "r_source": settings["r_m"]["source"],
    }
    path = args.out_dir / "estimate.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(est.summary())
    return 0
