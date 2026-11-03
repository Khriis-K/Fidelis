"""Plain Monte Carlo: run N seeded descents, count failures, report P(fail) with a confidence interval.

A descent fails if the navigator gives no fix at all, or its fix is more than r from the
truth. Until the pilot study (#13) sets r, the scene config's r is provisional, and every
output says so.

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

from fidelis.scenario import MC_SEED_LIMIT, ScenarioConfig, draw, load_config, pin_sun
from fidelis.simulation import SimulationRecord
from fidelis.terrain.fetch import sha256_of


def is_failure(record: SimulationRecord, r_m: float) -> bool:
    return record.status != "ok" or record.error_m > r_m


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
    failures: int
    p: float
    ci_low: float
    ci_high: float
    confidence: float
    r_m: float
    r_provisional: bool

    def summary(self) -> str:
        r = f"{'PROVISIONAL ' if self.r_provisional else ''}r = {self.r_m:g} m"
        return (
            f"P(fail) = {self.p:.3f} ({self.failures}/{self.n}), {self.confidence:.0%} Wilson interval"
            f" [{self.ci_low:.3f}, {self.ci_high:.3f}], at {r}"
        )


def estimate(records: list[SimulationRecord], r_m: float, r_provisional: bool, confidence: float = 0.95) -> Estimate:
    failures = sum(is_failure(r, r_m) for r in records)
    low, high = wilson_interval(failures, len(records), confidence)
    return Estimate(len(records), failures, failures / len(records), low, high, confidence, r_m, r_provisional)


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
    simulator = build_simulator(args.config, args.data_dir, (args.fidelity,))
    records = run_samples(simulator, config, seeds, args.fidelity)
    write_records(records, args.out_dir / "records.jsonl")

    est = estimate(records, settings["provisional_r_m"], r_provisional=True)
    out = asdict(est) | {
        "fidelity": args.fidelity,
        "seeds": [seeds.start, seeds.stop - 1],
        # None: each seed drew its own sun. Reruns of a pinned run need the same --sun-index.
        "sun_index_pinned": args.sun_index,
        "gpu": sorted({r.gpu for r in records}),
        "median_cost_s": median(r.cost_s for r in records),
        # Reruns need the same config: its hash says whether a later one still is.
        "config": str(args.config),
        "config_sha256": sha256_of(args.config),
        "r_note": "provisional: r will be set from the pilot study (#13), not from this run",
    }
    path = args.out_dir / "estimate.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(est.summary())
    return 0
