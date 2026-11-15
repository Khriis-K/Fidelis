"""The pre-registered multifidelity go/no-go with the sun varying (#48, docs/benchmarks/sun-sweep.md).

Reads the sweep's records, one directory per level ("mesh", "mesh-k<K>", and "mesh-rerun", the
real navigator run a second time), and the combined scan's records, which fit the free formula.
The quantity is the map-frame horizontal error. A cheap level is a go only if multifidelity with
it beats all three baselines at equal cost: plain MC, sun-stratified MC, and a control variate on
the formula (per-sun offset plus the focal shift -e*h*sin 2θ, fitted on other seeds).
Selection uses seeds 70,000-70,199; 70,200-70,299 are the holdout.

    uv run python -m fidelis.sun_sweep results/sun_sweep results/combined_scan/scan/records.jsonl
"""

import argparse
import itertools
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fidelis import fidelity_sweep
from fidelis.fidelity_sweep import BOOTSTRAP, LevelStats, Records, gate_failures, load_sweep, mark_eligible, median_cost
from fidelis.map_frame import horizontal_error_m
from fidelis.plain_mc import read_records
from fidelis.simulation import SimulationRecord, parse_fidelity

SELECTION = range(70000, 70200)
HOLDOUT = range(70200, 70300)
REAL, RERUN = "mesh", "mesh-rerun"
SUNS = 5  # config/jezero.yaml draws sun_index uniformly from 0-4, so each stratum weighs 1/5
METHODS = ("mfmc", "acvmf")


def horizontal_error(record: SimulationRecord) -> float:
    """The map-frame east-north error: what the 40 m line measures (ADR 0003, #46)."""
    return horizontal_error_m(record.map_frame_error_vector)


def focal_shift(scenario: dict) -> float:
    """The focal error's horizontal shift along the tilt, -e*h*sin 2θ (docs/benchmarks/focal-error.md)."""
    theta = math.radians(scenario["tilt_deg"])
    return -scenario.get("focal_error", 0.0) * scenario["altitude_m"] * math.sin(2 * theta)


def _features(record: SimulationRecord) -> np.ndarray:
    sc = record.scenario
    sun = np.eye(SUNS)[sc["sun_index"]]
    shift, az = focal_shift(sc), math.radians(sc["tilt_azimuth_deg"])
    return np.concatenate([sun, [shift * math.sin(az), shift * math.cos(az)]])


@dataclass
class Formula:
    """The free predictor: a per-sun offset plus the focal shift, as an east-north vector, by least squares.

    Fitted once on other seeds (the combined scan), then frozen, so its correlation on the sweep's
    seeds isn't fitted to them. Costs nothing: it reads only the scenario's inputs.
    """

    coefficients: np.ndarray  # features x (east, north)

    @classmethod
    def from_records(cls, records: list[SimulationRecord]) -> "Formula":
        ok = [r for r in records if r.status == "ok"]
        X = np.array([_features(r) for r in ok])
        Y = np.array([r.map_frame_error_vector[:2] for r in ok])
        return cls(np.linalg.lstsq(X, Y, rcond=None)[0])

    def predict(self, record: SimulationRecord) -> float:
        return float(np.hypot(*(_features(record) @ self.coefficients)))


def stratified_ratio(errors: np.ndarray, strata: np.ndarray) -> float:
    """Sun-stratified MC's variance over plain MC's at equal cost, proportional allocation, equal weights."""
    errors, strata = np.asarray(errors, float), np.asarray(strata)
    within = []
    for k in range(SUNS):
        part = errors[strata == k]
        if part.size < 2:
            raise ValueError(f"sun {k} has {part.size} descents; stratifying needs at least 2 in every sun")
        within.append(part.var(ddof=1))
    return float(np.mean(within) / errors.var(ddof=1))


def cv_ratio(errors: np.ndarray, predicted: np.ndarray) -> float:
    """A control variate's variance over plain MC's, for a free predictor with a known mean: 1 - rho^2."""
    if np.std(predicted) == 0:
        return 1.0
    return float(1 - np.corrcoef(errors, predicted)[0, 1] ** 2)


@dataclass
class Report:
    real_cost_s: float
    ceiling: float
    shuffle_rho: float
    gate_failures: list[str]
    levels: list[LevelStats]
    stratified_ratio: float | None = None
    cv_ratio: float | None = None
    option_ratios: dict[tuple[str, ...], tuple[float | None, str | None]] = field(default_factory=dict)
    chosen: tuple[str, ...] | None = None
    chosen_ratio: float | None = None
    chosen_method: str | None = None
    margin_interval: tuple[float, float] | None = None  # bootstrap 90%: chosen ratio - best baseline
    bootstrap_wins: float | None = None  # fraction of resamples where the chosen set beats every baseline
    holdout_ratio: float | None = None
    holdout_baseline: float | None = None

    @property
    def gates_passed(self) -> bool:
        return not self.gate_failures

    @property
    def sun_share(self) -> float:
        """The share of the real navigator's error variance the sun alone explains."""
        return 1 - self.stratified_ratio

    @property
    def formula_rho(self) -> float:
        return math.sqrt(max(0.0, 1 - self.cv_ratio))

    @property
    def best_baseline_ratio(self) -> float:
        return min(1.0, self.stratified_ratio, self.cv_ratio)

    @property
    def go(self) -> bool:
        return (
            self.gates_passed
            and self.chosen is not None
            and self.chosen_ratio < self.best_baseline_ratio
            and self.margin_interval is not None
            and self.margin_interval[1] < 0
            and self.holdout_ratio is not None
            and self.holdout_ratio < self.holdout_baseline
        )


def paired(levels: list[Records], seeds) -> np.ndarray:
    return fidelity_sweep.paired_errors(levels, seeds, horizontal_error)


def baselines(real: Records, formula: Formula, seeds) -> tuple[float, float]:
    """(sun-stratified ratio, formula control-variate ratio) for the real navigator on these seeds."""
    kept = [s for s in seeds if real[s].status == "ok"]
    errors = np.array([horizontal_error(real[s]) for s in kept])
    strata = np.array([real[s].scenario["sun_index"] for s in kept])
    predicted = np.array([formula.predict(real[s]) for s in kept])
    return stratified_ratio(errors, strata), cv_ratio(errors, predicted)


def best_baseline(real: Records, formula: Formula, seeds) -> float:
    return min(1.0, *baselines(real, formula, seeds))


def set_ratio(sweep: dict[str, Records], names: tuple[str, ...], seeds, method: str) -> float | None:
    return fidelity_sweep.set_ratio(sweep, names, seeds, method, error=horizontal_error, real=REAL)


def best_method(sweep, names, seeds) -> tuple[float | None, str | None]:
    ratios = {m: set_ratio(sweep, names, seeds, m) for m in METHODS}
    usable = {m: r for m, r in ratios.items() if r is not None}
    if not usable:
        return None, None
    method = min(usable, key=usable.get)
    return usable[method], method


def sourced_error_failures(sweep: dict[str, Records]) -> list[str]:
    """The formula was fitted with haze, focal and attitude error on; a run with any switched off
    (--clear-air, --exact-camera, --true-attitude) carries the same config SHA, so check the draws."""
    off = set()
    for name, level in sweep.items():
        for r in level.values():
            sc = r.scenario
            if not sc.get("dust_tau", 0) > 0:
                off.add((name, "haze"))
            if sc.get("focal_error", 0) == 0:
                off.add((name, "focal error"))
            if not any(sc.get(f"attitude_error_{a}_deg", 0) for a in "xyz"):
                off.add((name, "attitude error"))
    return [f"{name}: {what} is off in some records" for name, what in sorted(off)]


def all_gate_failures(sweep, candidates, ceiling, shuffle_rho) -> list[str]:
    """The first sweep's gates, plus one GPU model, one config, and the sourced errors on."""
    failures = gate_failures(sweep, candidates, ceiling, shuffle_rho) + sourced_error_failures(sweep)
    records = [r for level in sweep.values() for r in level.values()]
    if len(gpus := {r.gpu for r in records}) > 1:
        failures.append(f"records come from more than one GPU model: {sorted(gpus)}")
    if len(shas := {r.config_sha256 for r in records}) > 1:
        failures.append(f"records come from more than one config: {sorted(map(str, shas))}")
    return failures


def apply_rule(sweep: dict[str, Records], formula: Formula, seed: int = 0) -> Report:
    selection = list(SELECTION)
    candidates = sorted((n for n in sweep if parse_fidelity(n)[1] is not None), key=lambda n: parse_fidelity(n)[1])
    real_cost = median_cost(sweep[REAL], selection)
    ceiling = float(np.corrcoef(paired([sweep[REAL], sweep[RERUN]], selection))[0, 1])

    # Gate: shuffling the cheapest level's seeds must destroy the correlation.
    rng = np.random.default_rng(seed)
    errors = paired([sweep[REAL], sweep[candidates[0]]], selection)
    shuffle_rho = float(np.corrcoef(errors[0], rng.permutation(errors[1]))[0, 1])

    stats = [fidelity_sweep.level_stats(sweep, n, selection, error=horizontal_error, real=REAL) for n in candidates]
    mark_eligible(stats, ceiling, real_cost)

    report = Report(real_cost, ceiling, shuffle_rho, all_gate_failures(sweep, candidates, ceiling, shuffle_rho), stats)
    report.stratified_ratio, report.cv_ratio = baselines(sweep[REAL], formula, selection)
    if not report.gates_passed:
        return report

    # Higher K first: that's the next model down from the real navigator.
    eligible = sorted((s.name for s in stats if s.eligible), key=lambda n: -parse_fidelity(n)[1])
    options = [(n,) for n in eligible] + list(itertools.combinations(eligible, 2))
    report.option_ratios = {names: best_method(sweep, names, selection) for names in options}
    usable = {names: r for names, (r, _) in report.option_ratios.items() if r is not None}
    if not usable:
        return report
    report.chosen = min(usable, key=usable.get)
    report.chosen_ratio, report.chosen_method = report.option_ratios[report.chosen]

    # A resample where the chosen method can't allocate counts as a loss, not as missing.
    margins = []
    for _ in range(BOOTSTRAP):
        resample = list(rng.choice(selection, size=len(selection), replace=True))
        ratio = set_ratio(sweep, report.chosen, resample, report.chosen_method)
        margins.append(math.inf if ratio is None else ratio - best_baseline(sweep[REAL], formula, resample))
    margins = np.array(margins)
    report.margin_interval = tuple(float(q) for q in np.percentile(margins, [5, 95]))
    report.bootstrap_wins = float(np.mean(margins < 0))

    holdout = list(HOLDOUT)
    report.holdout_ratio = set_ratio(sweep, report.chosen, holdout, report.chosen_method)
    report.holdout_baseline = best_baseline(sweep[REAL], formula, holdout)
    return report


def markdown(report: Report) -> str:
    lines = [
        f"Real navigator: median cost {report.real_cost_s:.4f} s.",
        f"Ceiling (real vs real rerun): rho = {report.ceiling:.4f}. Shuffled control: rho = {report.shuffle_rho:.4f}.",
        "",
        "Baselines, predicted variance / plain MC at equal cost (selection seeds):",
        f"- sun-stratified MC: {report.stratified_ratio:.4f} (the sun alone explains {report.sun_share:.1%})",
        f"- control variate on the formula: {report.cv_ratio:.4f} (formula rho = {report.formula_rho:.4f})",
        "",
        f"| Level | Median cost s | Cost / real | rho | rho^2 (sun alone: {report.sun_share:.1%}) | Spearman "
        "| Seeds paired | Disagreements | Eligible |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in report.levels:
        verdict = "yes" if s.eligible else "no: " + "; ".join(s.reasons)
        lines.append(
            f"| {s.name} | {s.cost_s:.4f} | {s.cost_s / report.real_cost_s:.2f} | {s.rho:.4f} | {s.rho**2:.4f} | "
            f"{s.spearman:.4f} | {s.paired} | {s.disagreements} | {verdict} |"
        )
    lines.append("")
    if report.gate_failures:
        lines += ["**Gates failed, no levels chosen. No-go.**", *[f"- {f}" for f in report.gate_failures]]
        return "\n".join(lines)
    for names, (ratio, method) in sorted(report.option_ratios.items(), key=lambda kv: (kv[1][0] is None, kv[1][0])):
        shown = "no method can use this order" if ratio is None else f"{ratio:.4f} ({method})"
        lines.append(f"- {' + '.join(names)}: predicted variance / plain MC = {shown}")
    if report.chosen is None:
        lines.append("\n**No eligible level or pair. No-go.**")
        return "\n".join(lines)
    low, high = report.margin_interval
    holdout = "no method can use this order" if report.holdout_ratio is None else f"{report.holdout_ratio:.4f}"
    lines += [
        "",
        f"**Best: {' + '.join(report.chosen)}** ({report.chosen_method}), predicted variance / plain MC = "
        f"{report.chosen_ratio:.4f}, against the best baseline's {report.best_baseline_ratio:.4f}.",
        f"Bootstrap 90% interval on (chosen - best baseline): [{low:.4f}, {high:.4f}]; "
        f"chosen beats every baseline in {report.bootstrap_wins:.0%} of resamples.",
        f"Holdout (seeds {HOLDOUT.start:,}-{HOLDOUT.stop - 1:,}): {holdout}, against the best baseline's "
        f"{report.holdout_baseline:.4f}.",
        "",
        f"**{'Go' if report.go else 'No-go'}.**",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("formula_records", type=Path, help="the combined scan's records, which fit the formula")
    args = parser.parse_args()
    formula = Formula.from_records(read_records(args.formula_records))
    print(markdown(apply_rule(load_sweep(args.sweep_dir), formula)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
