"""The pre-registered rule for choosing the two cheaper fidelity levels (docs/benchmarks/fidelity-sweep.md).

Reads the sweep's records, one directory per level ("splat", "splat-k<K>", and "splat-rerun",
the real navigator run a second time), and applies the rule exactly as committed: null-case
gates, then per-level eligibility, then the pair MXMCPy's MFMC optimizer predicts saves the
most. Selection uses seeds 0-199 only; 200-299 are a holdout for one confirmation check.

    uv run python -m fidelis.fidelity_sweep results/fidelity_sweep
"""

import argparse
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from mxmc import Optimizer
from scipy.stats import spearmanr

from fidelis.plain_mc import read_records
from fidelis.simulation import SimulationRecord, parse_fidelity

SELECTION = range(0, 200)
HOLDOUT = range(200, 300)
REAL, RERUN = "splat", "splat-rerun"

# The plan's thresholds, fixed before the sweep ran.
CEILING_MIN = 0.9  # real vs real rerun
SHUFFLE_MAX = 0.2  # |rho| of real vs the cheapest level in shuffled seed order
RHO_FRACTION = 0.8  # of the ceiling
MAX_DISAGREEMENTS = 2  # fix-or-not disagreements with the real navigator, selection seeds
COST_FRACTION = 0.8  # of the real navigator's median cost
BUDGET = 1000  # real-navigator descents
BOOTSTRAP = 1000

Records = dict[int, SimulationRecord]


@dataclass
class LevelStats:
    name: str
    k: int
    cost_s: float
    rho: float  # Pearson of error_m with the real navigator's, seeds where both got a fix
    spearman: float
    paired: int  # how many seeds that was
    disagreements: int
    eligible: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass
class Report:
    real_cost_s: float
    ceiling: float
    shuffle_rho: float
    gate_failures: list[str]
    levels: list[LevelStats]
    pair_ratios: dict[tuple[str, ...], float | None] = field(default_factory=dict)
    chosen: tuple[str, ...] | None = None
    chosen_ratio: float | None = None
    bootstrap_interval: tuple[float, float] | None = None
    bootstrap_wins: float | None = None  # fraction of resamples where the chosen set still wins
    holdout_ratio: float | None = None
    holdout_flagged: bool = False

    @property
    def gates_passed(self) -> bool:
        return not self.gate_failures


def load_sweep(sweep_dir: Path) -> dict[str, Records]:
    """Each level's records by seed, keyed by its directory name."""
    return {
        d.name: {r.seed: r for r in read_records(d / "records.jsonl")}
        for d in sorted(Path(sweep_dir).iterdir())
        if (d / "records.jsonl").exists()
    }


def paired_errors(levels: list[Records], seeds) -> np.ndarray:
    """errors[i, j]: level i's error_m on the j-th seed where every level got a fix. Seeds may repeat."""
    kept = [s for s in seeds if all(level[s].status == "ok" for level in levels)]
    return np.array([[level[s].error_m for s in kept] for level in levels])


def median_cost(level: Records, seeds) -> float:
    return float(np.median([level[s].cost_s for s in seeds]))


def variance_ratio(costs: np.ndarray, covariance: np.ndarray, budget: float = BUDGET) -> float | None:
    """MXMCPy's MFMC variance at `budget` real-model runs, over plain MC's at the same budget.

    None when MFMC can't use the models in this order (its correlation ordering fails).
    """
    try:
        result = Optimizer(costs, covariance=covariance).optimize("mfmc", budget * costs[0])
    except Exception:  # MXMCPy raises its own InconsistentModelError and friends
        return None
    plain_mc = covariance[0, 0] / budget
    ratio = float(np.squeeze(result.variance)) / plain_mc
    return ratio if np.isfinite(ratio) else None


def set_ratio(sweep: dict[str, Records], names: tuple[str, ...], seeds) -> float | None:
    levels = [sweep[REAL]] + [sweep[n] for n in names]
    errors = paired_errors(levels, seeds)
    if errors.shape[1] < 3:
        return None
    costs = np.array([median_cost(level, seeds) for level in levels])
    return variance_ratio(costs, np.atleast_2d(np.cov(errors)))


def level_stats(sweep: dict[str, Records], name: str, seeds) -> LevelStats:
    real, level = sweep[REAL], sweep[name]
    errors = paired_errors([real, level], seeds)
    return LevelStats(
        name=name,
        k=parse_fidelity(name)[1],
        cost_s=median_cost(level, seeds),
        rho=float(np.corrcoef(errors)[0, 1]),
        spearman=float(spearmanr(errors[0], errors[1]).statistic),
        paired=errors.shape[1],
        disagreements=sum((real[s].status == "ok") != (level[s].status == "ok") for s in seeds),
    )


def best(sweep: dict[str, Records], options: list[tuple[str, ...]], seeds) -> tuple[tuple[str, ...] | None, dict]:
    ratios = {names: set_ratio(sweep, names, seeds) for names in options}
    usable = {names: r for names, r in ratios.items() if r is not None}
    return (min(usable, key=usable.get) if usable else None), ratios


def apply_rule(sweep: dict[str, Records], seed: int = 0) -> Report:
    selection = list(SELECTION)
    candidates = sorted((n for n in sweep if parse_fidelity(n)[1] is not None), key=lambda n: parse_fidelity(n)[1])
    real_cost = median_cost(sweep[REAL], selection)
    ceiling = float(np.corrcoef(paired_errors([sweep[REAL], sweep[RERUN]], selection))[0, 1])

    # Gate: shuffling the cheapest level's seeds must destroy the correlation.
    rng = np.random.default_rng(seed)
    errors = paired_errors([sweep[REAL], sweep[candidates[0]]], selection)
    shuffle_rho = float(np.corrcoef(errors[0], rng.permutation(errors[1]))[0, 1])

    failures = []
    if not ceiling >= CEILING_MIN:
        failures.append(f"ceiling correlation {ceiling:.3f} < {CEILING_MIN}")
    if not abs(shuffle_rho) < SHUFFLE_MAX:
        failures.append(f"shuffled correlation {shuffle_rho:.3f} not within ±{SHUFFLE_MAX}")
    for name in candidates:
        k = parse_fidelity(name)[1]
        wrong = {r.map_keypoints for r in sweep[name].values()} - {k}
        if wrong:
            failures.append(f"{name}: map_keypoints {sorted(wrong)} in its records, not {k}")

    stats = [level_stats(sweep, n, selection) for n in candidates]
    for s in stats:
        if s.rho < RHO_FRACTION * ceiling:
            s.reasons.append(f"correlation {s.rho:.3f} < {RHO_FRACTION} x ceiling")
        if s.disagreements > MAX_DISAGREEMENTS:
            s.reasons.append(f"{s.disagreements} fix-or-not disagreements > {MAX_DISAGREEMENTS}")
        if s.cost_s > COST_FRACTION * real_cost:
            s.reasons.append(f"cost {s.cost_s:.3f} s > {COST_FRACTION} x real")
        s.eligible = not s.reasons

    report = Report(real_cost, ceiling, shuffle_rho, failures, stats)
    if failures:
        return report

    # Higher K first: that's the next model down from the real navigator.
    eligible = sorted((s.name for s in stats if s.eligible), key=lambda n: -parse_fidelity(n)[1])
    options = list(itertools.combinations(eligible, 2)) or [(n,) for n in eligible]
    if not options:
        return report
    report.chosen, report.pair_ratios = best(sweep, options, selection)
    if report.chosen is None:
        return report
    report.chosen_ratio = report.pair_ratios[report.chosen]

    boot_ratios, wins = [], 0
    for _ in range(BOOTSTRAP):
        resample = list(rng.choice(selection, size=len(selection), replace=True))
        winner, ratios = best(sweep, options, resample)
        wins += winner == report.chosen
        if ratios[report.chosen] is not None:
            boot_ratios.append(ratios[report.chosen])
    report.bootstrap_interval = tuple(float(q) for q in np.percentile(boot_ratios, [5, 95]))
    report.bootstrap_wins = wins / BOOTSTRAP

    report.holdout_ratio = set_ratio(sweep, report.chosen, list(HOLDOUT))
    low, high = report.bootstrap_interval
    report.holdout_flagged = report.holdout_ratio is None or not low <= report.holdout_ratio <= high
    return report


def markdown(report: Report) -> str:
    lines = [
        f"Real navigator: median cost {report.real_cost_s:.4f} s.",
        f"Ceiling (real vs real rerun): rho = {report.ceiling:.4f}. Shuffled control: rho = {report.shuffle_rho:.4f}.",
        "",
        "| Level | Median cost s | Cost / real | rho | Spearman | Seeds paired | Disagreements | Eligible |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in report.levels:
        verdict = "yes" if s.eligible else "no: " + "; ".join(s.reasons)
        lines.append(
            f"| {s.name} | {s.cost_s:.4f} | {s.cost_s / report.real_cost_s:.2f} | {s.rho:.4f} | "
            f"{s.spearman:.4f} | {s.paired} | {s.disagreements} | {verdict} |"
        )
    lines.append("")
    if report.gate_failures:
        lines += ["**Gates failed, no levels chosen:**", *[f"- {f}" for f in report.gate_failures]]
        return "\n".join(lines)
    for names, ratio in sorted(report.pair_ratios.items(), key=lambda kv: (kv[1] is None, kv[1])):
        shown = "MFMC can't use this order" if ratio is None else f"{ratio:.4f}"
        lines.append(f"- {' + '.join(names)}: predicted variance / plain MC = {shown}")
    if report.chosen is None:
        lines.append("\n**No usable level or pair: map thinning fails as a fidelity knob (ADR 0002 §6).**")
        return "\n".join(lines)
    low, high = report.bootstrap_interval
    holdout = "MFMC can't use this order" if report.holdout_ratio is None else f"{report.holdout_ratio:.4f}"
    verdict = "outside the bootstrap interval: flagged as possibly overfit" if report.holdout_flagged else "inside"
    lines += [
        "",
        f"**Chosen: {' + '.join(report.chosen)}**, predicted variance / plain MC = {report.chosen_ratio:.4f}.",
        f"Bootstrap 90% interval [{low:.4f}, {high:.4f}]; same choice in {report.bootstrap_wins:.0%} of resamples.",
        f"Holdout (seeds 200-299): {holdout}, {verdict}.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    print(markdown(apply_rule(load_sweep(args.sweep_dir))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
