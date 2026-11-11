# Choosing the two cheaper fidelity levels (issue #10)

ADR 0002 makes the cheaper levels map-thinned navigators, `splat-k<K>`: the same splat render, matched against only the K highest-scoring map keypoints. This file fixes how the two K values are chosen **before** the sweep runs, so the choice can't be tuned to whatever the data happens to show. The plan below is committed first; results are added under it afterwards, and any departure from the plan is stated there with its reason.

## Plan

**Status: draft, not yet run.** Nothing below is a measurement.

### What runs

- **Levels.** Six thinned candidates, K = 1,000, 2,000, 4,000, 8,000, 16,000 and 32,000 of the wide map's 70,726 keypoints (`docs/benchmarks/descent-profile.md`). Also the real navigator (`splat`), and the real navigator **a second time** in its own process (`splat`, written to a separate output directory). That second run is the ceiling: the best correlation any level could reach, given the GPU jitter between processes (ADR 0001).
- **Seeds.** Selection uses seeds 0–199. Holdout uses seeds 200–299, kept for one confirmation check and never used to choose. Every level runs every seed.
- **Scene.** Splat under sun 0 (`--sun-index 0`), the only sun with a trained splat.
- **Hardware.** One SLURM job on one **NVIDIA A40**. The GPU type was pinned on 2026-11-11, before any sweep run (`docs/carc.md`). All eight runs go one after another in that job, on the same GPU, so the levels' costs aren't split across nodes. If the job doesn't fit the debug partition's 1 h limit, it runs on the `gpu` partition instead, still on one A40.
- **Commands.** One `fidelis-plain-mc config/jezero.yaml data/jezero results/fidelity_sweep/<level> --fidelity <level> --sun-index 0 --first-seed 0 --samples 300` per level. Seeds 200–299 are split off afterwards when the records are read.
- **Rough cost (an estimate, not a measurement).** The real navigator took 0.32 s per descent on an A40. Thinned levels should cost less. So 8 levels × 300 seeds is at most about 13 minutes of descents, plus a minute or two per process to load the splat and extract the map: well under an hour on one GPU.

### What each level is scored on

- **Cost** `w`: the median `cost_s` over seeds 0–199.
- **Correlation** `ρ`: Pearson correlation of `error_m` with the real navigator's `error_m`, seed by seed, over the seeds where both got a fix. Pearson, because MFMC's variance depends on exactly that. Spearman is reported alongside, as a check that one far-off fix isn't carrying the result.
- **Status disagreements**: the number of seeds where the level and the real navigator differ on whether they got a fix at all. These are the tail signal from ADR 0002 §4, so they're reported even when they're rare.

### The rule

1. **Null-case gates first.** If any gate fails, stop and report; don't pick levels.
   - The ceiling correlation (real vs real rerun) must be at least 0.9. If it's lower, process-to-process jitter swamps the error, and no correlation in this sweep means anything.
   - Shuffling: correlating the real navigator with the cheapest level's errors in a shuffled seed order must give |ρ| < 0.2. That confirms the correlations come from pairing seeds, not from something else in the pipeline.
   - Every candidate's `map_keypoints` in its records must equal its K, so no level is secretly the whole map.
2. **Eligibility.** A candidate is eligible if:
   - ρ ≥ 0.8 × the ceiling,
   - it has at most 2 status disagreements in seeds 0–199 (1%),
   - its cost is at most 0.8 × the real navigator's.
3. **Choose the pair.** For every eligible pair K_a > K_b, give MXMCPy's MFMC optimizer the three levels' costs and covariance (seeds 0–199, real navigator on top) and a budget of 1,000 real-navigator descents. Pick the pair with the lowest predicted estimator variance relative to plain Monte Carlo at the same budget. A pair MXMCPy can't allocate (MFMC's ordering conditions fail) is skipped.
4. **Fallbacks.**
   - If only one candidate is eligible, the ladder gets one cheaper level. Report that; don't lower the bars.
   - If none is eligible, map thinning has failed as a fidelity knob. Report it, and fall back per ADR 0002 §6 (the descent keypoint cap).
5. **How firm the choice is.** Bootstrap seeds 0–199 (1,000 resamples) and report a 90% interval on the chosen pair's predicted variance ratio, plus how often the same pair wins.
6. **Holdout check.** Recompute the chosen pair's predicted variance ratio on seeds 200–299. If it falls outside the 90% bootstrap interval, flag the choice as possibly overfit. Report it, but don't re-select on holdout data.

### What this doesn't settle

- **The correlation is on navigation error, not on the failure indicator.** So far every descent has got a fix within a few metres, so the indicator is nearly constant. ADR 0002 §4 requires these levels to be re-checked on the smoothed failure indicator once the pilot study (#13) sets r.
- **The thresholds are judgement calls.** That covers 0.9, 0.2, 0.8 × the ceiling, 1%, 0.8 × the cost, and the 1,000-descent budget. They're written here so they can't move after the data is in.

## Results

**Run on 2026-11-11.** One debug job on one NVIDIA A40, with the command from the plan (job ID not recorded). All 8 levels ran all 300 seeds. The analysis is `uv run python -m fidelis.fidelity_sweep results/fidelity_sweep`, which applies the rule above as written (`src/fidelis/fidelity_sweep.py`).

### What the rule printed

The real navigator's median cost was 0.3228 s. Ceiling (real vs real rerun): ρ = 1.0000. Shuffled control: ρ = −0.124. All gates passed.

| Level | Median cost s | Cost / real | ρ | Spearman | Seeds paired | Disagreements | Eligible |
|---|---|---|---|---|---|---|---|
| splat-k1000 | 0.1049 | 0.33 | −0.095 | −0.017 | 189 | 11 | no: correlation, disagreements |
| splat-k2000 | 0.1009 | 0.31 | −0.071 | 0.056 | 200 | 0 | no: correlation |
| splat-k4000 | 0.0977 | 0.30 | 0.141 | 0.223 | 200 | 0 | no: correlation |
| splat-k8000 | 0.1043 | 0.32 | 0.405 | 0.333 | 200 | 0 | no: correlation |
| splat-k16000 | 0.1217 | 0.38 | 0.545 | 0.478 | 200 | 0 | no: correlation |
| splat-k32000 | 0.1671 | 0.52 | 0.832 | 0.819 | 200 | 0 | yes |

The rule's mechanical output: **splat-k32000 alone**, with predicted MFMC variance / plain MC = **1.332**. The bootstrap 90% interval is [1.261, 1.393], and the holdout gives 1.357, inside that interval.

### A gap in the rule

A ratio above 1 means MFMC with this level would be **worse** than plain Monte Carlo at the same budget: about 33% more variance. The rule picks the lowest ratio among eligible levels but never required it to be below 1. That was an oversight in writing it, not a threshold that moved. Read for its intent, the result is **no usable level**: map thinning fails as a fidelity knob under this rule. The k32000 "choice" shouldn't be used.

### Diagnostics (computed from the same records, not part of the rule)

- **The rerun is bit-identical.** All 300 seeds gave exactly the same error in both processes. On the A40, the splat render and the navigator are deterministic across processes. ADR 0001's jitter came from Mitsuba's mesh renderer and doesn't show here. So the ceiling of 1.0 is real, but it tests nothing.
- **The real navigator's errors are small and tight:** median 2.20 m, 90th percentile 2.89 m, maximum 4.1 m. Its median is 1,993 inliers. No seed came near r = 12 m.
- **The cheap levels' failures are degraded fixes, not wild ones.** k1000 had 19 no-fix seeds and 17 fixes beyond 12 m (maximum 36 m), with a median of 20 inliers. k2000 had 2 no-fix seeds and 9 fixes beyond 12 m, all between 12 and 19 m, with a median of 43 inliers.
- **Cost stops falling at about 0.10 s** (30% of the real navigator) for K ≤ 8,000. Below that, matching no longer dominates, and the render, camera model and keypoint extraction set the floor (`descent-profile.md`: about 0.07 s of non-matching stages on this GPU type).
- The real navigator's records give 70,749 map keypoints, not the 70,726 quoted in `descent-profile.md`. That's not investigated, and it doesn't affect any result above.

### What it means

- **Thinning re-randomizes the metre-level error instead of tracking it.** The real navigator's spread is a few tenths of a metre around 2.2 m. That spread comes from which of about 2,000 inliers survive RANSAC, not from scenarios being easy or hard. Change the map's keypoints and you redraw it. Correlation needs a shared cause, such as hard scenarios that hurt every level. These 300 descents don't have one: the real navigator never fails. This is ADR 0002 §4's bulk-versus-tail caveat, showing up as the main result.
- **The descent keypoint cap (ADR 0002 §6) would most likely hit the same wall.** It also changes which matches survive. That's an inference, not a measurement.
- **The deeper problem is the failure indicator.** With 0/300 failures at r = 12 m, the real navigator's indicator is constant, so P(fail) has nothing for MFMC to estimate. Whether cheap levels track the real one *in the tail* can only be tested once descents actually fail. That needs the pilot study (#13), r, and possibly harder scenarios.
