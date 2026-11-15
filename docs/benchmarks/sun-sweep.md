# Multifidelity go/no-go with the sun varying (issue #48)

ADR 0005 decision 2 makes multifidelity one bounded experiment. This file fixes the experiment **before** it runs: the levels, the seeds, the three baselines, and the go/no-go rule, so none of them can be tuned to the data. Results go under the plan afterwards, and any departure from the plan is stated there with its reason.

## Plan

**Status: committed before any run (`36a4dd7`), then step 0 ran: no-go (see Results).** Nothing in the Plan or Prediction sections is a measurement, except the numbers marked as computed from existing records.

### Why it runs again, and what changed since the first sweep

- **The first sweep ran under one sun, on the splat, against the old map** (`fidelity-sweep.md`). No cheap level was usable: the best, k32000 (ρ = 0.832), would have given 1.33× plain MC's variance. Its diagnosis was that correlation needs a shared cause, and those descents had none.
- **The realistic model now has two shared causes.** The sun, and since #38, the focal-length error, whose horizontal shift −e·h·sin 2θ the fix follows with slope 1.002 (`combined-scan.md`). Every level that matches under the same scenario feels both.
- **A shared cause that's known from the inputs can be used for free.** The focal shift and a per-sun offset are computed from the scenario's own inputs, with no render. A cheap navigator that correlates only through them is doing what a formula does at zero cost. So this sweep adds a third baseline to the two #48 named: a control variate on that formula.

### What runs

- **Levels.** The real navigator on the mesh renderer (`mesh`), the same again in its own process (`mesh-rerun`, the ceiling), and six thinned levels, `mesh-k<K>` for K = 1,000, 2,000, 4,000, 8,000, 16,000 and 32,000 of the CTX map's 58,866 keypoints. These are the first sweep's K values, so the two compare. **Mesh, not splat:** the splat exists for sun 0 only, and #20 is closed (ADR 0005).
- **Scenario.** The realistic model as the combined scan ran it: the five realistic suns drawn per seed (no `--sun-index`), the CTX map, dust haze, focal error and attitude error all on, the photo texture.
- **Seeds.** Selection: 70,000–70,199. Holdout: 70,200–70,299, used for one confirmation check, never to choose. Step 0 below uses 70,300–70,329. All are fresh (the handoff lists 0–60,299 in ranges as spent).
- **The formula is fitted on other seeds.** It is fitted once on the combined scan's 300 records (seeds 60,000–60,299, `results/combined_scan/scan/records.jsonl`) and then frozen. So its correlation on the sweep's seeds isn't fitted to them. That's valid only while the sweep runs the combined scan's config: every record must carry config SHA-256 `0e7ccb36…`, as the combined scan's do.
- **Hardware.** One SLURM job on one NVIDIA A40, all eight runs in sequence, as the first sweep and the combined scan ran. The rule refuses records from more than one GPU model.

### Step 0: the cost floor, before the sweep

A cheap level can't beat the formula if it costs too much, however well it correlates. For MFMC the predicted variance ratio is at least the cheapest level's cost ratio w. With w₀ = 1 and ρ₀ = 1, the MFMC ratio (Σᵢ √(wᵢ(ρᵢ² − ρᵢ₊₁²)))² is at least Σᵢ wᵢ(ρᵢ² − ρᵢ₊₁²) ≥ w_min. For ACV the bound isn't derived here. A spot check through MXMCPy, at w = 0.1, 0.2 and 0.3 and ρ = 0.95, 0.99 and 0.999, gave ACV-MF, ACV-IS and ACV-KL ratios equal to MFMC's for one cheap level, and higher for two, all above w **[Computed]**. That's a check, not a proof. The formula's control variate gave 0.20–0.22 (computed below).

- **Run:** `mesh` and `mesh-k1000` on seeds 70,300–70,329, one A40, debug partition:

      sbatch --account=irahbari_1147 --partition=debug --gres=gpu:a40:1 -c 8 --mem=32G -t 10 -J cost_floor -o results/cost_floor_%j.log --wrap "module purge && module load ver/2607 gcc/15.3.0 cuda/13.0.3 && R='uv run --no-sync fidelis-plain-mc config/jezero.yaml data/jezero' && S='--first-seed 70300 --samples 30' && \$R results/cost_floor/mesh \$S --fidelity mesh && \$R results/cost_floor/mesh-k1000 \$S --fidelity mesh-k1000"

- **Rule:** if k1000's median cost is at least **0.22** × `mesh`'s, no level can beat the formula under MFMC, and the result is **no-go without the sweep**. 0.22 is the upper end of the formula's measured ratio, so a level just under it can't slip through on the formula's sampling noise. The result is stated as a cost-floor no-go, with the bound and the ACV spot check above, not as a measured ρ.
- **Why a step the issue didn't ask for.** #48 asks for the sweep, and step 0 can end the experiment before it. It's here because the bound makes the sweep's outcome predictable from one cost, and the sweep spends about 5–9 times step 0's SUs. If you'd rather run the sweep regardless, skip step 0: the rule below doesn't depend on it.
- **Otherwise** the sweep runs as below.
- **Rough cost (an estimate, not a measurement):** 60 descents at under 0.5 s, plus two scene loads, about 2 minutes, about 50 SUs.

### The sweep's commands

One job, the combined scan's module line:

    sbatch --account=irahbari_1147 --partition=debug --gres=gpu:a40:1 -c 8 --mem=32G -t 40 -J sun_sweep -o results/sun_sweep_%j.log --wrap "module purge && module load ver/2607 gcc/15.3.0 cuda/13.0.3 && R='uv run --no-sync fidelis-plain-mc config/jezero.yaml data/jezero' && S='--first-seed 70000 --samples 300' && \$R results/sun_sweep/mesh \$S --fidelity mesh && \$R results/sun_sweep/mesh-rerun \$S --fidelity mesh && for k in 32000 16000 8000 4000 2000 1000; do \$R results/sun_sweep/mesh-k\$k \$S --fidelity mesh-k\$k; done"

Analysis: `uv run python -m fidelis.sun_sweep results/sun_sweep results/combined_scan/scan/records.jsonl` (`src/fidelis/sun_sweep.py`).

- **Rough cost (an estimate, not a measurement).** The real navigator ran at 0.503 s per descent in the combined scan, so the two real runs are about 5 minutes. The thinned levels are unmeasured on the mesh. At 0.15–0.40 s that's another 4.5–12 minutes. With loading, about 11–19 minutes: **about 260–460 SUs** at 24 a minute. The 40-minute limit caps it at 960 SUs.

### What each level is scored on

The first sweep's measures, on the **map-frame horizontal error** (the 40 m line's measure, #46), not the 3D `error_m` the first sweep used:

- **Cost** w: median `cost_s` over the selection seeds.
- **Correlation** ρ: Pearson, against the real navigator, on seeds where both got a fix. Spearman alongside.
- **Status disagreements:** seeds where the level and the real navigator differ on getting a fix.

### The three baselines, at equal cost

All predicted variance over plain MC's at the same budget, on the real navigator's selection-seed errors:

- **Plain MC:** 1.
- **Sun-stratified MC,** proportional allocation: the mean of the five per-sun variances over the total variance. The config draws the five suns uniformly, so each weighs 1/5, which a test pins. The sun alone explains 1 minus this ratio, reported next to every level's ρ.
- **Control variate on the formula:** 1 − ρ_f², where ρ_f is the formula's correlation with the real navigator. The formula predicts each seed's east-north error as a per-sun offset plus −e·h·sin 2θ along the tilt azimuth, fitted by least squares, and takes its length. Its mean over the scenario distribution is known to any precision from scenario draws alone, with no render, so it counts as free.

### The rule

1. **Gates.** Any failure stops the rule and gives no-go.
   - Ceiling (real vs real rerun) at least 0.9.
   - Shuffled control: |ρ| under 0.2 between the real navigator and the cheapest level in shuffled seed order.
   - Every level's `map_keypoints` equals its K.
   - One GPU model and one config SHA across every record.
   - Haze, focal error and attitude error drawn (non-zero) in every record. The config SHA can't show this: `--clear-air`, `--exact-camera` and `--true-attitude` leave it unchanged, and the combined scan's control carries the same SHA as the scan.
2. **Eligibility,** as in the first sweep: ρ ≥ 0.8 × the ceiling, at most 2 status disagreements in the selection seeds, and cost at most 0.8 × the real navigator's.
3. **Options:** every eligible level alone, and every eligible pair (higher K nearer the real navigator). Each option gets MXMCPy's predicted ratio for MFMC and for ACV-MF at a budget of 1,000 real-navigator descents, and keeps the lower. The best option overall is the candidate.
4. **Go, only if all three hold.** This closes the first sweep's gap, which never required a ratio under 1.
   - The candidate's ratio beats the best baseline: lower than plain MC, sun-stratified MC and the formula's control variate.
   - It's firm: over 1,000 bootstrap resamples of the selection seeds, the 90% interval on (the candidate's ratio − the best baseline's ratio, both recomputed on the resample) lies entirely below 0. A resample where the candidate's method can't allocate counts as a loss.
   - It holds out of sample: on the holdout seeds, the candidate's ratio is below the best baseline's.
5. **On no-go,** M5 shrinks to the write-up (ADR 0005). Which baseline won, and by how much, is reported either way.

## Prediction

Computed from the combined scan's 300 records (seeds 60,000–60,299). These are existing records, not the sweep's.

- **The sun alone explains about 4% of the error's variance** under the realistic model: stratified ratio 0.958 over all 300 **[Computed]**. With sourced errors 2–4 off (the control), it explained 55% **[Computed]**. So the sun was the shared cause #48 was opened for, but the focal error now swamps it.
- **What that implies for ρ.** A level that shared only the sun with the real navigator could correlate at most about √0.04 ≈ 0.2. So any level that clears the eligibility bar (ρ ≥ 0.8 × the ceiling) does it through something other than the sun, most likely the focal shift every level shares.
- **The formula explains about 80%.** Fitted on one half of the 300 and scored on the other: ρ_f = 0.884 and 0.892, control-variate ratio 0.219 and 0.204 **[Computed]**. On the sweep's seeds, expect about the same, since the config is the same.
- **What that demands of a cheap level.** Under MFMC with one cheap level, the ratio is (√(1 − ρ²) + ρ√w)². To beat 0.20 it needs, for example, ρ = 0.95 at under 2.0% of the real navigator's cost, ρ = 0.98 at under 6.4%, or ρ = 0.99 at just under 9.6% **[Computed]**.
- **What we know of the cost floor.** On the splat, cost stopped falling at about 30% of the real navigator, with about 0.07 s of non-matching stages on an A40 (`fidelity-sweep.md`, `descent-profile.md`). The mesh render costs more than the splat's (0.234 against 0.184 s per descent on an L40S, `splat.md`). The real navigator's mesh descent here is 0.503 s with haze. **[Inference]** The mesh floor is likely 20–30% of the real navigator.
- **Prediction: no-go**, likely at step 0 on cost alone. A go would need a thinned navigator under about 10% of the real one's cost and correlating at 0.99, which no level has come near on either renderer. **[Inference]** The sources don't cover navigator-fidelity correlation, so this rests on our own records only.

## Results

**Step 0 ran on 2026-11-14.** SLURM job 14259865, debug partition, node b11-09, one NVIDIA A40, both runs in one job as planned. `COMPLETED` in 56 s (`sacct`), about 22 SUs at 24 a minute **[Computed]**, under the plan's estimate of 50. Every record names the A40 and carries config SHA-256 `0e7ccb36…`, the combined scan's. Seeds 70,300–70,329 ran in both levels. Haze, focal error and attitude error were drawn in every record. `mesh` matched 58,866 map keypoints and `mesh-k1000` matched 1,000. **No departures from the plan.**

### The rule's answer

| Level | Median cost s | Min | Max |
|---|---|---|---|
| mesh | 0.5134 | 0.4453 | 0.6735 |
| mesh-k1000 | 0.3140 | 0.2865 | 0.4562 |

**Cost ratio 0.612,** against the bar of 0.22. Bootstrapping the 30 seeds of each level (5,000 resamples) gives a 90% interval of [0.580, 0.631] on the ratio of medians **[Computed]**. That is nearly three times the bar.

**No-go, from the cost floor.** Under MFMC no level can get its variance ratio below its cost ratio, and the cheapest level costs 61% of the real navigator. The free formula's control variate gives 0.20–0.22. So no thinned level can beat it, however well it correlates. The ACV case rests on the spot check in step 0, not a proof. The full sweep doesn't run. Per ADR 0005 decision 2, M5 shrinks to writing this up.

### Prediction against result

| Predicted | Result | Held? |
|---|---|---|
| No-go, likely at step 0 on cost alone | No-go at step 0 | Yes |
| **[Inference]** The mesh floor is likely 20–30% of the real navigator | 61% | **No**, about twice as high |

The verdict held, and the cost estimate was wrong. The floor came from the splat. On the splat, matching was 75% of a descent and the render 2% (`descent-profile.md`). **[Inference]** On the mesh with haze, the stages that don't depend on the map (render, haze, camera model, descent keypoints) take about 0.3 s, so cutting the map saves at most about 0.2 s. The mesh renderer still hasn't been profiled, so the split isn't measured. It doesn't change the verdict: even a level costing only those stages would sit near 0.6.

### Exploratory, not part of the rule

- **The cheapest level fails often.** `mesh-k1000` got a fix on 21 of 30 seeds (6 `no_solution`, 3 `too_few_matches`), with a median of 24 inliers. The real navigator got a fix on all 30, horizontal error median 3.97 m, worst 13.65 m.
- **On the 21 paired seeds, ρ = 0.67.** At that correlation and cost, the MFMC ratio would be about (√(1 − 0.45) + 0.67·√0.61)² ≈ 1.6, worse than plain MC **[Computed]**. 21 seeds is too few to rely on.
- **The formula is a real saving on its own.** A control variate on it would bring the variance on mean horizontal error to about 0.2 × plain MC's (computed on combined-scan records, `combined-scan.md` seeds). The formula only reads the scenario's inputs, so the saving costs nothing. That is classic variance reduction, not multifidelity. Whether #21 should show it is a separate decision.

## What this doesn't settle

- **A cheap level combined with the formula** (a three-model estimator that uses the formula as one model) isn't tested. The cheap level would then only need to explain what the formula misses. That's a separate question, asked only if someone wants multifidelity beyond this go/no-go.
- **The thresholds are the first sweep's judgement calls,** plus the step-0 bar of 0.20, the lower end of the formula's measured control-variate ratio.
- **The QoI is the mean horizontal error** (ADR 0004 decision 2). Nothing here speaks to P(fail), which is 0 in every scan.
