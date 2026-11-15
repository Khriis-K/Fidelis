# Combined scan: sourced errors 2–4 together (issue #39)

The one pre-registered scan of dust haze (#37), focal-length error (#38) and attitude-knowledge error (#39), on top of the real CTX map (#36). ADR 0005 decision 4 replaced one scan per error with this one. This file fixes what will be measured **before** the run. Results go under the plan afterwards, and any departure from the plan is stated there with its reason.

**What it measures, worded with care.** The failure rate **under the scenario model's sourced ranges**: haze uniform over τ 0.5–1 (`dust-haze.md`), focal error uniform over the pre-calibration ±1% (`focal-error.md`), and attitude and misalignment errors uniform over their bounds (`attitude-error.md`). Two of those are labelled ranges, not flight values. So the result is not "the realistic failure rate" without that qualifier, and it isn't the certified bound either: that comes from the 10,000-descent run (#47, ADR 0004).

## Plan

**Status: pre-registered 2026-11-14, before any descent was run with all three errors.** Nothing in this section is a measurement.

### What runs

Three runs on the **same 300 seeds**, in one `sbatch` job, so they share a node and a GPU type. Each input has its own random stream (`src/fidelis/scenario.py`), so switching an error off leaves every other draw of a seed unchanged.

| Run | Flags | What it's for |
|---|---|---|
| **Scan** | none: `config/jezero.yaml` as is | The headline. CTX map, haze, focal error, attitude error |
| **Control** | `--clear-air --exact-camera --true-attitude` | The null case: the #36 model on these seeds. If it doesn't look like the CTX map scan, the new code broke something and the scan isn't trusted |
| **No attitude** | `--true-attitude` | Isolates the attitude error seed for seed. In the scan it is hidden under the focal error (`attitude-error.md`) |

- **Seeds.** 60,000–60,299. Taken so far: 0–299, 10,000–10,299, 20,000–20,299, 30,000–30,005, 40,000–40,299, 50,000–50,008 (50,006–50,008 by the #39 smoke test, `attitude-error.md`). Any further smoke tests use seeds below 60,000, never these.
- **Navigator and renderer.** The real navigator on the mesh renderer (`--fidelity mesh`), matching against the CTX map. `--sun-index` is left off, so each seed draws one of the five realistic suns.
- **Hardware.** One NVIDIA A40 on `debug` (`docs/carc.md`), as the CTX map scan, so its costs compare.
- **Command** (the user runs it, from the clone on `/project2`):

      sbatch --account=irahbari_1147 --partition=debug --gres=gpu:a40:1 -c 8 --mem=32G -t 30 -J combined_scan -o results/combined_scan_%j.log --wrap "module purge && module load ver/2607 gcc/15.3.0 cuda/13.0.3 && R='uv run --no-sync fidelis-plain-mc config/jezero.yaml data/jezero' && S='--fidelity mesh --first-seed 60000 --samples 300' && \$R results/combined_scan/scan \$S && \$R results/combined_scan/control \$S --clear-air --exact-camera --true-attitude && \$R results/combined_scan/no_attitude \$S --true-attitude"

- **Rough cost (an estimate, not a measurement).** The CTX map scan ran 2 × 300 descents in 5 min 23 s, about 129 SUs. Three runs should take about 8–10 minutes, so about 190–240 SUs at 24 SUs a minute. The haze adds per-pixel work to each image; how much on an A40 isn't measured. The 30-minute limit caps the job at 720 SUs.
- **If the job can't run all three,** the scan comes first, then the control. Without the control the scan is reported as untrusted against a null case; without the no-attitude run, the attitude prediction is reported as untested.

### What counts as a failure (fixed by ADR 0003 and #46, not chosen here)

- **r = 40 m, horizontal, in the map frame.** A descent fails if it gets no fix, or its fix is more than 40 m from the truth horizontally after the map offset (+1.89 m east, −1.49 m north) is removed.
- **No-fix and beyond-r failures are counted separately.**
- r is not tuned, whatever the count turns out to be.

### What gets reported

1. **The headline:** failures out of 300 in the scan, no-fix and beyond-r separately, with the 95% Wilson interval `fidelis-plain-mc` prints. Quoted with its ranges, as above.
2. **The null case:** the control's failures, and its map-frame horizontal error (median, 90th percentile, max), against the CTX map scan's on other seeds (median 4.37 m horizontal, worst 10.02 m; `ctx-map-scan.md`).
3. **By sun:** failures, and the scan's error median, 90th percentile and maximum, for each of the five suns.
4. **Per seed, scan against control:** the change in horizontal error (mean, median, 10th–90th percentile), the number of seeds whose error grew, and the change in inliers.
5. **Focal error, binned by |e|** (carried over from #38): 0–0.25%, 0.25–0.5%, 0.5–0.75%, 0.75–1%, about 75 seeds each. For each bin, the scan's horizontal error, and the along-tilt part of the scan-minus-control shift against the geometric term −e·h·sin 2θ. The residual is everything that isn't the focal geometry: haze, attitude and matching effects together. This scan can't split haze from focal-error matching effects; that would need a fourth run, not planned.
6. **Haze, binned by τ:** 0.5–0.625, 0.625–0.75, 0.75–0.875, 0.875–1. Inliers, as a ratio to the control's for the same seed, and the scan's horizontal error. Haze attacks contrast, so inliers are where it should show.
7. **Attitude, scan against no-attitude, seed for seed:** the change in horizontal error (mean, median, 10th–90th percentile, largest), against the total attitude rotation drawn (binned 0–0.5°, 0.5–1°, 1–1.5°, above 1.5°), and the change in inliers.
8. **Exploratory, labelled as such:** failures and errors at 12 m; vertical error, since the focal error lifts the fix by about e·h·cos 2θ (`focal-error.md`; height doesn't count against r); cost per descent.

### Prediction from the sources

Each error's prediction is in its own write-up, made before its build. Put together:

- **Failures at 40 m: 0 of 300**, so the 95% Wilson interval would be [0, 1.26%]. No lost fixes in any run.
- **The control looks like the CTX map scan:** median horizontal error about 4.4 m, worst about 10 m, 0 failures. Other seeds, so not identical.
- **The focal error is the largest shift.** −e·h·sin 2θ over the scenario's draws **[Computed]**, 10⁶ draws of the formula: median 2.67 m, 90th percentile 8.24 m, 99th percentile 11.93 m, max 14.32 m. The binned residual against that term should stay within a few metres per seed, with no trend across the bins. Seed 50,004 in the #38 smoke test was 1.8–2.9 m off it, which sets the scale.
- **The scan's worst horizontal error under about 25 m.** The CTX map scan's worst (10.02 m) plus the focal term's largest (14.3 m) lined up exactly is 24.3 m; root sum of squares is 17.5 m **[Computed]**. Haze is predicted to add no shift, only fewer inliers; attitude under about 1.5 m per seed.
- **Haze:** inliers fall with τ, most at τ near 1. No lost fixes (`dust-haze.md`).
- **Attitude:** mean change near zero, per-seed change under about 1.5 m **[Inference]**, inliers barely moving (`attitude-error.md`). Kept as written although one of the three smoke seeds, run after the prediction, already moved 1.68 m: the prediction isn't revised to fit a smoke test. The 3 m surprise line below stands.

**A surprise would be** any lost fix; a horizontal error over about 26 m; a focal residual that grows with |e| (matching reacting to the wrong camera, not geometry); or an attitude change over about 3 m on any seed. Each would be reported against the prediction it breaks.

### What each outcome means

- **0 failures (predicted).** Under the sourced ranges, the four sourced errors don't push the navigator past 40 m in 300 descents: P(fail) under 1.26% at 95% confidence. The pitch then rests on the lighting-robustness framing (ADR 0005) and the 10,000-descent bound (#47), which inherits these ranges.
- **A few failures.** Report which kind, and where they cluster: by |e| first, then sun, τ and attitude size. Nothing is tuned in response. Beyond-r failures at the top of the focal range would be read against the pre-calibration label: they say what an uncalibrated camera does, not what the flight camera did.
- **Many failures.** Check the control first. If it's clean, read the errors against each bin above before any conclusion; a failure rate well above the prediction means one of the three models (or its interaction with the matcher) behaves unlike its prediction, and that gets found before #47 runs.
- **The control isn't clean.** The scan isn't trusted. The code changed since #36 (#37, #38, #46, #39), and something on the zero-error path moved. Fix that first, then pre-register the scan again on fresh seeds.

### Known before the run

- **No descent has been run with all three errors on the full map.** Laptop smoke tests on a thinned map (`mesh-k8000`) checked each error's plumbing alone; they aren't results.
- **The control is the #36 model, but not the #36 code.** Since then the failure line moved to horizontal error (#46) and three error paths were added, each with a test that zero error reproduces the old output. The control is where that shows on real renders.
