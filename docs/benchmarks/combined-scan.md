# Combined scan: sourced errors 2–4 together (issue #39)

The one pre-registered scan of dust haze (#37), focal-length error (#38) and attitude-knowledge error (#39), on top of the real CTX map (#36). ADR 0005 decision 4 replaced one scan per error with this one. This file fixes what will be measured **before** the run. Results go under the plan afterwards, and any departure from the plan is stated there with its reason.

**What it measures, worded with care.** The failure rate **under the scenario model's sourced ranges**: haze uniform over τ 0.5–1 (`dust-haze.md`), focal error uniform over the pre-calibration ±1% (`focal-error.md`), and attitude and misalignment errors uniform over their bounds (`attitude-error.md`). Two of those are labelled ranges, not flight values. So the result is not "the realistic failure rate" without that qualifier, and it isn't the certified bound either: that comes from the 10,000-descent run (#47, ADR 0004).

## Plan

**Status: pre-registered 2026-11-14, before any descent was run with all three errors** (committed at `2a252e9`). Nothing in this section is a measurement; the results are under it.

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

## Results

**Run on 2026-11-14.** SLURM job 14258755, debug partition, node b11-09, one NVIDIA A40, all three runs in sequence in one `sbatch` job, as planned. `COMPLETED` in 8 min 3 s (`sacct`). At 24 SUs a minute that is about 193 SUs **[Computed]**, at the bottom of the plan's 190–240 estimate. Every record names the A40 and carries config SHA-256 `0e7ccb36…`, which matches the committed `config/jezero.yaml` at `2a252e9`. All 300 seeds (60,000–60,299) ran in every run, on the CTX map (58,866 keypoints), under the realistic suns and the photo texture. Median cost per descent: 0.503 s for the scan, 0.447 s for the control, 0.504 s with no attitude error. **No departures from the plan.**

The numbers below come from the records with a throwaway analysis script (not committed), using `fidelis.plain_mc.read_records`, `fidelis.map_frame.horizontal_error_m` and `fidelis.simulation.believed_attitude`. "Error" is the map-frame horizontal error unless it says otherwise.

### 1. The headline

**0 failures in 300 at r = 40 m horizontal in the map frame: 0 no-fix, 0 beyond r. P(fail) = 0, 95% Wilson interval [0, 0.0126].** Offset removed: +1.89 m east, −1.49 m north. Every descent got a fix; the fewest inliers in any descent was 698. The worst was **17.45 m** (seed 60019: sun 1, 16.2° tilt, τ 0.96, focal error −0.77%), under half of r.

This is the rate **under the sourced ranges**: haze over τ 0.5–1, focal error over the pre-calibration ±1%, attitude and misalignment errors uniform over their bounds. It is not the flight rate, and it isn't the certified bound, which comes from #47.

### 2. The null case: the control is clean

0 failures in 300 (0 no-fix, 0 beyond r). Error median 4.29 m, 90th percentile 7.59 m, worst 9.55 m. The CTX map scan, on other seeds, gave a median of 4.37 m horizontal and a worst of 10.02 m (`ctx-map-scan.md`). So on the zero-error path the code since #36 behaves as before, and the scan is trusted.

### 3. By sun

| Sun | Az, el | Descents | Failures | Scan median | Scan 90th pct | Scan max | Control median | Control max |
|---|---|---|---|---|---|---|---|---|
| 0 | 257°, 40° | 55 | 0 | 5.18 m | 8.88 m | 14.95 m | 3.43 m | 6.49 m |
| 1 | 240°, 25° | 62 | 0 | 6.09 m | 9.37 m | 17.45 m | 6.55 m | 9.12 m |
| 2 | 240°, 55° | 58 | 0 | 4.85 m | 11.72 m | 16.31 m | 3.02 m | 6.56 m |
| 3 | 310°, 25° | 63 | 0 | 6.11 m | 11.17 m | 17.31 m | 6.68 m | 9.55 m |
| 4 | 310°, 55° | 62 | 0 | 3.91 m | 9.62 m | 10.91 m | 2.34 m | 6.57 m |
| All | | 300 | 0 | 5.34 m | 9.74 m | 17.45 m | 4.29 m | 9.55 m |

The low suns (1 and 3) were the worst in the control. In the scan their medians barely move, and even fall a little, while the high suns' medians rise by 1.6–1.8 m. The focal shift adds to whatever error a seed already has, and its direction is set by the tilt (`focal-error.md`), so it can cancel a low sun's steady offset as well as add to it.

### 4. Per seed, scan against control

The error grew in 189 of 300 seeds. Change: mean +1.23 m, median +0.55 m, 10th–90th percentile −2.04 to +5.40 m, range −5.96 to +13.24 m. Inliers moved little: median ratio 0.993. The rank correlation between a seed's error in the two runs is 0.40, down from 0.71 between the two maps in #36: the focal shift reorders which descents are worst.

### 5. Focal error, binned by |e|

Along-tilt part of the scan-minus-control shift, against −e·h·sin 2θ. The residual is everything that isn't the focal geometry: haze, attitude and matching effects together.

| abs(e) | Seeds | Scan median | Scan max | Predicted shift (median size) | Residual: median | 10th–90th pct | Median size |
|---|---|---|---|---|---|---|---|
| 0–0.25% | 64 | 4.00 m | 8.47 m | 0.71 m | +0.13 m | −0.93 to +0.81 m | 0.48 m |
| 0.25–0.5% | 68 | 4.55 m | 11.47 m | 2.49 m | −0.05 m | −0.71 to +0.82 m | 0.36 m |
| 0.5–0.75% | 72 | 5.66 m | 13.96 m | 4.18 m | +0.02 m | −0.82 to +0.88 m | 0.40 m |
| 0.75–1% | 96 | 6.19 m | 17.45 m | 6.01 m | −0.04 m | −0.82 to +0.99 m | 0.44 m |

- **The fix moves as the geometry says.** A straight-line fit of the along-tilt shift to −e·h·sin 2θ over all 300 has a slope of 1.002. The residual has a median of 0.00 m and a largest size of 2.39 m. The across-tilt shift is small too: median −0.12 m, range −4.31 to +3.53 m.
- **The residual doesn't grow with |e|**: its median size is 0.36–0.48 m in every bin. So there's no sign that the matcher reacts to the wrong camera: the focal error costs its geometry and nothing measurable on top.
- **The error grows with |e|**, as it should: median 4.00 m in the lowest bin, 6.19 m in the highest, and every error over 11.5 m sits in the top two bins.
- The 0.75–1% bin holds 96 seeds, not about 75. The edges are 0.25% wide and the draws are uniform, so this is draw scatter (a χ² test wasn't run).

### 6. Haze, binned by τ

| τ | Seeds | Inliers, scan / control (median) | Lowest ratio | Inliers, no-attitude / control (median) | Scan median | Scan max |
|---|---|---|---|---|---|---|
| 0.5–0.625 | 82 | 1.002 | 0.824 | 1.001 | 5.31 m | 16.31 m |
| 0.625–0.75 | 78 | 0.996 | 0.605 | 1.000 | 5.29 m | 16.75 m |
| 0.75–0.875 | 85 | 0.994 | 0.739 | 0.993 | 5.62 m | 17.31 m |
| 0.875–1 | 55 | 0.977 | 0.820 | 0.982 | 4.63 m | 17.45 m |

Inliers fall with τ, but only by about 2% at the top of the range (rank correlation between τ and the no-attitude/control inlier ratio: −0.16). No fix was lost, and the error doesn't rise with τ.

### 7. Attitude, scan against no-attitude, seed for seed

The total attitude rotation drawn had a median of 0.82° and a maximum of 1.77°.

| Rotation | Seeds | Change in error: mean | Median size | Largest size | Horizontal vector change: median | Largest |
|---|---|---|---|---|---|---|
| 0–0.5° | 52 | +0.06 m | 0.32 m | 2.63 m | 0.55 m | 2.67 m |
| 0.5–1° | 153 | +0.00 m | 0.32 m | 2.05 m | 0.57 m | 3.68 m |
| 1–1.5° | 92 | −0.05 m | 0.32 m | 2.84 m | 0.60 m | 2.88 m |
| Over 1.5° | 3 | +0.16 m | 0.36 m | 0.47 m | 0.37 m | 0.52 m |
| All | 300 | −0.00 m | 0.32 m | 2.84 m | 0.56 m | 3.68 m |

- **No bias.** The mean change is 0.00 m (10th–90th percentile −0.64 to +0.63 m).
- **No trend with size.** The median change is 0.32–0.36 m in every bin. The largest vector change (3.68 m, seed 60126) came from a 0.60° rotation. That is matching scatter, not geometry, as the prediction said it would be.
- **Inliers:** median ratio 0.998; per-seed changes from −276 to +117, on a median of about 1,330. The biggest swings left the error within about 1 m (seed 60230 lost 276 inliers, and its error went from 2.12 to 1.50 m).

### 8. Exploratory (labelled as such)

- **At 12 m:** 13 of 300 descents in the scan, 12 with no attitude error, 0 in the control.
- **Vertical error:** the focal error lifts or drops the fix, as e·h·cos 2θ says (`focal-error.md`). Scan: median +2.65 m, range −22.76 to +23.61 m; control: median +0.29 m, range −2.68 to +3.38 m. Height doesn't count against r (#46), but the 3D error is now 13.64 m median against 5.34 m horizontal.
- **Cost:** about 0.06 s more per descent with the errors on (0.503 against 0.447 s, medians), likeliest the haze's per-pixel work. Not profiled.

## Against the prediction

| Prediction (before the run) | Result | Held? |
|---|---|---|
| 0 of 300 at 40 m, no lost fixes in any run | 0 of 300 in all three runs, every descent fixed | Yes |
| Control looks like the CTX map scan (median about 4.4 m, worst about 10 m) | Median 4.29 m, worst 9.55 m | Yes |
| Focal residual within a few metres, no trend across the bins | Slope 1.002; residual median size 0.36–0.48 m in every bin, largest 2.39 m | Yes |
| Worst error under about 25 m | 17.45 m | Yes |
| Haze: inliers fall with τ, most near τ = 1; no lost fixes | Falls by about 2% at the top; no lost fixes | Yes, and smaller than expected |
| Attitude: mean change near zero | 0.00 m | Yes |
| Attitude: per-seed change under about 1.5 m **[Inference]** | Over 1.5 m in 12 of 300 seeds (4%); largest 2.84 m | **No**, as the smoke test already hinted |
| Surprise line: an attitude change over about 3 m | Error change: largest 2.84 m. Horizontal vector change: one seed of 300 at 3.68 m | The error-change line held; the vector change crossed it once |

**What the 1.5 m miss means.** The prediction got the mechanism right: no bias, and no trend with size. It got the scatter about twice too small. The **[Inference]** scaled the CPU fixture's 0.44 m by the pixel size, from 1.5 m to about 4.5 m. The real scatter has a 90th percentile of 0.90 m (error change) and a largest of 2.84 m, so the scaling under-called the tail. The fixture moved one axis at a time; here all three move at once. Either way it is under a tenth of r.

## What this means

The plan's "0 failures" branch applies. Under the sourced ranges, adding dust haze, a focal-length error and attitude errors on top of the real CTX map makes fixes less accurate and doesn't make the navigator fail. Almost all of the extra error is the focal error's geometry, predicted to within half a metre per seed; haze and attitude only add matching scatter. The worst of 300 is 17.45 m against a 40 m line. So P(fail) is under 1.26% at 95% confidence, under these ranges.

That leaves the realistic model with no failures to estimate, as ADR 0004 anticipated. The 10,000-descent bound (#47) inherits these ranges, and the multifidelity estimators are demonstrated on mean error instead. The largest single error source is now a labelled range, the pre-calibration focal error. A sourced post-calibration figure would likely shrink these errors, not grow them **[Inference]**: calibration exists to make the error smaller than the pre-calibration bound.
