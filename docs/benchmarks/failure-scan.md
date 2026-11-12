# Failure scan: does the real navigator ever fail? (before the pilot, #13)

The fidelity sweep (`docs/benchmarks/fidelity-sweep.md`) saw 0 failures in 300 descents, all under the landing-day sun. A failure-probability estimate needs failures, so this scan asks whether, and where, the real navigator fails. This file fixes the failure line and the analysis **before** the run, so neither can be tuned to the data. Results go under the plan afterwards, and any departure from the plan is stated there with its reason.

## Plan

**Status: pre-registered 2026-11-11, before the run.** Nothing in this section is a measurement; the results are under "Results".

### What runs

- **Navigator and renderer.** The real navigator on the mesh renderer (`--fidelity mesh`), the reference fidelity.
- **Seeds.** 10,000–10,299 (300 descents). They don't overlap any seed used so far (0–299).
- **Scenario.** Drawn from `config/jezero.yaml` as is. `--sun-index` is left off, so each seed draws one of the five config suns, including the two 25° suns. Tilt, blur, noise and altitude are drawn as usual. The navigator still gets the true attitude (`src/fidelis/simulation.py`).
- **Hardware.** One debug job on one NVIDIA A40 (`docs/carc.md`). The mesh renderer hasn't run on an A40 before. If it fails there, that's recorded, and the scan isn't moved to another GPU type without saying so here.
- **Command.**

      srun --account=irahbari_1147 --partition=debug --gres=gpu:a40:1 -c 8 --mem 32G -t 30 bash -c "module purge && module load ver/2607 gcc/15.3.0 cuda/13.0.3 && uv run --no-sync fidelis-plain-mc config/jezero.yaml data/jezero results/failure_scan --fidelity mesh --first-seed 10000 --samples 300"

- **Rough cost (an estimate, not a measurement).** Mesh descents took about 0.4–0.6 s each on the laptop GPU (`docs/benchmarks/plain-mc.md`), so roughly 3 minutes plus scene loading.

### What counts as a failure (fixed now)

- **r = 12 m**, the config's `provisional_r_m`. A descent fails if it gets no fix, or its fix is more than 12 m from the truth (`is_failure` in `src/fidelis/plain_mc.py`). The config explains the choice: two map pixels, well beyond the 1.9 m of perfect-conditions fixes, so it marks a fix that has gone wrong, not one that's merely imprecise.
- The scan does **not** pick r. The pilot (#13) is meant to set r so that 2–10% of descents fail. That's a later, separate decision, and the scan's error distribution may inform it. Any other r shown with these results is labelled exploratory.

### What gets reported

1. **The headline:** failures out of 300 at r = 12 m, with the 95% Wilson interval that `fidelis-plain-mc` prints.
2. **By sun:** failures and the error distribution (median, max) for each of the five suns. This is the one planned breakdown. Earlier runs suggest low suns roughly double the error (`docs/benchmarks/plain-mc.md`: 4.9–6.1 m at 25° against 1.4–3.3 m otherwise, from 20 descents).
3. **Exploratory, labelled as such:** failures and errors against tilt, blur and altitude, and the errors of the worst descents.

### What each outcome means

- **0 failures.** The navigator doesn't fail under the current scenario model, even with low suns. The 95% upper bound on P(fail) is then about 1.3%. The next levers (IMU attitude error, wider blur and noise, splats for more suns under #20) are the user's call. If none produces failures, the honest summary is that the navigator is too reliable to need cheap estimation.
- **Some failures.** Report where they cluster. Size the pilot from the observed rate: how many descents it would take to reach the 10–20 reference-level failures #13 needs.
- **No-fix failures (no fix at all).** Report them separately from far-off fixes. They're a different way to fail.

## Results

**Run on 2026-11-11.** One debug job on one NVIDIA A40, with the command from the plan (job ID not recorded). All 300 seeds ran. Every record carries config SHA-256 `05bbc8fb…`, which matches the committed `config/jezero.yaml`. This was the mesh renderer's first A40 run, and it ran without errors. Median cost was 0.479 s per descent (max 0.785 s). No departures from the plan.

### Headline

**0 failures in 300 at r = 12 m. P(fail) = 0, 95% Wilson interval [0, 0.0126].** Every descent got a fix, and the fewest inliers in any descent was 1,187.

### By sun (planned)

| Sun | Az, el | Descents | Failures | Median error | 90th pct | Max |
|---|---|---|---|---|---|---|
| 0 | 257°, 40° | 61 | 0 | 2.30 m | 2.87 m | 3.47 m |
| 1 | 240°, 25° | 58 | 0 | 4.39 m | 5.26 m | 5.59 m |
| 2 | 240°, 55° | 68 | 0 | 1.57 m | 2.28 m | 2.68 m |
| 3 | 310°, 25° | 61 | 0 | 4.84 m | 5.69 m | 6.70 m |
| 4 | 310°, 55° | 52 | 0 | 1.99 m | 2.61 m | 3.05 m |

The low suns roughly double the error, as the 20-descent run suggested. Even so, the worst descent of all 300 (seed 10045, sun 3) was 6.70 m off, just over half of r.

### Exploratory (not planned; labelled as such)

- **The sun sets a fixed offset, and the scatter around it is small.** Averaging each sun's error vectors gives an offset of 1.5–4.7 m, depending on the sun. Around that average, descents scatter by only 0.8–1.3 m RMS (rms distance from the mean vector). So the low-sun error is mostly a consistent bias, not a few descents going wrong.
- **The bias points roughly away from the sun, the way shadows fall.** Bearings of the mean offset (east, north): sun 1 (azimuth 240°) gives 54°, against 60° directly away from the sun; sun 3 (310°) gives 155°, against 130°; sun 2 (240°) gives 56°. A plausible cause is shadow edges shifting matched features. That's an inference from the directions; no test isolates it.
- **Nothing else matters much.** Spearman rank correlation of error with each other drawn input: altitude 0.12, exposure gain −0.09, noise −0.07, blur 0.05, tilt 0.02.
- **The worst eight descents are all sun 3** (5.66–6.70 m). They span the full drawn range of tilt (0.1–17.4°) and exposure gain (0.50–1.44), so they don't point to a second cause.
- **Map keypoints are 70,749 in every record**, as in the sweep, against 70,726 in `descent-profile.md`. Explained later: it is the GPU model, not the map (`ctx-map-scan.md`).

### What it means

The plan's "0 failures" branch applies: under the current scenario model, even at its lowest 25° sun, the real navigator doesn't fail. The exploratory result makes this stronger than "we haven't seen one yet". Errors are a per-sun bias plus about 1 m of scatter, and the worst bias (4.7 m) sits about 7 m below r. Reaching 12 m would take scatter several times larger than anything seen here. If the scatter's tails are anything like normal, more descents of this kind won't produce failures (an inference, not a test of the tails). Failures would have to come from a lever that adds a new source of error: IMU attitude error, suns lower than 25°, or wider blur and noise. Which lever, if any, is the user's call. If none is realistic, the honest summary is that the navigator is too reliable to need cheap estimation.
