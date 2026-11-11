# Failure scan: does the real navigator ever fail? (before the pilot, #13)

The fidelity sweep (`docs/benchmarks/fidelity-sweep.md`) saw 0 failures in 300 descents, all under the landing-day sun. A failure-probability estimate needs failures, so this scan asks whether, and where, the real navigator fails. This file fixes the failure line and the analysis **before** the run, so neither can be tuned to the data. Results go under the plan afterwards, and any departure from the plan is stated there with its reason.

## Plan

**Status: pre-registered 2026-11-11, not yet run.** Nothing below is a measurement.

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

Not yet run.
