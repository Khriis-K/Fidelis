# Attitude-knowledge error (#39): the model and a laptop smoke test

**Status:** built and smoke-tested on the laptop; prediction written 2026-11-14, before the build. Scanned together with sourced errors 2 and 3: results, and how the prediction held, in `combined-scan.md` (ADR 0005 decision 4). Nothing in this file is a result.

## What changes

Until #39 the navigator was handed the camera's true attitude. Now it gets the attitude the spacecraft believes it has: the truth turned by two small errors, both from Johnson et al. 2017 via `docs/research/lvs-error-sources.md`, section 1.

- **Attitude at LVS start.** The DIMU-propagated attitude, "less than 0.5° per axis" (p.8). JPL's Monte Carlo drew it Gaussian, "less [than] 0.5° per axis" (p.12). The sigma level isn't stated, so this reads as a bound.
- **LCAM-to-IMU misalignment.** The camera sits on the rover and the IMU on the descent stage, so their alignment can slip. JPL simulated it as Gaussian within ±0.75° per axis and found it cost "no more than 3.0m" (pp.11–12).

The render keeps the true camera. Only the navigator's belief is wrong.

## The model

- **Six scenario inputs**, in degrees: `attitude_error_x_deg`, `_y_deg`, `_z_deg`, and `misalignment_x_deg`, `_y_deg`, `_z_deg`. Each is a rotation about one of the camera's own axes (x right, y down, z along the optical axis; `docs/camera-conventions.md`). About z is a heading error; about x and y is a tilt error.
- **The believed attitude** is R_true · R(attitude error) · R(misalignment), where each R is the rotation of that axis-angle vector (camera-to-world, as `CameraPose` stores it). At these sizes the order of the two makes no measurable difference. A Fidelis choice.
- **Zero error gives back the true attitude exactly.** The rotation of a zero vector is the identity. Tests check that zero error hands the navigator the true attitude bit for bit (`tests/test_simulation.py`), so a zero-error run is the run from before #39. Records from before #39 carry none of the six inputs and load as 0, which is what they had.

### From a "less than" bound to a distribution (the written choice)

**Each axis is drawn uniform over the bound: ±0.5° for the attitude error, ±0.75° for the misalignment, all six independent.** A Fidelis choice, made for three reasons:

1. **The source gives a bound, not a sigma.** A Gaussian needs a sigma we'd have to invent. If 0.5° were 3σ, the typical error would be a third of what it would be at 1σ, and nothing in the source picks between them.
2. **Uniform leans towards large errors.** No draw breaks the bound, and errors near it are as likely as small ones. Any Gaussian cut at the same bound puts more of its draws near zero. So if the effect grows with the error's size, as the fixture suggests, uniform overstates it rather than hiding it.
3. **It matches #38.** The focal error is uniform over its bound for the same reason (`focal-error.md`).

The cost of this choice: the realistic attitude error is probably smaller than this. The reconstructed flight attitude error was under 0.1° per axis (Johnson et al. 2022 p.15, via the research note). So these draws sit at the pessimistic end, on purpose.

**Sizes this gives** **[Computed]**, 20,000 draws of the composed rotation (a throwaway script, not committed):

| Rotation | Median | 99th pct | Max seen | Hard bound |
|---|---|---|---|---|
| Total | 0.85° | 1.55° | 1.89° | 2.17° (√3 × 1.25°) |
| About the optical axis (heading) | 0.38° | 1.13° | 1.25° | 1.25° |
| Off-axis (tilt) | 0.67° | 1.35° | 1.70° | 1.77° (√2 × 1.25°) |

## Prediction from the sources

Written before the build, as `CLAUDE.md`'s literature null-case test asks. It updates the comment on #39 (2026-11-13) with the sizes above.

**1. The source's conditions.** JPL's 3.0 m misalignment cost came from LVS's own estimator. In its coarse phase LVS took attitude from the IMU and **solved for horizontal position only** (Johnson et al. 2023, via the research note). A tilt it doesn't know about then reads directly as a position error: 0.75° at 2.2 km is about 29 m on the ground before the fine phase corrects it. JPL also swept attitude errors of 0–5° in 0.5° steps, and LVS "was able to take out all of these" (2017 p.13, Table 3). That is 11 values per axis, all passing; no failure was reported at any of them.

**2. Where Fidelis sits.** Fidelis's navigator is structurally different from LVS's coarse phase. It solves the **full pose** with PnP and no starting guess (`src/fidelis/nav/pnp.py`, `SOLVEPNP_SQPNP` inside RANSAC). The attitude only enters the warp to a nadir, north-up view before matching (`src/fidelis/nav/navigator.py`). The matched keypoints go back through the inverse of that same warp, so they land on the right camera pixels whatever attitude was used. So **an attitude error can't bias the geometry. It can only change which features match.** That is LVS's 3 m mechanism switched off. Our errors reach at most 2.17° in total, inside JPL's tested 5°, and under the 3° where Fidelis already saw little effect:

- **CPU fixture** (`navigator.md`, "Warping by attitude"; 256 px, 1.5 m descent pixels on a 2 m map). Up to 3° of heading or tilt error, one axis at a time, the fix moved by at most 0.44 m from the same view's perfect-attitude fix, and every case kept its fix.
- **Jezero, one render.** A 15° heading error with no warp at all cost 0.31 m (2.19 m against 1.88 m north-up). That's twelve times our largest heading error (1.25°).

**3. The numbers we expect.**

- **No lost fixes, and no failures at 40 m, from attitude error.**
- **No bias.** Across the scan the mean change in map-frame horizontal error from adding attitude error should be near zero, inside the per-seed scatter.
- **Per-seed scatter under about 1.5 m.** **[Inference]** Scaling the fixture's 0.44 m by the pixel size, 1.5 m there against 4.3–4.8 m at Jezero's altitudes (descent pixel = 2h / 1024), gives about 1.4 m. If the effect scales with the pixel, that's the size. It might not; the fixture tried one axis at a time, and here all three move at once.
- **Inliers barely move.** A sub-2° residual turn leaves the warped image nearly nadir and north-up.

**A surprise would be** a seed whose horizontal error changes by more than about 3 m when the attitude error is switched on, or a lost fix. That would mean matching is more sensitive to a small residual rotation than the fixture showed. The geometry can't cause it.

**What the combined scan can and can't see.** In the scan itself, attitude error is hidden under the focal error, which moves fixes by up to about 14 m (`focal-error.md`). Only a run with the attitude error switched off and everything else the same isolates it, seed for seed. The scan plan (`combined-scan.md`) includes one.

## Limits known before the run

- **JPL's 3.0 m can't show up here.** Fidelis's PnP solve doesn't use the attitude for geometry. A real LVS fine phase also estimates attitude, but its coarse phase doesn't. So a small attitude error costs Fidelis less than it cost LVS. That's a property of our navigator, not evidence about LVS.
- **Static errors only.** JPL's later simulator split the misalignment into a static part and a dynamic part driven by accelerations (Aaron et al. 2022 p.3). Fidelis takes one image per descent, so only the static part has meaning. Flight showed the dynamic part under 0.025° (Mohan et al. 2021 p.16).
- **The axes are the camera's.** JPL's "per axis" doesn't say whose axes. At these sizes and with every axis drawn the same way, which frame is used changes little. A Fidelis choice.

## Smoke test on the laptop (not a result)

Run on 2026-11-14, after the build. Three seeds, 50,006–50,008, which no earlier run used and which are now spent. Mesh renderer on the RTX 5070 Laptop GPU, CTX map thinned to its 8,000 highest-scoring keypoints (`mesh-k8000`), because the laptop can't hold the full map's matching (`docs/benchmarks/plain-mc.md`). Each seed ran three ways: the true attitude, the attitude error it drew, and every axis at its bound (+0.5°, −0.5°, +0.5° and +0.75°, −0.75°, +0.75°, 2.17° in total). Everything else about the seed stayed the same, its haze and focal error included. The values were forced by a throwaway script, not committed: `dataclasses.replace(draw(config, seed), ...)`, run through `build_simulator(..., ("mesh-k8000",))`. "Change" is the change in map-frame horizontal error from the true-attitude run, as a length, not a vector.

| Seed | Tilt | τ, e | Attitude | Rotation (total, about the axis) | Status | Inliers | Horizontal error | Change |
|---|---|---|---|---|---|---|---|---|
| 50006 | 15.2° | 0.84, +0.60% | true | 0 | ok | 315 | 1.66 m | |
| 50006 | | | drawn | 0.51°, −0.24° | ok | 317 | 2.11 m | +0.45 m |
| 50006 | | | bounds | 2.17°, +1.25° | ok | 310 | 1.49 m | −0.17 m |
| 50007 | 0.4° | 0.85, −0.67% | true | 0 | ok | 193 | 2.38 m | |
| 50007 | | | drawn | 0.99°, +0.47° | ok | 196 | 2.44 m | +0.06 m |
| 50007 | | | bounds | 2.17°, +1.25° | ok | 194 | 2.91 m | +0.53 m |
| 50008 | 17.0° | 0.95, +0.67% | true | 0 | ok | 264 | 14.13 m | |
| 50008 | | | drawn | 0.45°, +0.22° | ok | 261 | 12.45 m | −1.68 m |
| 50008 | | | bounds | 2.17°, +1.25° | ok | 264 | 14.62 m | +0.49 m |

- **The plumbing works.** The simulator reads all six inputs, and every run got a fix.
- **Inliers didn't move** (within 7 per seed), as predicted.
- **The changes are scatter, not a trend.** The largest, −1.68 m on seed 50008, came from a 0.45° draw; the 2.17° bound on the same seed moved it +0.49 m. That's the "different features match" effect the prediction describes, not geometry. It is a little over the "under about 1.5 m" **[Inference]** above, and under the 3 m that would count as a surprise. Seed 50008 is also the seed with a large focal shift (14 m at 17° tilt), and its fix may simply be more sensitive. Three seeds on a thinned map say nothing about the scan; the no-attitude run in `combined-scan.md` measures this on 300.
