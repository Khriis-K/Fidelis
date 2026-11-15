# Dust haze (#37): the model and a laptop smoke test

**Status:** built, and scanned together with sourced errors 3 and 4 in #39: results in `combined-scan.md` (ADR 0005, decision 4). Nothing in this file is a result.

## The model

`src/fidelis/render/haze.py`, applied first in `camera_image` (`src/fidelis/render/degrade.py`):

- **Airlight (Koschmieder) model.** Each pixel becomes scene × t + veil × (1 − t), with t = exp(−τ_path / cos θ), where θ is the angle between that pixel's line of sight and straight down.
- **Dust below the camera.** τ_path = τ × (1 − exp(−h / H)), where h is the camera's height above the ground.
- **τ** is the scenario input `dust_tau`, uniform over 0.5–1: the range LVS's pre-launch tests used (Johnson et al. 2017, via `docs/research/lvs-error-sources.md`). It is a labelled range, not the landing-day haze.
- **H = 7.4 km**, the central case in `docs/research/dust-vertical-distribution.md`: a Conrath profile with a 20 km dust top. The note's bounds are 11 km (the least haze) and 5.7 km (the most). Confidence is low.
- **Pessimistic on purpose.** All the light the dust removes counts as veil, and the veil is as bright as the brightest ground, so exposure doesn't move and contrast falls by exactly t.
- **Not modelled:** dust dimming the direct sun and filling shadows with sky light.
- **τ = 0 reproduces the old image exactly**, a test checks it. Records from before #37 carry no `dust_tau` and load as 0, which is what they had.

**Approximation.** The slant path uses the dust below the camera's own spot on the ground. A ray that lands on lower or higher ground crosses a little more or less dust; at 2.2–2.45 km over the tile's relief (at most 352 m below, `scenario.terrain_relief_m`) that is a second-order term, not measured.

## What the haze range means for the failure-rate bound

`dust_tau` sits in the shared `scenario` section, so **every realistic run from #37 on is hazy**, including the 10,000-descent run (ADR 0004 decision 6, #47). Its P(fail) bound is then conditional on τ in 0.5–1, the range LVS was tested over. That range leaves out clear air and the flight value of about 0.47 (whose wavelength is unknown; if infrared, the visible value is about twice it, `docs/research/dust-vertical-distribution.md`). ADR 0005 decision 4 builds the sourced errors into the realistic model, so this is intended, but a bound must be quoted with it.

Runs from before #37 stay clear-air. The workflow's `plain_mc*` and `match_viz` rules pass `--clear-air` (dust_tau = 0, every other draw unchanged), and so must any hand rerun of an earlier scan.

## Prediction from the sources, revised for H = 7.4 km

The prediction on #37 (2026-11-13) assumed H = 11.1 km, unsourced, and its "image corner at 18° tilt" row used a line of sight about 63° off vertical. Both are revised here. **[Computed]** with `fidelis.render.haze` on the real 1024 px, 90° camera: the worst ray is the image corner with the camera tilted towards that corner, 54.7° off vertical at nadir and 72.7° at 18° tilt. Contrast in DN is t times the clear-sky value, as the model makes it, against the #37 comment's worst clear footprint (4 m/px, exposure gain 0.5: span 34.1 DN, whole-image sd 10.9 DN). That footprint and gain are themselves the worst cases the sampler draws.

| H | τ | Camera | τ_path | t at centre | t, image mean | t, worst corner | Worst corner: span / sd (DN) |
|---|---|---|---|---|---|---|---|
| 11 km (least haze) | 1 | 2450 m, 18° | 0.200 | 0.811 | 0.758 | 0.511 | 17.4 / 5.6 |
| **7.4 km (used)** | 0.5 | 2200 m, 0° | 0.129 | 0.879 | 0.848 | 0.800 | 27.3 / 8.7 |
| **7.4 km (used)** | 1 | 2200 m, 0° | 0.257 | 0.773 | 0.720 | 0.641 | 21.8 / 7.0 |
| **7.4 km (used)** | 1 | 2450 m, 18° | 0.282 | 0.743 | 0.677 | 0.387 | 13.2 / 4.2 |
| 5.7 km (most haze) | 1 | 2450 m, 18° | 0.349 | 0.692 | 0.617 | 0.309 | 10.5 / 3.4 |

Against JPL's ~4 DN matching limit (Johnson et al. 2017 p.15, via `docs/research/lvs-error-sources.md`):

- **Read as a span**, the worst corner keeps 13.2 DN at the H used, about 3 times the limit. Even the most-haze bound keeps 10.5 DN.
- **Read as a standard deviation**, the worst corner reaches the limit: 4.2 DN at the H used, 3.4 DN at the most-haze bound. That needs every worst case at once (τ = 1, top altitude, full tilt towards a corner, the dimmest exposure, the lowest-contrast footprint), and holds only in the far corner. The image as a whole keeps a mean t of 0.68, so a whole-image sd of about 7.4 DN.
- **Prediction: no lost fixes from haze.** The worst draws may lose features in the far corners, so some inlier counts may fall at high τ. The limit is JPL's matcher's; where ALIKED + LightGlue stops matching hasn't been measured, so a surprise here would be about our matcher, not the haze.
- **Compared with the first prediction:** contrast at the worst corner is lower than it said (0.39 kept, not 0.67), partly from the lower H and partly from the corrected corner geometry. The conclusion holds with less margin.

## Smoke test on the laptop (not a result)

Run on 2026-11-14. Three seeds, 50,000–50,002, which no earlier run used and which are now spent. Mesh renderer on the RTX 5070 Laptop GPU, CTX map **thinned to its 8,000 highest-scoring keypoints** (`mesh-k8000`) because the laptop can't hold the full map's matching (`docs/benchmarks/plain-mc.md`). Each seed ran three ways: τ = 0, the τ it drew, and τ = 1. Everything else about the seed stayed the same.

| Seed | τ | Status | Inliers | Map-frame error | Image sd |
|---|---|---|---|---|---|
| 50000 | 0 | ok | 194 | 6.74 m | 32.2 DN |
| 50000 | 0.616 | ok | 199 | 5.21 m | 25.8 DN |
| 50000 | 1 | ok | 196 | 4.79 m | 22.5 DN |
| 50001 | 0 | ok | 200 | 5.81 m | 27.2 DN |
| 50001 | 0.948 | ok | 195 | 5.81 m | 21.6 DN |
| 50001 | 1 | ok | 195 | 5.97 m | 21.3 DN |
| 50002 | 0 | ok | 283 | 6.50 m | 18.2 DN |
| 50002 | 0.619 | ok | 282 | 6.09 m | 15.4 DN |
| 50002 | 1 | ok | 276 | 6.06 m | 14.2 DN |

- **The plumbing works.** The simulator reads H from the config, and every hazy descent got a fix.
- **Contrast fell as expected.** At τ = 1 the image's standard deviation is 69.9–78.3% of the same seed's clear-air value. Contrast falls by t, which ranges over the image and is lowest at its corners, and noise and clipping sit on top of that, so this is a whole-image figure.
- **Errors and inliers barely moved**, but three seeds on a thinned map can't say anything about the effect on error or failures. That's #39's scan.
