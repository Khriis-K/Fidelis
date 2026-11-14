# Dust haze (#37): the model and a laptop smoke test

**Status:** built, not scanned. The haze is scanned once, together with sourced errors 3 and 4, in #39 (ADR 0005, decision 4). Nothing here is a result.

## The model

`src/fidelis/render/haze.py`, applied first in `camera_image` (`src/fidelis/render/degrade.py`):

- **Airlight (Koschmieder) model.** Each pixel becomes scene × t + veil × (1 − t), with t = exp(−τ_path / cos θ), where θ is the angle between that pixel's line of sight and straight down.
- **Dust below the camera.** τ_path = τ × (1 − exp(−h / H)), where h is the camera's height above the ground.
- **τ** is the scenario input `dust_tau`, uniform over 0.5–1: the range LVS's pre-launch tests used (Johnson et al. 2017, via `docs/research/lvs-error-sources.md`). It is a labelled range, not the landing-day haze.
- **H = 7.4 km**, the central case in `docs/research/dust-vertical-distribution.md`: a Conrath profile with a 20 km dust top. The note's bounds are 11 km (the least haze) and 5.7 km (the most). Confidence is low.
- **Pessimistic on purpose.** All the light the dust removes counts as veil, and the veil is as bright as the brightest ground, so exposure doesn't move and contrast falls by exactly t.
- **Not modelled:** dust dimming the direct sun and filling shadows with sky light.
- **τ = 0 reproduces the old image exactly**, a test checks it. Records from before #37 carry no `dust_tau` and load as 0, which is what they had.

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
- **Contrast fell as expected.** At τ = 1 the image's standard deviation is 70–78% of the same seed's clear-air value. Contrast falls by t, which ranges over the image and is lowest at its corners, and noise and clipping sit on top of that, so this is a whole-image figure.
- **Errors and inliers barely moved**, but three seeds on a thinned map can't say anything about the effect on error or failures. That's #39's scan.
