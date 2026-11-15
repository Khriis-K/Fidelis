# Focal-length calibration error (#38): the model and a laptop smoke test

**Status:** built, and scanned together with sourced errors 2 and 4 in #39: results in `combined-scan.md` (ADR 0005, decision 4). Nothing in this file is a result.

## The model

- **The navigator's camera is miscalibrated; the render's isn't.** The renderer uses LCAM's true camera. The navigator warps, matches and solves with the same camera, its focal length times 1 + e (`Intrinsics.with_focal_error`, `src/fidelis/render/camera.py`). The principal point stays where it is. Lens distortion isn't modelled: Fidelis renders a pinhole, and LCAM's distortion residual after correction isn't sourced.
- **e is the scenario input `focal_error`**, uniform over ±1%. ±1% is LCAM's *pre-calibration* bound, the largest term in JPL's 2017 error budget (Johnson et al. 2017, via `docs/research/lvs-error-sources.md`). The post-calibration uncertainty was searched for twice and not found (research note, section 4). So this is a **labelled range**, not the flight camera's error.
- **Uniform, not Gaussian.** JPL drew it Gaussian within ±1%. The sigma isn't stated, and uniform covers the range evenly, so #39 can read results per range point. A Fidelis choice.
- **e = 0 reproduces the old navigator exactly.** Records from before #38 carry no `focal_error` and load as 0, which is what they had. The workflow rules behind those results pass `--exact-camera`, so rerunning them doesn't add a focal error.

## What a focal error does to the fix

A focal length that's too long makes every ray look closer to the optical axis than it is. Over flat ground the solve explains that by moving the camera:

- **At nadir, exactly e × h straight up.** Seen square on, a longer focal length looks the same as a camera farther away.
- **Off nadir, to first order in e, e·h·sin 2θ horizontally, back from the tilt direction, and e·h·cos 2θ up** (θ is the tilt). Derived in the vertical plane of the tilt. Fit the ground points x = h·tan(θ + φ), with each ray's angle shrunk to tan φ′ = tan φ / (1 + e). Moving the camera that much fits every ray to first order. The rest of the image, out of that plane, bends the horizontal term by under 2% at 18°. That was checked on a full-image grid in `tests/nav/test_pnp.py`, which allows 3% along the tilt and 2% across it. The height term bends more, up to about 7% at 18°, but height doesn't count against the line (#46).

At our worst case (h = 2450 m, θ = 18°, e = ±1%) that's 14.4 m horizontally **[Computed]**. JPL's figure for ±1% was up to 16.9 m (99%), off nadir, at their own altitudes and tilts.

## Prediction from the sources

Posted on #38 before the build (2026-11-13, revised 2026-11-14 for the horizontal line):

- **0 failures at 40 m across the whole ±1% range.**
- The focal term alone, from a perfect-match geometry check over 400 draws: 13.6 m (99th percentile) and 14.2 m (worst) horizontally at ±1% **[Computed]**. The check was a throwaway script, not committed; its method is on #38. The formula above gives the same size: 14.4 m at the worst draw.
- Combined with the CTX map scan's worst descent (10.02 m, `ctx-map-scan.md`): about 17.3 m if the two combine as a root sum of squares, about 24.2 m if they line up exactly **[Computed]**. The 40 m line would need a focal error of about 2.1–2.7%.
- **A surprise** would be a horizontal error over about 25 m at ±1%, or lost fixes. Either would point at matching or RANSAC reacting to the wrong camera, not at the geometry.

## How the range enters the combined scan (#39)

- #39's 300 descents each draw `focal_error` uniform over ±1%, alongside the drawn haze and attitude error. No separate focal-error scan.
- **Labelled as a range.** Any P(fail) from #39 holds only for focal errors spread over the pre-calibration ±1%, as it already holds only for haze over τ 0.5–1 (`dust-haze.md`). It is not the flight camera's rate, which needs the post-calibration figure that isn't found.
- **Read per range point.** #39 reports the map-frame horizontal error binned by |e|: 0–0.25%, 0.25–0.5%, 0.5–0.75% and 0.75–1%, about 75 descents each. That shows whether misses, if any, come from the top of the range. #39's plan should also check the binned error against the geometric term e·h·sin 2θ, so a matching effect shows up as a residual on top of the geometry.
- **A caution for #39's wording.** Its issue says the combined scan gives "the realistic failure rate". With haze and the focal error both labelled ranges, it gives a rate *under those ranges*. The write-up should say so.

## Smoke test on the laptop (not a result)

Run on 2026-11-14. Three seeds, 50,003–50,005, which no earlier run used and which are now spent. Mesh renderer on the RTX 5070 Laptop GPU, CTX map thinned to its 8,000 highest-scoring keypoints (`mesh-k8000`), because the laptop can't hold the full map's matching (`docs/benchmarks/plain-mc.md`). Each seed ran four ways: e = 0, the e it drew, +1% and −1%. Everything else about the seed stayed the same, haze included. The fixed values were forced by a throwaway script, not committed: `dataclasses.replace(draw(config, seed), focal_error=e)`, run through `build_simulator(..., ("mesh-k8000",))`. No CLI flag pins e. "Shift" is the change in map-frame error from e = 0, along the tilt direction. "Predicted" is −e·h·sin 2θ.

| Seed | Tilt, h | e | Status | Inliers | Horizontal error | Shift | Predicted | Height shift |
|---|---|---|---|---|---|---|---|---|
| 50003 | 14.4°, 2284 m | 0 | ok | 203 | 5.90 m | | | |
| 50003 | | +0.30% | ok | 203 | 8.56 m | −3.30 m | −3.27 m | +5.92 m |
| 50003 | | +1% | ok | 202 | 15.82 m | −10.85 m | −11.03 m | +20.64 m |
| 50003 | | −1% | ok | 201 | 7.27 m | +10.93 m | +11.03 m | −20.17 m |
| 50004 | 15.7°, 2328 m | 0 | ok | 258 | 6.68 m | | | |
| 50004 | | −0.47% | ok | 265 | 4.80 m | +2.74 m | +5.68 m | −10.97 m |
| 50004 | | +1% | ok | 265 | 19.27 m | −13.89 m | −12.12 m | +19.60 m |
| 50004 | | −1% | ok | 261 | 7.47 m | +10.33 m | +12.12 m | −21.20 m |
| 50005 | 17.3°, 2298 m | 0 | ok | 275 | 5.20 m | | | |
| 50005 | | +0.47% | ok | 277 | 10.12 m | −6.00 m | −6.10 m | +9.10 m |
| 50005 | | +1% | ok | 275 | 17.18 m | −13.41 m | −13.04 m | +19.44 m |
| 50005 | | −1% | ok | 275 | 10.65 m | +13.25 m | +13.04 m | −20.26 m |

- **The plumbing works.** The simulator reads the drawn e, and every run got a fix.
- **The fix moves as the geometry says.** The along-tilt shift is within 0.4 m of −e·h·sin 2θ in 6 of 9 runs, every run of seeds 50003 and 50005. Seed 50004 is off by 1.8–2.9 m in all three, likeliest from the matches themselves: the warp to nadir uses the believed camera, so each e gives a slightly different image to match. Its shift isn't linear in e either: −0.47% gave 2.74 m and −1% gave 10.33 m, a ratio of 0.27 where the geometry says 0.47. So the scatter is a few metres per run, not a fixed bias. Three seeds can't separate that from anything else.
- **Inlier counts didn't move** (within 7 per seed), so at this size the wrong camera doesn't cost matches. That's the first hint for the "surprise" above, not a test of it.
- **The horizontal error depends on the sign.** The focal shift adds to whatever error the seed already had, and its direction is set by the tilt, so +1% and −1% give different totals (15.8 m against 7.3 m for seed 50003). The worst here was 19.27 m. Three seeds on a thinned map say nothing about failures; that's #39's scan.
