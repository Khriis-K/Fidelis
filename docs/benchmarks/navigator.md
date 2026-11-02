# Navigator checks (issue #7)

The navigator (`src/fidelis/nav/`): ALIKED keypoints in the descent image, pretrained LightGlue matches to the CTX-like orbital map, then OpenCV PnP inside RANSAC for the camera pose. Measured on the laptop (RTX 5070 Laptop GPU, 8 GB) on 2026-11-01; every table below is from the rerun on 2026-11-02, which added the attitude-warped heading rows (issue #34). Reproduce with:

    uv run python scripts/check_navigator.py data/jezero

## The question: where does matching break as the map coarsens?

The spec's main early risk was that a 6 m CTX-like map would be too coarse to match against sharp HiRISE-derived descent renders, even in perfect conditions. The sweep renders perfect-conditions descent images and locates them against maps averaged from the tile's 2 m orthoimage to 2, 4, 6, 10 and 12 m pixels.

**Perfect conditions:** nadir, north up, 1024 spp (about 0.3% render noise; no added noise or blur), and a **zenith sun**. The orthoimage already carries HiRISE's own lighting, and a zenith sun of irradiance π renders flat ground at exactly its albedo, so the render adds as little shading as the renderer allows on top of what the map shows. That's how this check reads the spec's "sun matching the map". The camera sits at (150, −230) m from the tile centre, on no pixel grid, at the top and bottom of LVS's band: 4.2 and 2.2 km above the tile's mean ground (−2553.2 m). Descent pixels there cover 8.2 and 4.3 m of ground.

**The finest map is 2 m, not HiRISE's 0.25 m.** The render's texture is the 2 m tile orthoimage, so a finer map would hold no detail the render has. "Full resolution" in this sweep means 2 m.

**Same ground at every resolution.** The sweep's maps are a 6 km square around the tile centre: the whole tile at 2 m (5010 px, about 100,000 keypoints) doesn't fit LightGlue on 8 GB. The 6 m row is therefore not the workflow's map; that one is in the next section.

### 4.2 km (8.2 m descent pixels)

| Map | Map keypoints | Matches | Inliers | Status | Position error (m) | Locate (s) |
|---|---|---|---|---|---|---|
| 2 m | 34235 | 14 | 0 | no_solution | - | 0.72 |
| 4 m | 9157 | 1633 | 1632 | ok | 1.72 | 0.12 |
| 6 m | 4096 | 1851 | 1851 | ok | 0.88 | 0.09 |
| 10 m | 1793 | 1537 | 1537 | ok | 1.72 | 0.10 |
| 12 m | 1265 | 1031 | 1031 | ok | 3.28 | 0.10 |

### 2.2 km (4.3 m descent pixels)

| Map | Map keypoints | Matches | Inliers | Status | Position error (m) | Locate (s) |
|---|---|---|---|---|---|---|
| 2 m | 34235 | 3346 | 3298 | ok | 0.57 | 0.41 |
| 4 m | 9157 | 3572 | 3568 | ok | 0.55 | 0.09 |
| 6 m | 4096 | 2218 | 2218 | ok | 0.92 | 0.08 |
| 10 m | 1793 | 928 | 914 | ok | 3.23 | 0.11 |
| 12 m | 1265 | 641 | 628 | ok | 1.89 | 0.10 |

**The numbers drift between runs.** The 2026-11-01 run of this table had the 2 m row at 3343 matches, 3308 inliers and 1.38 m, against 0.57 m here, on the same code path; the 4 m and 12 m rows moved by 0.01–0.02 m and a few matches. The 4.2 km table and every full-map row reproduced exactly, timings aside. Likely non-deterministic GPU kernels in ALIKED or LightGlue, which matter most on the 2 m map's 34,235 keypoints; not investigated. How much any one row can drift is unmeasured; one rerun is not a spread.

**Matching does not break at CTX-like resolution.** At 6 m it locates within 0.92 m at both ends of the band, with about 2,000 inliers. It holds out to 12 m, the coarsest map tried, within 3.3 m.

**What breaks it is the scale ratio, not coarseness.** The one failure is the *sharpest* map at the *highest* altitude: a 2 m map against 8.2 m descent pixels, a map 4× finer than the image. ALIKED's keypoints and descriptors aren't scale-invariant, so they need the two images at a similar ground scale. Map pixel over descent pixel, the sweep worked from 0.47 (2 m at 2.2 km) to 2.8 (12 m at 2.2 km) and failed at 0.24 (2 m at 4.2 km). Where in 0.24–0.47 it breaks wasn't measured. LVS's 6 m map sits at 0.73–1.4 across the band, comfortably inside the range that worked.

**The spec's fallbacks aren't needed.** The spec's fallback order (restrict matching to the LVS band, try the fine-tuned matcher, then a less-coarse map recorded in an ADR) applies only if the CTX-like map fails, and it didn't. The map stays at 6 m. That holds for perfect conditions only; see the next section.

The 6 m window map's keypoint count is the cap: its 1000 px fits in one 1024 px tile, which reached `MAX_KEYPOINTS` (4,096).

## The workflow's full-tile 6 m map, and conditions beyond perfect

The whole 1670 px map, as the navigator uses it: four tiles, 14,290 keypoints. The 2.2 km views here are rendered from the same position as above.

| Map | Map keypoints | Matches | Inliers | Status | Position error (m) | Locate (s) |
|---|---|---|---|---|---|---|
| 4.2 km, zenith sun | 14290 | 3524 | 3521 | ok | 0.63 | 0.13 |
| 2.2 km, zenith sun | 14290 | 1954 | 1944 | ok | 1.88 | 0.11 |
| 2.2 km, landing-day sun (257, 40) | 14290 | 1918 | 1911 | ok | 3.40 | 0.12 |
| 2.2 km, zenith sun, heading 15 | 14290 | 1947 | 1929 | ok | 2.19 | 0.12 |
| 2.2 km, zenith sun, heading 45 | 14290 | 148 | 90 | ok | 12.31 | 0.12 |
| 2.2 km, zenith sun, heading 90 | 14290 | 23 | 0 | no_solution | - | 0.23 |
| 2.2 km, zenith sun, heading 180 | 14290 | 63 | 0 | no_solution | - | 0.21 |
| 2.2 km, zenith sun, heading 15, warped by attitude | 14290 | 1997 | 1986 | ok | 1.83 | 0.23 |
| 2.2 km, zenith sun, heading 45, warped by attitude | 14290 | 1927 | 1914 | ok | 1.62 | 0.17 |
| 2.2 km, zenith sun, heading 90, warped by attitude | 14290 | 1958 | 1948 | ok | 1.84 | 0.14 |
| 2.2 km, zenith sun, heading 180, warped by attitude | 14290 | 1954 | 1946 | ok | 1.86 | 0.12 |

- **The full map is a little worse at 2.2 km than the window** (1.88 m against 0.92 m, 264 fewer matches): it has 3.5× as many keypoints, most of them outside the view, for LightGlue to sort through. One position, so whether that difference is typical isn't known.
- **The landing-day sun (azimuth 257°, elevation 40°) still locates within 3.4 m.** Its shading and shadows don't match the map's lighting, and matches barely drop. One sun, one position.
- **Heading breaks the bare matcher.** Matched as it is, a view turned 15° from north-up still locates (2.19 m); at 45° it keeps 90 inliers and misses by 12 m; at 90° and 180° it fails. ALIKED and LightGlue aren't rotation-invariant, and the scenario sampler draws heading uniformly over 0–360°.
- **Warping by the known attitude fixes it (issue #34).** Like LVS, which knew its attitude from the IMU, `locate` can take the camera's attitude and warp the image to a nadir, north-up view before matching (see below). With the true attitude, every heading tried locates within 1.62–1.86 m on 1,914–1,986 inliers, the same as the north-up view's 1.88 m on 1,944. The 90° and 180° rows aren't bit-identical to north-up because each heading is its own render, with its own render noise. One position, one altitude, zenith sun, and a *perfect* attitude: what attitude-knowledge error costs is a follow-up.

**Memory and time.** Peak torch GPU memory was 3.8 GB over the whole run (the 2 m map's 25 tiles and LightGlue on 34,235 keypoints), on top of Mitsuba's scene, which the script frees before navigating. Locating one image against the full map takes 0.11–0.23 s, after the map's features are extracted once.

## Warping by attitude (issue #34)

`locate(image, intrinsics, attitude)` takes the camera-to-world rotation the spacecraft believes it has. The image is warped by the homography `K R_nadirᵀ R K⁻¹` to what a nadir, north-up camera at the same place would see; that is exact for any terrain, since only the rotation changes. The canvas grows to hold the whole warped image, and keypoints within 8 px of its empty corners are dropped (their straight edges aren't in the map). Matched keypoints go back through the inverse homography, so PnP and `NavResult.image_points` stay in the camera's own pixels. Without an attitude the image is matched as it is.

**Turns off the pixel grid cost accuracy on the synthetic CPU fixture, not (yet) on Jezero.** Reproduce with `uv run python scripts/check_attitude_warp.py` (CPU, about a minute). On the tests' analytic nadir view (256 px, 1.5 m descent pixels against a 2 m map), turns by multiples of 90° only move whole pixels and reproduce north-up exactly: 0.174 m. Other turns resample the image and land further off, with the navigator's bilinear warp:

| Heading | 0° | 15° | 30° | 45° | 60° | 90° | 135° | 180° | 270° |
|---|---|---|---|---|---|---|---|---|---|
| Bilinear warp (m) | 0.174 | 0.795 | 0.405 | 0.496 | 0.435 | 0.174 | 0.496 | 0.174 | 0.174 |
| Bicubic warp (m) | 0.174 | 0.614 | 0.421 | 0.557 | 0.244 | 0.174 | 0.527 | 0.174 | 0.174 |

At eight more positions (bilinear), north-up gave 0.134–0.410 m, 45° gave 0.213–0.957 m and 15° gave 0.383–0.769 m; the ranges overlap, so there's no single cost ratio. Resampling explains part of it: the north-up view turned 45° and back, with no navigator warp, went from 0.174 to 0.339 m. Bicubic is mixed (better at 15° and 60°, worse at 30°, 45° and 135°), so the warp stays bilinear. On the Jezero renders above, 15° and 45° landed no worse than north-up; there the unexplained bias of about 1.9 m at 2.2 km is larger than any resampling loss.

**Tests.** The CPU tests require 90° and 180° within the north-up test's quarter pixel, 15° and 45° within half a map pixel (the GPU test's tolerance), and a 180° view with no attitude or a north-up one to stay lost. The returned pixels are checked against the *true* camera, not the fix: mutation testing showed that a back-mapping turned the wrong way passes a check against the fix, because PnP turns the camera to suit the pixels. A homography test with a tilted camera covers tilt, which every nadir test is blind to (a nadir camera's rotation matrix is symmetric, so transposing it changes nothing).

## The GPU test (acceptance: perfect conditions, small stated tolerance)

`tests/nav/test_navigator.py::test_a_perfect_conditions_render_of_jezero_locates_within_half_a_ctx_map_pixel` renders the perfect-conditions view at 2.3 and 4.1 km above the mean ground, near both ends of the LVS band, and requires each fix within **half the map's ground pixel: 2.85 m** (its east–west ground pixel is 5.69 m, its north–south 6 m). Thousands of inliers average far below one map pixel, so missing that means something systematic.

**Why both altitudes.** Mutation testing (`spec-verify`) showed the test at 2.3 km alone was vacuous for the bug it was meant to catch. Dropping the half-pixel centre offset from the map's georeferencing moves every map point by half a pixel, about 2.85 m east and 3 m north (4.1 m). Near 2.2 km the correct fix already sits 1.88 m off (table above), and the slip partly cancelled that error: the slipped fix stayed inside 2.85 m and the test passed. Near 4.2 km the correct fix is 0.63 m off, and the same slip failed it. With both altitudes the test catches the slip. Where the 1.88 m comes from wasn't investigated; a convention error would show at both altitudes, not mainly at one.

Run it with `uv run pytest -m gpu tests/nav`.

The CPU tests check the same chain without a renderer: an analytic nadir view of a synthetic flat map has to locate within a quarter of its 1.9 m ground pixel, and the map's pixel-to-world conversion has to reproduce a tilted plane exactly from a DTM on a different grid. The attitude warp's CPU tests are in the section above.

## Choices worth knowing

- **ALIKED resizes inputs to 1024 px unless told not to.** LightGlue's ALIKED wrapper defaults to `resize=1024`, which would quietly turn the 1670 px, 6 m map into a 1024 px, 9.8 m one. Both images are extracted at their own resolution.
- **The map is extracted in 1024 px tiles.** ALIKED's dense features for a whole map don't fit on 8 GB: a 5010 px map needs about 13 GB for one 128-channel layer, and the prototype ran out of memory on it. Keypoints near a tile's edge get descriptors from less context; not measured.
- **Elevation comes from the tile's 4 m DTM**, the same terrain the mesh is built from. LVS's onboard elevation map was coarser, so the map's 3D points are better here than in flight.
- **A fix needs 10 RANSAC inliers.** Below that the result is a failure record (`no_solution` or `too_few_matches`), not an exception. Ten is twice RANSAC's minimal sample; random matches almost never agree beyond it. A fix with a few more than 10 inliers can still be badly wrong; the navigator reports it as `ok` with its inlier count, and the Monte Carlo counts it as the large error it is rather than the navigator second-guessing it.
- **The RANSAC threshold is OpenCV's default, 8 px.** At 2.2 km that's 34 m of ground, about six CTX map pixels.

**Caveats:** one laptop GPU, one camera position, nadir or near-nadir views, two suns, a perfectly known attitude, pretrained (not fine-tuned) LightGlue. The sweep's maps are block averages of the 2 m texture (CTX's optics and noise aren't modelled; see `src/fidelis/terrain/matching_map.py`), and the descent renders come from the same orthoimage as the map, so the two share everything but resolution and the renderer's shading. That makes this a best case.

**What this does and doesn't settle.** The spec's risk was HiRISE-sharp descent renders (0.25 m detail) against the 6 m map. These renders carry only the 2 m texture's detail, so the sweep retires the risk for 2 m renders only; rerun it when sharper renders (the splat scenes) exist. The zenith sun is this check's reading of "sun matching the map", since no sun reproduces the orthoimage's baked-in HiRISE lighting. The 1.88 m error near 2.2 km is unexplained.
