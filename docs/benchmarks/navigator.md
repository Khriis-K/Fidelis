# Navigator checks (issue #7)

The navigator (`src/fidelis/nav/`): ALIKED keypoints in the descent image, pretrained LightGlue matches to the CTX-like orbital map, then OpenCV PnP inside RANSAC for the camera pose. Measured on the laptop (RTX 5070 Laptop GPU, 8 GB) on 2026-11-01. Reproduce with:

    uv run python scripts/check_navigator.py data/jezero

## The question: where does matching break as the map coarsens?

The spec's main early risk was that a 6 m CTX-like map would be too coarse to match against sharp HiRISE-derived descent renders, even in perfect conditions. The sweep renders perfect-conditions descent images and locates them against maps averaged from the tile's 2 m orthoimage to 2, 4, 6, 10 and 12 m pixels.

**Perfect conditions:** nadir, north up, 1024 spp (about 0.3% render noise; no added noise or blur), and a **zenith sun**. The orthoimage already carries HiRISE's own lighting, and a zenith sun of irradiance π renders flat ground at exactly its albedo, so the render adds as little shading as the renderer allows on top of what the map shows. That's how this check reads the spec's "sun matching the map". The camera sits at (150, −230) m from the tile centre, on no pixel grid, at the top and bottom of LVS's band: 4.2 and 2.2 km above the tile's mean ground (−2553.2 m). Descent pixels there cover 8.2 and 4.3 m of ground.

**The finest map is 2 m, not HiRISE's 0.25 m.** The render's texture is the 2 m tile orthoimage, so a finer map would hold no detail the render has. "Full resolution" in this sweep means 2 m.

**Same ground at every resolution.** The sweep's maps are a 6 km square around the tile centre: the whole tile at 2 m (5010 px, about 100,000 keypoints) doesn't fit LightGlue on 8 GB. The 6 m row is therefore not the workflow's map; that one is in the next section.

### 4.2 km (8.2 m descent pixels)

| Map | Map keypoints | Matches | Inliers | Status | Position error (m) | Locate (s) |
|---|---|---|---|---|---|---|
| 2 m | 34235 | 14 | 0 | no_solution | - | 0.69 |
| 4 m | 9157 | 1633 | 1632 | ok | 1.72 | 0.13 |
| 6 m | 4096 | 1851 | 1851 | ok | 0.88 | 0.09 |
| 10 m | 1793 | 1537 | 1537 | ok | 1.72 | 0.11 |
| 12 m | 1265 | 1031 | 1031 | ok | 3.28 | 0.11 |

### 2.2 km (4.3 m descent pixels)

| Map | Map keypoints | Matches | Inliers | Status | Position error (m) | Locate (s) |
|---|---|---|---|---|---|---|
| 2 m | 34235 | 3343 | 3308 | ok | 1.38 | 0.52 |
| 4 m | 9157 | 3573 | 3569 | ok | 0.54 | 0.12 |
| 6 m | 4096 | 2218 | 2218 | ok | 0.92 | 0.09 |
| 10 m | 1793 | 928 | 914 | ok | 3.23 | 0.14 |
| 12 m | 1265 | 641 | 628 | ok | 1.87 | 0.14 |

**Matching does not break at CTX-like resolution.** At 6 m it locates within 0.92 m at both ends of the band, with about 2,000 inliers. It holds out to 12 m, the coarsest map tried, within 3.3 m.

**What breaks it is the scale ratio, not coarseness.** The one failure is the *sharpest* map at the *highest* altitude: a 2 m map against 8.2 m descent pixels, a map 4× finer than the image. ALIKED's keypoints and descriptors aren't scale-invariant, so they need the two images at a similar ground scale. Map pixel over descent pixel, the sweep worked from 0.47 (2 m at 2.2 km) to 2.8 (12 m at 2.2 km) and failed at 0.24 (2 m at 4.2 km). Where in 0.24–0.47 it breaks wasn't measured. LVS's 6 m map sits at 0.73–1.4 across the band, comfortably inside the range that worked.

**The spec's fallbacks aren't needed.** The spec's fallback order (restrict matching to the LVS band, try the fine-tuned matcher, then a less-coarse map recorded in an ADR) applies only if the CTX-like map fails, and it didn't. The map stays at 6 m. That holds for perfect conditions only; see the next section.

The 6 m window map's keypoint count is the cap: its 1000 px fits in one 1024 px tile, which reached `MAX_KEYPOINTS` (4,096).

## The workflow's full-tile 6 m map, and conditions beyond perfect

The whole 1670 px map, as the navigator uses it: four tiles, 14,290 keypoints. The 2.2 km views here are rendered from the same position as above.

| Map | Map keypoints | Matches | Inliers | Status | Position error (m) | Locate (s) |
|---|---|---|---|---|---|---|
| 4.2 km, zenith sun | 14290 | 3524 | 3521 | ok | 0.63 | 0.13 |
| 2.2 km, zenith sun | 14290 | 1954 | 1944 | ok | 1.88 | 0.14 |
| 2.2 km, landing-day sun (257, 40) | 14290 | 1918 | 1911 | ok | 3.40 | 0.14 |
| 2.2 km, zenith sun, heading 45 | 14290 | 148 | 90 | ok | 12.31 | 0.15 |
| 2.2 km, zenith sun, heading 90 | 14290 | 23 | 0 | no_solution | - | 0.26 |
| 2.2 km, zenith sun, heading 180 | 14290 | 63 | 0 | no_solution | - | 0.22 |

- **The full map is a little worse at 2.2 km than the window** (1.88 m against 0.92 m, 264 fewer matches): it has 3.5× as many keypoints, most of them outside the view, for LightGlue to sort through. One position, so whether that difference is typical isn't known.
- **The landing-day sun (azimuth 257°, elevation 40°) still locates within 3.4 m.** Its shading and shadows don't match the map's lighting, and matches barely drop. One sun, one position.
- **Heading breaks it.** Turned 45° from north-up it keeps 90 inliers and misses by 12 m; at 90° and 180° it fails. ALIKED and LightGlue aren't rotation-invariant. The scenario sampler draws heading uniformly over 0–360°, so as things stand most Monte Carlo descents would fail for this reason alone. LVS knew its attitude from the IMU and warped each descent image before matching; Fidelis will need the same step (derotating by the known heading) before the Monte Carlo runs. Not part of #7, whose perfect conditions are north-up.

**Memory and time.** Peak torch GPU memory was 3.8 GB over the whole run (the 2 m map's 25 tiles and LightGlue on 34,235 keypoints), on top of Mitsuba's scene, which the script frees before navigating. Locating one image against the full map takes 0.13–0.26 s, after the map's features are extracted once.

## The GPU test (acceptance: perfect conditions, small stated tolerance)

`tests/nav/test_navigator.py::test_a_perfect_conditions_render_of_jezero_locates_within_half_a_ctx_map_pixel` renders the perfect-conditions view at 2.3 and 4.1 km above the mean ground, near both ends of the LVS band, and requires each fix within **half the map's ground pixel: 2.85 m** (its east–west ground pixel is 5.69 m, its north–south 6 m). Thousands of inliers average far below one map pixel, so missing that means something systematic.

**Why both altitudes.** Mutation testing (`spec-verify`) showed the test at 2.3 km alone was vacuous for the bug it was meant to catch. Dropping the half-pixel centre offset from the map's georeferencing moves every map point by half a pixel, about 2.85 m east and 3 m north (4.1 m). Near 2.2 km the correct fix already sits 1.88 m off (table above), and the slip partly cancelled that error: the slipped fix stayed inside 2.85 m and the test passed. Near 4.2 km the correct fix is 0.63 m off, and the same slip failed it. With both altitudes the test catches the slip. Where the 1.88 m comes from wasn't investigated; a convention error would show at both altitudes, not mainly at one.

Run it with `uv run pytest -m gpu tests/nav`.

The CPU tests check the same chain without a renderer: an analytic nadir view of a synthetic flat map has to locate within a quarter of its 1.9 m ground pixel, and the map's pixel-to-world conversion has to reproduce a tilted plane exactly from a DTM on a different grid.

## Choices worth knowing

- **ALIKED resizes inputs to 1024 px unless told not to.** LightGlue's ALIKED wrapper defaults to `resize=1024`, which would quietly turn the 1670 px, 6 m map into a 1024 px, 9.8 m one. Both images are extracted at their own resolution.
- **The map is extracted in 1024 px tiles.** ALIKED's dense features for a whole map don't fit on 8 GB: a 5010 px map needs about 13 GB for one 128-channel layer, and the prototype ran out of memory on it. Keypoints near a tile's edge get descriptors from less context; not measured.
- **Elevation comes from the tile's 4 m DTM**, the same terrain the mesh is built from. LVS's onboard elevation map was coarser, so the map's 3D points are better here than in flight.
- **A fix needs 10 RANSAC inliers.** Below that the result is a failure record (`no_solution` or `too_few_matches`), not an exception. Ten is twice RANSAC's minimal sample; random matches almost never agree beyond it. A fix with a few more than 10 inliers can still be badly wrong; the navigator reports it as `ok` with its inlier count, and the Monte Carlo counts it as the large error it is rather than the navigator second-guessing it.
- **The RANSAC threshold is OpenCV's default, 8 px.** At 2.2 km that's 34 m of ground, about six CTX map pixels.

**Caveats:** one laptop GPU, one camera position, nadir or near-nadir views, two suns, pretrained (not fine-tuned) LightGlue. The sweep's maps are block averages of the 2 m texture (CTX's optics and noise aren't modelled; see `src/fidelis/terrain/matching_map.py`), and the descent renders come from the same orthoimage as the map, so the two share everything but resolution and the renderer's shading. That makes this a best case.
