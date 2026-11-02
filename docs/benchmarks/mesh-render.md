# Mesh render checks (issue #5)

The Mitsuba path-traced renderer (`src/fidelis/render/mesh_render.py`) on the Jezero mesh from issue #4, measured on the laptop (RTX 5070 Laptop GPU, 8 GB; Mitsuba 3.9.1, `cuda_ad_rgb`) on 2026-11-01. Reproduce with:

    uv run python scripts/check_mesh_render.py data/jezero/mesh data/jezero/render

Pose and pixel conventions are in `docs/camera-conventions.md`.

## Projection matches OpenCV

The unit test renders three small glowing spheres with an oblique, rolled camera (96 × 80 px, fx = fy = 70, principal point (40.3, 45.8)) and requires each sphere's image centroid to land within 0.1 px of `cv2.projectPoints`. The first run missed by up to 14.4 px in x and 12.6 px in y. The sphere aimed at the principal point rendered at (54.70, 33.19) instead of (40.3, 45.8): mirrored about the film centre (47.5, 39.5). That pinned the error to the sign of Mitsuba's `principal_point_offset`. With the sign flipped, all three spheres pass at 0.1 px.

## What the renderer does

- Mitsuba's `path` integrator, `max_depth` 3: the camera ray, a shadow ray to the sun, and one bounce of light off the terrain.
- Smooth shading: Mitsuba's per-vertex normals (the PLY has none), so shading varies smoothly across each 1 m triangle instead of in flat facets. Shadow rays still use the true triangles.
- The sun is a directional light with irradiance π, so sunlit flat ground renders at albedo × sin(elevation), where albedo is the orthoimage value / 255. A unit test checks that on a synthetic tile, within 3%.
- No sky light. Shadows get only light bounced off the terrain, so they're darker than on Mars, where a dusty sky fills them.
- Output: a float32 image of linear radiance, one channel. That's a design choice, not something #5 asked for: LCAM is a monochrome camera, and the texture is greyscale anyway. If splat training (#20) wants three channels, they'd be three copies of this one. Exposure, noise and blur are left to later steps.
- `terrain_scene(mesh_dir, sun)` loads the mesh once per sun. `render(scene, pose, intrinsics, spp, seed)` then renders any number of poses.

## Descent-camera images (acceptance: given pose, intrinsics and sun, an image)

The target camera is LCAM-like: **1024 × 1024 px, 90° field of view** (Maki et al. 2020; see `docs/camera-conventions.md`). Test views are nadir, north up, 900 m above the tile's mean ground (−2571.8 m). A 90° camera sees 2 × altitude across, so 900 m covers 1.8 km of the 2.0 km mesh.

**Altitude limit.** On this 2 km tile, a nadir 90° camera above about 1000 m sees past the mesh's edge. LVS worked from 4.2 km down to 2.2 km above the ground (NASA/JPL landing press kit), entirely above that limit. A 10 km tile at 4 m mesh spacing fits on the laptop and renders at the same speed. See `docs/research/descent-altitude-and-tile-size.md`. This needs a decision before the seeded scenario sampler (issue #6) fixes its altitude range.

## Render time (acceptance: time per image at the target resolution)

1024 × 1024 px, 64 samples per pixel (spp), five renders per sun with different seeds:

| Sun (azimuth, elevation) | Scene load | Render, median (min–max) | 1024 spp |
|---|---|---|---|
| east (90°, 10°) | 2.1 s | 0.053 s (0.052–0.054) | 0.78 s |
| west (270°, 10°) | 1.9 s | 0.054 s (0.053–0.056) | 0.78 s |
| east (90°, 30°) | 1.8 s | 0.053 s (0.051–0.054) | 0.76 s |
| west (270°, 30°) | 1.8 s | 0.053 s (0.052–0.054) | 0.77 s |
| south (180°, 60°) | 1.8 s | 0.052 s (0.052–0.068) | 0.75 s |

- **Per image: about 0.053 s at 64 spp** once the scene is loaded and Dr.Jit's kernels are compiled. Loading the 185 MB mesh takes about 2 s per sun.
- The first render in a fresh process is slower while kernels compile: **0.54 s** with a cold kernel cache, 0.12 s on the next run, when Dr.Jit's on-disk cache had them.
- Times are from `time.perf_counter()` around `render()`, which includes copying the image off the GPU.

**Caveats:** one laptop GPU, nadir views only. Oblique views and lower altitudes trace the same number of rays, but more complex bounce paths could change timing; CARC GPUs are untested. A cost measurement for the multifidelity study needs a pinned GPU model (see `CLAUDE.md`).

## Noise and determinism (acceptance: same inputs, same image, up to a stated tolerance)

**Stated tolerance: with the same scene, pose, spp and seed on the same machine and Mitsuba variant, no pixel differs by more than 1e-6** (`RENDER_DETERMINISM_TOLERANCE`; sunlit pixels are around 0.05 to 0.4).

- `scalar_rgb` (CPU, the unit tests) repeats bit for bit. A unit test asserts the tolerance and that a different seed changes the image.
- `cuda_ad_rgb` (the GPU): the largest same-seed difference over five suns was **1.8e-7**. It isn't zero because the GPU film adds samples atomically, in no fixed order.
- **Monte Carlo noise at 64 spp:** RMS difference from a 1024 spp render of the same view was **1.2–1.6% of the median pixel**. Two 64 spp renders with different seeds differ by 1.7–2.2% RMS. 64 spp is the setting these measurements use, not a requirement. The sample count for real renders is a fidelity choice, which belongs to the fidelity levels (#10).
- Not checked: CPU against GPU, or one GPU model against another. They'll differ by more than 1e-6, since floating-point order differs. Comparisons across machines should use the noise level above as the yardstick, not the determinism tolerance.

## Shadows move with the sun (acceptance: documented visual check across at least two angles)

**Unit test, geometry with a known answer.** A synthetic tile has a flat floor and a 6 m tall plateau. Under a sun in the west at 45° and in the north at 30°, the ground is in shadow (below 5% of sunlit) from 2 m past the plateau's edge to 1.5 m short of the predicted tip, height / tan(elevation) away. It's fully lit 2.5 m past the tip, beside the shadow, and 15 m out on the sunward side.

The sunward check had to move 15 m out, for two reasons found by measuring (45° sun, 64 spp, with and without the bounce):

- **Bounce light.** Ground in front of the sunlit wall also sees the wall, so it's brighter than open ground. At 6 m from the plateau's centre (2.5 m from the wall's foot) it was 5.9% brighter under the 45° sun, all of it bounce (0.0% with `max_depth` 2), and 9.8% brighter under a 30° sun. Under the 45° sun it was still 5.1% brighter at 8 m, and under 3% beyond about 12 m. A unit test checks the bounce is there (it fails if `max_depth` drops to 2).
- **Smooth shading normals.** Within about 1 m of the foot, ground renders 30–40% brighter even with no bounce at all. Mitsuba computes per-vertex normals for a PLY without them, averaging the faces around each vertex. So the ground vertex at the foot of a 6 m, near-vertical wall borrows some of the wall's tilt towards the sun. Smooth normals are the usual choice for a height map, and on Jezero's gentle slopes the effect is small. It's only this large at artificial cliffs like the test plateau.

**Real tile, visual check.** `scripts/check_mesh_render.py` writes `shadows_low_sun.png` (the central 256 × 256 px, about 450 m across, east sun at 10° beside west sun at 10°) and `shading.png` (east 30°, west 30°, south 60°) to `data/jezero/render/`. Checked by eye on 2026-11-01:

- **East sun at 10°:** the ridge that crosses the view's upper left throws a black shadow strip onto the ground west of its west-facing scarp.
- **West sun at 10°:** the same scarp is brightly lit and its shadow is gone; the dark sides of the ridges and dunes switch to the east.
- **30° and 60°:** almost no cast shadows. Few slopes in this part of Jezero are steeper than 30°: pixels darker than 10% of the median are 0.9% (east) and 2.4% (west) of the view at 10°, and 0.0% at 30° and 60°. At those angles the sun changes shading, not shadows: east and west light opposite faces of the same ridges, and the 60° sun flattens the relief.

**Caveat: the texture has HiRISE's own lighting in it.** The orthoimage is a photo, so its shading and shadows stay where HiRISE's sun put them, whatever sun we render. Rendered sun angle changes the shading and shadows of the 1 m geometry only; the photo's own pattern stays fixed, which will understate how much sun angle changes the descent image. Removing it is issue #19.

## The orthoimage isn't calibrated reflectance

Issue #4 asked this issue to check the USGS product description before treating the orthoimage as reflectance. The USGS page for the HiRISE orthomosaic (doi:10.5066/P9QJDP48, read 2026-11-01) describes 8-bit "DN" values that were tone-matched between images by subtracting constant DN offsets (10 for two images, 9 for another), with no seam blending. It says nothing about I/F calibration, a stretch, or a conversion to reflectance. So `value / 255` is a relative brightness that's consistent across the mosaic, not a physical albedo. That's fine for matching, which cares about relative contrast. But absolute radiance values from this renderer aren't physical, so exposure should be set relative to the image, not in physical units.
