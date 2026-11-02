# Descent altitude and tile size

**Question:** at what altitudes did Perseverance's Lander Vision System (LVS) image the ground, and does Fidelis need, and can it get, a bigger Jezero tile than the current 2 km? Investigated 2026-11-01, after issue #5 found that a 90° camera sees past the 2 km tile above about 1 km.

## Answer

- **LVS worked from 4.2 km down to 2.2 km above the ground**, so the descent camera sees 4.4 to 8.4 km of ground, straight down. The 2 km tile covers none of that band.
- **A 10 km tile is available and renders fine, but only at reduced resolution.** At 4 m mesh spacing with a 2 m texture, it renders a 1024 × 1024 LCAM view in 0.056 s using 2.1 GB of GPU memory. At 2 m / 1 m it filled the laptop's 8 GB GPU and rendered corrupt images, with no error raised.
- **The current mesh is also 5.4% too wide east–west**, independent of size. That should be fixed in the same change.

Recommendation: rebuild the Jezero tile at about 10 km × 10 km, centred on the landing site as now, with a 4 m mesh, a 2 m texture, and east–west distances corrected. Keep the 1 m / 0.25 m tile only if a later use needs views below about 1 km.

## LVS altitudes (source)

NASA/JPL, *Mars 2020 Perseverance Landing Press Kit* (February 2021), pp. 21–22, https://www.jpl.nasa.gov/news/press_kits/mars_2020/download/mars_2020_landing_press_kit.pdf (read 2026-11-01). It's a pre-landing description of the plan, not a flight reconstruction:

- The Lander Vision System "operates when the spacecraft is between about 2.6 to 1.4 miles, or 4.2 to 2.2 kilometers, above the ground."
- After the camera turns on, LVS uses an initial 5 seconds to take three images and compute a coarse position, then takes and processes an image every second for finer fixes.
- Around back-shell separation, the rover picks a landing site using the last LVS fix. The powered descent vehicle separates at about 2.1 km above the surface.

Not found: flown (as opposed to planned) altitudes, and the LVS map's footprint. Johnson et al.'s *Mars 2020 Lander Vision System Flight Performance* (AIAA SciTech 2022) and Cheng, Ansar & Johnson 2021 have them, but the publisher pages returned 403 and NTRS holds metadata only.

**What this means for the camera.** LCAM is 1024 × 1024 px over 90° (Maki et al. 2020; `docs/camera-conventions.md`), so a nadir view covers 2 × altitude across, and one pixel covers altitude / 512 at the centre:

| Altitude | Ground across (nadir) | Ground per pixel (centre) |
|---|---|---|
| 4.2 km (LVS starts) | 8.4 km | 8.2 m |
| 2.2 km (LVS ends, near the Monte Carlo handoff image) | 4.4 km | 4.3 m |

At the image edges, rays hit the ground at 45°, so pixels there cover about twice as much.

## The projection stretches east–west by 5.4%

The USGS mosaics are equirectangular with the standard parallel at the equator (`lat_ts=0`), so one map metre east–west is cos(latitude) of a real metre. GDAL's inverse projection at the landing site gives **1000 map metres east–west = 948.6 m of ground; north–south = 1000.0 m** (cos 18.4447° = 0.9486). `tile.py` and `mesh.py` treat map metres as ground metres, so the mesh is 5.4% too wide east–west. That makes east–west slopes 5.4% too gentle and shadows cast east or west 5.4% too long. Any pose given in mesh metres is off by the same factor in x. Nothing in the repo corrected for this before this note.

Fix: scale local x by cos(latitude at the tile centre) when building the mesh, and record the factor in `mesh.json`. Across a 10 km tile, cos(latitude) varies by less than 0.05%, so one factor is enough.

A smaller effect: Mars curves away from a flat tile by d² / 2R, which is **3.7 m at 5 km from the centre** (R = 3,396,190 m). That's below a 4–8 m pixel at LVS altitudes. It's worth a line in the limitations, not a fix yet.

## How far the data extends

Read from `data/jezero/raw/` at 20 m sampling (DTM nodata, or ortho value 0, counts as missing):

- The mosaics span 21.4 × 21.5 km of map, but only 66.1% of that has data. The landing site sits off-centre: 7.9 km from the east edge and 8.2 km from the south edge.
- **The largest square of valid DTM and orthoimage centred on the landing site is 10.14 km** of map metres (9.6 km of ground east–west). A 12 km square would be 98.2% valid, so it would have holes.
- Not checked: whether a bigger valid square exists if the tile isn't centred on the landing site.

So a centred 10 km tile (9.49 km of ground east–west once corrected, 10 km north–south) leaves this much margin around a nadir view:

| Altitude | Nadir footprint | Spare ground each side (east–west) | Tilt or offset that still fits |
|---|---|---|---|
| 4.2 km | 8.4 km | 0.5 km | about 3.5° of tilt, or 0.5 km of offset |
| 2.2 km | 4.4 km | 2.5 km | about 20° of tilt, or 2.5 km of offset |

The Monte Carlo estimate uses the handoff image near 2.2 km, so it has plenty of room. Showcase descents that start at 4.2 km only fit if they stay close to nadir and close to the centre.

## Can the laptop render it (measured)

10 km tiles cut from the raw mosaics with area averaging, meshed with `build_terrain_mesh`, and rendered with `render()` (`cuda_ad_rgb`, RTX 5070 Laptop GPU, 8 GB). Views are nadir, LCAM intrinsics, 64 spp, west sun at 30°, five renders per altitude after a warm-up:

| Mesh / texture | Vertices / triangles | PLY | GPU memory after load | Render, 4.2 km | Render, 2.2 km | Image |
|---|---|---|---|---|---|---|
| 1 m / 0.25 m (current, 2 km tile) | 4.0 M / 8.0 M | 0.19 GB | not measured | 0.053 s* | — | fine |
| 2 m / 1 m | 25.0 M / 50.1 M | 1.15 GB | +7.7 GB (of 8.2) | 2.96 s | 3.95 s | **corrupt**: lower ~11% streaked and partly black |
| 4 m / 2 m | 6.3 M / 12.5 M | 0.29 GB | +2.1 GB | 0.056 s | 0.057 s | fine, no missing rows |

\* From `docs/benchmarks/mesh-render.md`, at 900 m.

- **The 2 m mesh failed silently.** Mitsuba raised no error. The image came back with streaked rows and a black wedge where the south of the mesh should be, and renders were 55–75× slower. The most likely cause is running out of GPU memory, with the ray-tracing structure built incompletely; that cause is inferred, not confirmed. A render-validity check (no rows of exact zeros) should guard any bigger tile.
- **4 m is not a loss at LVS altitudes.** The mesh vertices (4 m) and texture pixels (2 m) are both still at or below LCAM's 4.3–8.2 m pixel. Supersampling a 0.25 m texture into an 8 m pixel just adds noise; a pre-averaged texture is closer to what the camera's pixel actually integrates.
- Not measured: CARC GPUs (40–80 GB) would hold a finer mesh; oblique views; splat training cost over a 10 km scene.

## What changes if we do it

- **Tile config:** `size_m` 2000 → 10000, plus a mesh resolution and a texture resolution (new), with the east–west scale applied in `mesh.py`.
- **Checks:** re-run `scripts/check_terrain_mesh.py` and `scripts/check_mesh_render.py`. The registration check has to compare in corrected metres.
- **Matching map:** the CTX-like map grows to match, still at 6 m.
- **Scenario sampler (#6):** altitude band 2.2–4.2 km from the press kit, with tilt and offset limits from the margin table above, or an explicit off-tile check.
- **Splats (#20):** a 10 km scene at 4 m detail, not 2 km at 1 m. Cost not measured.

Scratch scripts for these measurements were not committed. The numbers above are the ones they printed.
