# The wide Jezero scene: a CTX ring around the HiRISE tile (issue #33)

On the 10 km HiRISE tile alone, the scenario sampler had to cap camera tilt at 3° and offset at ±400 m, or tilted views would see past the terrain. Flight swung about −12° to +5°, and the true-position σ was about 1.1 km. This issue surrounds the tile with a ring built from the USGS Mars 2020 TRN CTX products, which lifts that cap. Built and measured on 2026-11-02, on the laptop (NVIDIA GeForce RTX 5070 Laptop GPU, 8,151 MiB).

    uv run snakemake --profile profiles/local jezero
    uv run python scripts/check_wide_scene.py config/jezero.yaml data/jezero data/jezero/render/wide

## What was built

- **Data.** CTX DTM (20 m) and orthoimage (6 m), 46 MB together, fetched by `fidelis-fetch` and checked against SHA-256 pins in `config/jezero.yaml` (pinned from the first download). A workflow test checks that all four downloads (two HiRISE, two CTX) carry a pin.
- **Not on the HiRISE grid.** The research note said CTX shares the HiRISE grid origin. It doesn't: CTX sits 0.749 m east and 3.140 m north of a 6 m HiRISE pixel corner (all four GeoTIFF headers). The note now carries a correction. Wherever the two meet, CTX values are bilinear samples.
- **`fidelis-scene`** cuts a CTX window of 26,100 × 26,040 map metres, centred on the tile centre and sized so the ring's outer vertices reach at least 13 km of map from it every way (12.33 km of ground east–west). The window has no DTM or ortho nodata.
- **Heights blended.** Over the tile, CTX − HiRISE (CTX bilinear at the 4 m tile's pixel centres) had a **median of +0.32 m and an sd of 3.11 m**. The research note's +0.54 / 4.09 m came from a different comparison: the whole overlap, on CTX's 20 m grid. Across a 200 m band inside the tile mesh's edge, the heights ease linearly from CTX's (at the outermost vertices) to HiRISE's own.
- **Brightness matched.** CTX was given the HiRISE tile's mean and spread, compared at 6 m: **gain 1.248, offset −68.0 DN**. The tile's texture eases into the matched CTX across the same 200 m band.
- **Ring mesh.** One vertex per CTX pixel centre, with a hole where the tile goes. A strip of triangles stitches the hole's edge to exact copies of the tile mesh's outermost vertices: 1,709,126 vertices (the hole's vertices are kept but unused) and 2,901,024 triangles, a 72 MB PLY plus a 10 MB texture. The tile mesh is unchanged in size: 6,275,025 vertices and 12,540,032 triangles.
- **Curvature.** Both meshes and the orbital map's 3D points drop by d² / 2R from the tile centre (R = 3,396,190 m): 3.7 m at 5 km, 24.9 m at 13 km.
- **Matching map.** It now spans the whole scene: 4339 × 4349 pixels at 6 m on the tile's grid. Inside the tile it is the tile texture averaged as before. Outside, it is the matched CTX texture resampled bilinearly. The orbital map reads heights from the tile DTM inside the tile and from the ring DTM beyond it.

## Acceptance checks

| Criterion | Check | Result |
|---|---|---|
| Reproducible CTX download | SHA-256 pins; workflow test that every download is pinned | pinned; test passes |
| No crack at the boundary | `test_ring.py`: with vertices identified by exact float32 position, every edge of tile + ring is shared by two triangles except on the scene's outer edge; the triangles' area seen from above equals the scene's exactly (no holes, no overlaps); every triangle faces up | passes on a synthetic tile whose grid is off the ring's, as the real one is |
| Height step within a stated tolerance | **Tolerance: 1 mm** (`HEIGHT_STEP_TOLERANCE_M`): the tile's outermost heights against the CTX surface at the same points | passes; unblended, the synthetic step is the 0.54 m offset |
| Curvature against d² / 2R | against the exact sphere R − √(R² − d²) to 1 mm out to 13 km; mesh vertices and orbital-map points both carry it | passes |
| Render time, GPU memory, seam | below | measured |
| Flight-like tilt (≥ 15°) and offset (≥ 2σ) | `load_config` with tilt 0–18°, offsets ±2.5σ (±2750 m) | passes |
| Jezero exclusion over the wider footprint | `jezero_exclusion` in the config covers both mosaic footprints, and a test checks the scene lies inside it | passes |

**Test strength (spec-verify).** For each criterion, one targeted break was put into the code or config and the matching test was rerun. **The tests caught all 15 breaks.** Breaks tried: an unpinned CTX ortho; downloads restricted to HiRISE; stitch copies made without curvature (a crack); stitch triangles facing down; a blend ramp starting at the tile bounds instead of the mesh edge; a blend applied everywhere; no brightness match; drop d²/R; a flat mesh; a flat orbital map; ring heights ignored; HiRISE never pasted into the map; tilt left at 3°; the camera-over-tile guard removed; an exclusion box left at the HiRISE footprint. Every file was restored byte for byte after each run.

## How far the sampler can now reach

The sampler's check needs to know how much lower a view's edge can be than the ground under the camera. Whole-scene relief is **1203 m**, because the 26 km square takes in Jezero's crater rim. But the rim is uphill, and uphill ground only shortens a view. So the bound is the tile's highest ground down to the scene's lowest, curvature included: **351.3 m**, configured as 352. `load_config` now refuses any offset that could take the camera off the HiRISE tile, since the bound holds only while the camera is over it.

New bounds: tilt 0–18° (the top of the 7–18° pre-flight averages) and offsets truncated at 2.5σ = ±2750 m. Worst case: 2450 m + 352 m at 18° through the image corner, from 2750 m out, reaches about 11.8 km of the 12.33 km east–west half-width. 20° no longer fits, and the config test checks that it is refused.

## GPU cost (laptop)

| | 10 km tile (`mesh-render.md`) | Wide scene |
|---|---|---|
| Scene load | 1.5–1.9 s | 2.14 s |
| GPU memory once loaded and rendered | 2,102 MiB | 3,082 MiB |
| Render, 1024² at 64 spp | median 0.053–0.058 s | median 0.064 s (max 0.071) |
| Render, 1024² at 256 spp (`mesh_spp`) | not measured | median 0.231 s (max 0.276) |

The wide-scene timings come from five views: the sampler's four worst cases (highest altitude, 18° tilt, leaning outward from a 2750 m offset, an image corner turned outward) plus one nadir seam view. All five passed the blank-line check, which also fails a view that runs off the scene along a whole image edge. The 10 km numbers are nadir views only, so the two columns are not the same views. GPU memory is the whole GPU's, as nvidia-smi reports it.

## Seam visibility

The seam view is straight down from 2.2 km above the tile's east edge, north up, so the seam runs down the image's middle column. In the 256 spp render, divided by its mean brightness:

- **Brightness step across the seam: 0.0028.** At 40 control columns away from it: median 0.0081, max 0.039.
- **Mean horizontal gradient at the seam: 0.017.** Controls: median 0.029, max 0.059.

So there is **no line and no brightness step at the seam beyond what ordinary terrain columns show**. That was the matcher-bait worry in the ticket.

**There is a sharpness change, though.** In the same image, the CTX side has **0.24–0.35×** the fine detail (mean |Laplacian| in 100 px bands) of the HiRISE side, at every distance out to 500 px. That is the data, not a blending fault: a 6 m texture on a 20 m mesh can't hold the detail of a 2 m texture on a 4 m mesh, and the camera's 4.3 m pixel at 2.2 km would resolve it. A blend could hide a step, but it can't add detail CTX never had.

## Caveats

- **Fewer features over the ring.** Oblique views see the ring at their far side, where it is smoother than real terrain, so the matcher will likely find fewer keypoints there. How much that moves the failure rate is not measured.
- **Same photo, larger area.** In the ring, the render texture and the matching map both come from the CTX orthoimage. That flatters matching, as the inner tile already did with HiRISE. This is an existing limitation, now over about 7× the area.
- **North-west corner.** USGS made no CTX DTM there and filled the orthoimage from a CTX image orthorectified to HRSC terrain. The DTM has no nodata there, but which source fills it isn't stated and wasn't checked.
- **One east–west scale for the whole scene.** cos(latitude) at the tile centre. Over ±13.85 km north–south it changes by up to 0.14%, about 17 m at the scene's corners. The mesh and the map share that frame, so the navigator sees a consistent world, just one 0.14% off true shape at the far corners.
- **Orbital-map heights across the stitch** come from the ring DTM's bilinear surface, not the stitch triangles themselves. The two agree at both edges of the strip, which is at most about 30 m wide.
- **Earlier results predate this.** `docs/benchmarks/plain-mc.md` used the 10 km tile and the clipped sampler (config SHA-256 `f764e270…`). They are not results for this scene.
- One laptop GPU, one sun (landing day, 257° / 40°), five views.
