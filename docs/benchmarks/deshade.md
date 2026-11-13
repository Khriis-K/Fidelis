# Shadow reduction on the orthoimage (issue #19)

The scene's texture is a HiRISE photo, so the sun it was taken under is painted into it. The mesh renderer shades the terrain again under a simulated sun, and the painted shading never moves. `fidelis-deshade` (`src/fidelis/terrain/deshade.py`) estimates the painted terrain shading and divides it out, writing a second texture beside the first:

    uv run snakemake --profile profiles/local data/jezero/mesh/texture_deshaded.png

CPU only. On the laptop Snakemake reported 23.6 s for the first build (its preflight included) and 8 s for a rebuild. **Nothing renders from the new texture by default.** Every result so far used the photo texture (`texture.png`) and stays reproducible. `terrain_scene(mesh_dir, sun, texture="texture_deshaded.png")` picks the new one; the stress case (#43) is its first user.

## Why this was built now

A laptop check before building (recorded on issue #19, 2026-11-13) predicted that de-shading makes renders under the five realistic suns look **more** like the map, not less, because those suns are within 39° of the map's. So this step is not a fix to the realistic failure rate. It is groundwork for the stress case, where the simulated sun is on the far side of the sky from the baked one.

## Method

On a gently sloping diffuse surface, brightness is close to albedo × (1 + p·zx + q·zy), where zx and zy are the slopes east and north and p, q depend on where the sun was. p and q are fitted from the photo itself, by least squares of its local brightness against the 4 m DTM's slopes, in blocks of about 1 km. The fitted shading is then divided out of the 2 m orthoimage.

**Why fit it, when the sun angles are sourced?** The orthoimage is a patchwork of at least four HiRISE images taken under different suns, and USGS doesn't publish the seams (`docs/research/ctx-and-hirise-sun-angles.md`). One sourced sun for the whole tile would be wrong on part of it. The photo's shading is also about 1.4 times stronger than a plain diffuse model predicts at the sourced sun (the #19 comment), and the fit takes whatever strength is there.

**Settings** (`config/jezero.yaml`, `deshade:`; all Fidelis choices): 1 km blocks, structure finer than about 100 m counts as shading, and the fitted shade is never taken below 0.3 of flat ground's.

## Results (laptop, 2026-11-13, from `data/jezero/mesh/texture_deshaded.json`)

**The metric: how much brighter are slopes facing the sun than slopes facing away?** The ticket asks for "reduced shadow contrast in shadowed regions". This photo has almost no cast shadow (see Limitations), so the dark regions measured here are slopes that face away from the sun. Slopes steeper than 5°, brightness taken against the local average, over the tile's centre (within 2.5 km east–west of the middle). The sun used to sort the slopes is the sourced one for the HiRISE image there (bearing 267.37°), not the fitted one, so the fit isn't marking its own work.

| | Sun-facing ÷ sun-away brightness |
|---|---|
| Photo texture | 1.372 |
| De-shaded texture | 0.984 |

1.0 means no shading left. The photo's sun-facing slopes are 37% brighter; after, they are 1.6% darker (slightly over-corrected).

**Lit, flat ground is not degraded.** On ground flatter than 2°, the de-shaded texture keeps 0.974 of the photo's fine contrast (structure finer than about 8 m, the descent camera's pixel scale). This is the guard against the null case: blurring the whole image flat would also lower shading contrast, and scores under 0.5 on this check in the tests.

**Clipping.** 0.02% of pixels came out brighter than 255 and were clipped. A DTM with nodata pixels is refused, as the mesh builder does.

**The fitted suns agree with the sourced ones in the centre and the east, and not on the west edge.** The fit is never given a sun. Fitted bearing per 1 km block, by block column from west to east (10 rows of blocks in each), against the sourced bearing of the HiRISE image the research note found on top there:

| Block column (west to east) | Fitted bearing: median (min–max) | Median r² | Sourced bearing there |
|---|---|---|---|
| 0 | 239° (233–266°) | 0.38 | 259.8° (PSP_002387, west edge) |
| 1 | 263° (233–273°) | 0.27 | between the west-edge and centre samples |
| 2 | 272° (260–282°) | 0.26 | 267.4° (ESP_045994, centre) |
| 3 | 272° (253–282°) | 0.26 | 267.4° |
| 4 | 271° (264–304°) | 0.21 | 267.4° |
| 5 | 275° (272–293°) | 0.11 | 267.4° |
| 6 | 273° (252–287°) | 0.09 | 267.4° |
| 7 | 226° (205–273°) | 0.11 | between the centre and east-strip samples |
| 8 | 212° (192–265°) | 0.04 | 219.2° (ESP_048842, east strip) |
| 9 | 200° (158–262°) | 0.04 | 219.2° |

r² is the share of a block's local brightness the fit explains. Columns 2–6 sit within 4–8° of the sourced centre sun. The east strip is in the right part of the sky (the median is 7–19° off) but scattered, and the fit explains almost nothing there. The west edge is about 20° off; whether that is the fit or a different image on top there isn't known (the research note sampled 17 points and called the west edge its narrowest call). Single blocks stray much further than the medians, as the ranges show. Fitted *elevations* read low in the west and centre (median 29–36° in columns 0–4 against a sourced 34° and 42°), which is the "stronger than diffuse" shading again, not a different sun.

**Relit renders** (`scripts/check_deshade.py`, four renders on the laptop GPU, nadir from 2.2 km, 256 spp; numbers from its `relit.json`). Under the landing-day sun (257°, 40°) and a morning sun on the opposite side (87°, 25°, the #43 candidate), how alike the two suns' renders are (correlation of high-passed log brightness):

| Texture | Landing sun vs morning sun |
|---|---|
| Photo | 0.726 |
| De-shaded | 0.628 |

Baked shading doesn't move with the sun, so it holds the two renders together; removing it lets them differ more. The picture (`data/jezero/render/deshade/relit.png`, generated by the script and not committed, like every render) is a check by eye, and the difference at this crop is subtle. Under the morning sun the photo texture's craters look flatter, with brightness on both walls (the painted western light and the simulated eastern light); with the de-shaded texture their western walls are the brighter ones, as an eastern sun should make them. Under the landing sun the two textures look nearly the same, as predicted. The correlation above is the number; the picture only says it changes in the expected direction.

## Limitations

- **Cast shadows are not removed.** Only shading that goes with slope. At the photo's sun (42° high) cast shadows cover about 0.009% of the tile centre at 4 m (the #19 comment), so little is lost here, but a photo taken under a low sun would need more.
- **Nothing finer than the 4 m DTM is removed.** Shading from rocks and small scarps stays in the texture. The descent camera's pixels are 4.3 m or coarser, so it mostly can't resolve that either; this isn't measured.
- **The fit can't tell shading from albedo that happens to follow slope** (dust that settles on one side of ridges, say). On a synthetic scene whose brightness has nothing to do with its slopes, the step still changes pixels by under 2% rms, less than a fifth of what it removes from a shaded one, and a block's fitted bearing can be off by several degrees; the tests allow 12° (`tests/terrain/test_deshade.py`).
- **Where the fit is weak, little is removed.** In the east strip the fit explains a median 5% of local brightness, so the baked shading there is mostly untouched or the terrain is too flat to show any; the two aren't separated.
- **Blocks blend across seams.** Fitted values hold at block centres and are blended between them, so within about a block of a seam between two HiRISE images the correction is a mix of both suns.
- **Steep slopes.** The model is linear in slope. A diffuse surface also dims by the cosine of its slope whatever the sun's bearing, about 6% at 20° and 13% at 30° [Computed], and that part is left in. The 0.3 floor caps the brightening of slopes facing away.
- **The CTX ring is not de-shaded.** Beyond the 10 km tile the scene is CTX on a 20 m DTM, with CTX's own baked suns. Views that reach the ring still carry it.
- **The metric is checked only over the tile centre**, where one sourced sun holds. The east strip and west edge have no independent check beyond the fitted bearings above.
- **It is a workflow rule, not part of the default build.** `snakemake jezero` doesn't make the de-shaded texture; it is built when asked for by name, so that earlier results aren't sent back through the queue.
- **No navigation result yet.** This shows the texture and the renders changed as intended. What it does to the navigator under a stress sun is #43's question.
