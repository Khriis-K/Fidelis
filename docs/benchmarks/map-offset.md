# The whole-map offset between the LVS map and the rendered scene (issue #40)

ADR 0003 measures navigation error in the **map frame**: against where the matching map puts the ground, as LVS was judged. Fidelis renders from the HiRISE orthoimage, but the real LVS map is the USGS CTX orthomosaic (`ctx_ortho` in `config/jezero.yaml`). Any single shift between the two would count as navigation error if left in, so it is measured once, here, by registering the two orbital images to each other. It is never estimated from descent results, since that would also hide navigator bias. Measured on 2026-11-11 on the laptop CPU (no GPU).

    uv run snakemake --profile profiles/local jezero_map_offset
    uv run python scripts/check_map_offset.py data/jezero/tile/ortho.tif data/jezero/raw/ctx_ortho.tif OUT_DIR

## Result

**CTX minus HiRISE: +1.99 m east, −1.49 m north** (map metres), **±0.4 m east and ±0.6 m north**. On the ground that is +1.89 m east (the projection stretches east–west by 1/cos(latitude), scale 0.9486), so **2.41 m in all**. Recorded in `config/jezero.yaml` as `map_offset` (`east_m: +1.99`, `north_m: -1.49`, in map metres, plus `ground_m` for the scene's ground-metre frame).

Sign: CTX shows the ground that HiRISE shows at (x, y) at (x + 1.99, y − 1.49). So a navigator matching HiRISE-rendered views against the CTX map should land about 2 m east and 1.5 m south of truth while doing nothing wrong. The map-frame error (#35) subtracts this.

For scale: it is 6% of the 40 m failure line. It is far inside the 300 m absolute tie error the LVS map was allowed, and smaller than the ~14 m mean offset between two independent JPL Jezero maps (Cheng et al. 2021 pp.19–20, via `docs/research/lvs-error-sources.md`).

**It doesn't affect any result so far.** Today's matching map is HiRISE-derived over the tile, so its offset is zero by construction (ADR 0003). The number matters once the CTX map is swapped in (#36).

## Method

- **Area.** The HiRISE tile (`data/jezero/tile/ortho.tif`, 2 m, the texture the scene renders) against the CTX ortho over it. The windows cover a 9,984 m square centred on the tile: map x 4,585,879–4,595,863 m, y 1,088,315–1,098,299 m. The CTX ring outside the tile isn't used, since there the scene *is* CTX.
- **CTX stays on its own pixels.** CTX's grid sits 0.749 m east and 3.140 m north of the tile's 6 m corners. Instead of resampling CTX, the 2 m tile is sampled bilinearly at the centres of the nine 2 m cells inside each 6 m CTX pixel and averaged. So any interpolation error is on the 2 m scale. Why this matters is under "Checks".
- **Windows.** 13 × 13 non-overlapping windows of 128 CTX pixels (768 m). Each window is normalised (mean 0, sd 1, so brightness differences don't count) and Hann-tapered (so its edges don't).
- **Phase correlation, iterated.** Each window's shift is the peak of the phase correlation, refined to sub-pixel by a parabola through the peak on each axis. Because both windows share the same taper, one pass is pulled towards zero shift (by 10–20% on the synthetic test texture). So the CTX window is shifted back by the estimate (Fourier shift) and correlated again until the correction is under 10⁻⁴ pixel.
- **Whole-map offset** = the median of the window shifts. Windows with a correlation peak under 0.1 are dropped as too weak, and so is any window more than max(5 robust sd, 30 m) from the median, as a wrong peak. 30 m sits above the LVS map's 18 m (99%) local-distortion requirement, so real distortion of the allowed size is kept. **On the real pair, neither rule dropped anything:** all 169 windows were used, with peaks of 0.23–0.49 (median 0.38).
- **Residual** = each window's shift minus the offset. A plane fit across the tile (`trend_m_per_km`, map metres per map km) shows whether it varies systematically.

Code: `src/fidelis/terrain/map_offset.py`. Tests: `tests/terrain/test_map_offset.py`. The synthetic pairs there are a broadband texture (1,500 waves, 15 m to 1 km) evaluated exactly at every pixel centre, with the CTX copy on a grid offset like the real one. The tests recover a known 14 m / −9 m shift through a brightness change, shifts of ¼, ½ and ¾ of a CTX pixel, zero from identical images, and a 3 m/km warp as a trend rather than absorbing it.

**Test strength (spec-verify).** For each of the issue's four acceptance criteria, targeted breaks were put into the code or config, and the matching tests were rerun. **The tests caught all 13 breaks.** The breaks were:

- north taken from +rows;
- one phase-correlation pass instead of iterating;
- CTX resampled onto the tile grid (the pixel-locking bug);
- no brightness normalisation;
- trend slopes zeroed;
- residuals not taken against the offset;
- outliers kept;
- weak windows kept;
- a featureless window reported as confident;
- standard error over n instead of √n;
- windows not centred;
- the config carrying the old biased +2.21 m;
- the config's north sign flipped.

The first run found two tests that couldn't catch their break. The brightness test checked only the shift, which phase correlation barely changes, so it now also requires the response to be unchanged. The config test matched a bare "+2.21" that appears elsewhere in this write-up, so it now matches the config's own `key: value`. Every file was restored byte for byte after each run.

## Checks (`scripts/check_map_offset.py`)

**Injection on the real pair.** The CTX file's geotransform was moved by a known amount (exact, no resampling), and the measured offset had to move by the same amount. Over ten shifts (¼ to ¾ of a pixel, whole pixels, and up to ±20 m on both axes), the **largest error was 0.10 m** (rms 0.04 m).

That check caught a bug in the first version, which resampled CTX bilinearly onto the tile's 6 m grid. Whole-pixel shifts came back exactly, but fractional ones came back up to **0.51 m** short, pulled towards whole pixels ("pixel locking"). The real pair sits at a fractional offset, so the first version reported +2.21 / −1.35 m, about 0.2 m off on each axis. Keeping CTX on its own grid fixed it. The synthetic test at ¼, ½ and ¾ pixel now guards it.

**Window size.**

| Window | Windows | East (m) | North (m) | Residual rms (m) | Residual 99th pct (m) |
|---|---|---|---|---|---|
| 64 px (384 m) | 676 | +1.99 | −1.45 | 1.34 | 3.36 |
| **128 px (768 m)** | **169** | **+1.99** | **−1.49** | **1.15** | **2.96** |
| 256 px (1,536 m) | 36 | +2.07 | −1.51 | 1.09 | 2.98 |

The offset moves by at most 0.08 m across window sizes.

**Halves of the tile** (128 px windows, 78 per half):

| Half | East (m) | North (m) |
|---|---|---|
| West | +1.78 | −0.77 |
| East | +2.05 | −1.91 |
| South | +2.39 | −1.35 |
| North | +1.64 | −1.61 |

## Is one rigid shift enough?

**Mostly, but not entirely. The residual is small and systematic.**

- **Size.** After removing the offset, the window residuals have an rms of **1.15 m** and a 99th percentile of **2.96 m** (both axes combined). The largest single window is about 3.6 m north, in the west.
- **It varies systematically.** Neighbouring windows' residuals correlate at 0.78–0.87, so this is a smooth field, not noise. The biggest term is the north offset changing from west to east: **−0.265 m per km**. That's about +1.3 m at the west edge and −1.3 m at the east edge. The plane fit's other gradients are −0.106 m/km (east offset changing northward), +0.041 and −0.046 m/km. Expressed as a rotation [Computed], it is 0.08 mrad clockwise, plus about 0.19 mrad of shear. Both are small next to the LVS map's 1 mrad orientation requirement. The residual grids also show structure a plane doesn't capture, such as a 2–3.6 m northward patch in the south-west. Those are printed by the check script, not copied here.
- **What this means for Fidelis.** Per ADR 0003, this local distortion **stays in the error**. It is real map-versus-scene disagreement that a navigator matching against CTX would carry. Only the single offset is removed.

## Uncertainty, and why it isn't the standard error

The windows' standard error is 0.04 m east and 0.08 m north, but it **is not a valid uncertainty**. It assumes independent windows, and the residuals are strongly correlated (0.8+), so the effective number of independent windows is far below 169. The quoted **±0.4 m east and ±0.6 m north** is instead half the spread of the four half-tile medians (east 1.64–2.39, north −1.91 to −0.77, printed by `scripts/check_map_offset.py`; the pipeline's `offset.json` doesn't carry it). It is **a Fidelis heuristic bound, not a 1σ or any stated confidence level**. The halves overlap, so they aren't independent, and the spread mostly reflects the residual's spatial trend. It answers a practical question: how much the "one shift" depends on which part of the tile is used. It covers the method's own error (≤ 0.10 m from injection) and the window-size spread (≤ 0.08 m). It does **not** cover lighting bias (see Caveats), which isn't sized and could be larger.

## Caveats

- **Lighting can masquerade as an offset.** The two images were taken at different times under different suns. Shadows fall away from the sun, so a different sun direction moves the apparent position of every shadowed feature. Registration can't tell that from a true geometric offset. Neither image's sun angle is recorded in the repo (the HiRISE sun angle is listed as "not found" in the research note), so this bias isn't sized.
- **Different terrain models.** Each ortho was projected onto its own DTM, and over the tile those differ by a median of +0.32 m with an sd of 3.11 m (`docs/benchmarks/wide-scene.md`). Off-nadir imaging turns height error into horizontal shift. That is part of what the residual measures, and isn't separated out.
- **Scales under one window aren't resolved.** Each window gives one shift for 768 m (384 m at the smallest size tried). Distortion that varies over less than that averages out here. The LVS requirement's 120 m scale isn't tested.
- **It's the offset between these two files.** It doesn't say how either sits on true Mars. USGS made both as a set of Mars 2020 TRN products (Fergason et al. 2020, cited in the config). Whether that set was co-registered to HiRISE, which would explain such a small offset, wasn't checked in the source. Cheng et al. 2021 describe a final 17.96 m west / 20.25 m north shift of the onboard map to align with HRSC. Whether the USGS release includes it isn't known, and for Fidelis it doesn't matter: the offset that counts is between the map it will match against and the scene it renders.
- **One tile.** Only the 10 km HiRISE tile around the landing site was measured.
