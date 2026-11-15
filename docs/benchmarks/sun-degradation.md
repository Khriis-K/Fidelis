# How navigation degrades with sun difference (issue #22)

The report ADR 0006 keeps: map-frame horizontal error against the angle between the descent sun and the map's sun, and an exceedance curve P(error > x). It is built from saved records only. No new descents are rendered.

## Plan

**Status: written 2026-11-15, before the report code ran on any records.** The results go under it.

### Two series, not one curve

The saved records hold two sets of sun differences, and they differ in more than the sun:

| Series | Records | Sun differences | Texture | Sourced errors (haze, focal, attitude) |
|---|---|---|---|---|
| **Realistic** | `results/certified_bound/batch{0..3}` (#47), 10,000 descents | 8.2°, 21.8°, 28.4°, 30.2°, 39.0° (the five config suns) | Photo | On |
| **Stress** | `results/stress_sun/{anchor,stress1,stress2,stress3}` (#43), 300 descents each, the same seeds | 39.0°, 73.4°, 98.3°, 112.6° | De-shaded | Off |

So the two series are reported side by side and **never joined into one line**. Where they meet at 39.0°, the gap between them is the texture and the sourced errors together, not the sun. The realistic suns also vary in elevation (25°, 40°, 55°), while the stress suns all sit at 25°. Within the realistic series, sun difference and elevation are confounded.

`stress3_photo` (the 112.6° sun on the photo) is listed as a row of its own, as a control.

Sun difference is the 3D angle between the two sun directions (`fidelis.stress.sun_difference_deg`), against the CTX map's sun (267.37°, 42.42°).

### Per group (one sun of one series)

- **Descents, and no-fix count.** No-fix descents have no error. They are left out of the error statistics and counted in the exceedance curve as beyond every x.
- **Horizontal map-frame error:** median and 90th percentile, each with a 95% percentile-bootstrap interval (2,000 resamples of descents, fixed seed). The **maximum** is given without an interval: a bootstrap can never go above the sample maximum, so its interval would be meaningless.
- **Mean shift:** the size and bearing of the mean east-north error, the size with a bootstrap interval. The bearing is also given relative to the sun's bearing, since the stress scan found the shift points away from the sun.
- **Scatter:** each descent's distance from its group's mean shift, median with a bootstrap interval.

### Exceedance curve

Over the realistic series only (all 10,000 descents, all suns pooled as the sampler draws them): P(error > x) for x = 2, 5, 10, 15, 20, 25, 30 and 40 m, with a 95% Wilson upper bound. **Exploratory.** r = 40 m is the official line (LVS's requirement, ADR 0003). A stress series isn't a failure rate, so it gets no exceedance curve (ADR 0003 decision 1).

### Checks

- **The stress rows must reproduce `stress-sun-scan.md`** (its exploratory table: shifts 4.67, 7.14, 8.42, 9.93 m; scatter medians 2.32, 2.92, 3.42, 3.55 m). They come from the same records, so any mismatch is a bug in the new code, not a finding.
- **The realistic series must reproduce `certified-bound.md`'s counts:** 10,000 descents, 1 no-fix, 0 beyond 40 m, worst horizontal error no larger than 22.62 m (that figure is 3D).
- **A constant error.** A test feeds the same vector to every descent. Its scatter and its median's bootstrap interval must come out zero-width, and its shift must equal the vector.

### Prediction from the sources

The per-sun 3D medians of the 10,000 are already published (`certified-bound.md`, section 3). What's new here is the horizontal error, the intervals, the shift-versus-scatter split for the realistic suns, and the exceedance curve.

- **Within the realistic series, error follows elevation, not sun difference.** The published medians already order the suns that way: the two 25° suns are highest (6.84 m and 6.35 m, at 28.4° and 39.0°), then the 40° sun (4.94 m, at 8.2°), then the two 55° suns (4.18 m and 3.77 m, at 21.8° and 30.2°). Horizontal medians should keep that order. **So the realistic series won't show a clean rise with sun difference.**
- **The realistic shift points away from the sun, and is largest under the low suns [Inference].** The stress scan found the shift 187–218° from the sun's bearing. The mechanism it proposed (shading moves edges) should hold here too, so expect 150–250°. Under the 8.2° sun, the shift should be mostly the CTX offset residual found in `ctx-map-scan.md` (about 1.4 m east), so its bearing won't follow the sun.
- **Exceedance:** P(error > 40 m) is 1 in 10,000 (the one no-fix). P(error > 20 m) should be a handful in 10,000, since the 99th percentile was 15.44 m (3D). P(error > 10 m) should be about 5–12%: the combined scan had 13 of 300 (4.3%) above 12 m in 3D, and more sit above 10 m **[Inference]**. The sources don't cover the shape of the tail beyond that.

**A surprise would be** a realistic series ordered by sun difference rather than elevation, or a stress row that doesn't reproduce the stress-scan table.

## Results

**Run on 2026-11-15** on the laptop, from the saved records, with `snakemake sun_degradation` (about 25 s; output in `results/reports/sun_degradation.md`, gitignored). 2,000 bootstrap resamples, seed 0. All errors are horizontal, map frame, metres.

**One bug found by the plan's own check, and fixed before these numbers:** the first run put 9,999 descents and 0 beyond 40 m in the exceedance table, against the bound's 10,000 and 1. A variable from the stress loop had overwritten the realistic no-fix count. A test now covers it (`test_report_counts_the_realistic_no_fixes_in_the_exceedance_curve`).

**One correction to the plan:** the plan called `certified-bound.md`'s per-sun medians and its 22.62 m maximum 3D. They match the horizontal values here exactly, so they were horizontal.

### Realistic series (#47: photo texture, sourced errors on)

| Sun (az, el) | Sun difference | Descents | No fix | Median (95% CI) | 90th pct (95% CI) | Max | Mean shift (95% CI) | Shift bearing | Scatter median (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| 257°, 40° | 8.2° | 2,024 | 0 | 4.94 (4.82–5.09) | 9.70 (9.36–10.04) | 18.76 | 3.64 (3.49–3.79) | 79° (182° from sun) | 3.29 (3.15–3.46) |
| 240°, 55° | 21.8° | 2,032 | 0 | 4.18 (4.04–4.33) | 9.21 (8.90–9.45) | 22.62 | 2.60 (2.45–2.75) | 64° (184° from sun) | 3.27 (3.12–3.44) |
| 240°, 25° | 28.4° | 1,998 | 0 | 6.84 (6.72–7.02) | 11.40 (11.02–11.87) | 20.80 | 5.99 (5.82–6.14) | 57° (177° from sun) | 3.18 (3.02–3.36) |
| 310°, 55° | 30.2° | 2,005 | 1 | 3.77 (3.63–3.95) | 8.97 (8.52–9.28) | 19.25 | 2.29 (2.13–2.44) | 129° (179° from sun) | 3.13 (2.95–3.27) |
| 310°, 25° | 39.0° | 1,941 | 0 | 6.35 (6.21–6.54) | 11.30 (10.89–11.59) | 19.56 | 5.56 (5.40–5.72) | 144° (194° from sun) | 3.39 (3.23–3.56) |
| Pooled | — | 10,000 | 1 | 5.33 (5.26–5.40) | 10.16 (10.00–10.35) | 22.62 | 3.20 (3.13–3.27) | 93° | 4.36 (4.29–4.45) |

### Stress series (#43: de-shaded texture, sourced errors off; a diagnostic, not a rate)

| Run | Sun (az, el) | Sun difference | Descents | No fix | Median (95% CI) | 90th pct (95% CI) | Max | Mean shift (95% CI) | Shift bearing | Scatter median (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| Anchor | 310°, 25° | 39.0° | 300 | 0 | 4.84 (4.55–5.15) | 7.76 (7.27–8.10) | 9.67 | 4.67 (4.43–4.91) | 168° (218° from sun) | 2.32 (2.11–2.47) |
| Stress 1 | 357.4°, 25° | 73.4° | 300 | 0 | 7.42 (7.19–7.82) | 11.04 (10.52–11.75) | 16.26 | 7.14 (6.83–7.45) | 210° (213° from sun) | 2.92 (2.71–3.14) |
| Stress 2 | 37.4°, 25° | 98.3° | 300 | 0 | 8.58 (8.26–8.99) | 12.77 (12.31–13.27) | 18.71 | 8.42 (8.07–8.78) | 241° (204° from sun) | 3.42 (3.15–3.61) |
| Stress 3 | 87.4°, 25° | 112.6° | 300 | 0 | 10.40 (9.76–10.72) | 14.03 (13.38–15.11) | 20.33 | 9.93 (9.62–10.26) | 274° (186° from sun) | 3.55 (3.28–3.78) |
| Stress 3, photo | 87.4°, 25° | 112.6° | 300 | 0 | 5.67 (5.36–5.87) | 7.72 (7.38–7.98) | 11.46 | 5.38 (5.19–5.55) | 276° (188° from sun) | 2.09 (1.96–2.23) |

### Exceedance, realistic series pooled (exploratory; r = 40 m is the official line)

| x | Descents beyond x | P(error > x) | 95% upper bound |
|---|---|---|---|
| 2 m | 9,100 of 10,000 | 91.00% | 91.55% |
| 5 m | 5,397 of 10,000 | 53.97% | 54.95% |
| 10 m | 1,063 of 10,000 | 10.63% | 11.25% |
| 15 m | 127 of 10,000 | 1.27% | 1.51% |
| 20 m | 3 of 10,000 | 0.030% | 0.088% |
| 25 m | 1 of 10,000 | 0.010% | 0.057% |
| 30 m | 1 of 10,000 | 0.010% | 0.057% |
| 40 m | 1 of 10,000 | 0.010% | 0.057% |

From 25 m up, the one descent beyond x is the no-fix. No descent with a fix was more than 22.62 m off.

## Against the prediction

| Prediction (before the run) | Result | Held? |
|---|---|---|
| Realistic error follows elevation, not sun difference | 25° suns 6.84 and 6.35 m; 40° sun 4.94 m; 55° suns 4.18 and 3.77 m. Intervals don't overlap between elevations | Yes |
| Realistic shift points 150–250° from the sun, largest under low suns | 177–194° at every sun. 5.6–6.0 m under the 25° suns, 2.3–2.6 m under the 55° suns | Yes |
| Under the 8.2° sun, the shift is mostly the CTX offset residual and doesn't follow the sun | 3.64 m at 182° from the sun. All five suns are in the west, so "away from the sun" and the east residual point the same way here. They can't be separated | Can't tell |
| P(> 20 m) a handful in 10,000 | 3 | Yes |
| P(> 10 m) about 5–12% **[Inference]** | 10.6% | Yes |
| Check: stress rows reproduce `stress-sun-scan.md` | Shifts 4.67 / 7.14 / 8.42 / 9.93 m, scatter 2.32 / 2.92 / 3.42 / 3.55 m, bearings 168–274°: all match | Yes |
| Check: 10,000 descents, 1 no-fix, max ≤ 22.62 m | 10,000, 1, 22.62 m (after the bug fix) | Yes |

## What this means

- **In the realistic range, the descent sun's elevation matters more than its angle from the map's sun.** The sun 8.2° from the map's gives more error than either 55° sun, which sit 22–30° away. The low suns roughly double the shift. So "error vs sun difference" isn't a clean axis below 40°. What drives the realistic error is how much shading the descent sun casts.
- **The error is a shift away from the descent sun, at every sun, in both series.** The shift's bearing follows the sun's azimuth almost exactly (177–194° from it in the realistic series, 186–218° in the stress one). The scatter about the shift hardly changes: 3.1–3.4 m across the realistic suns, 2.3–3.6 m along the stress doses. That's consistent with the stress scan's proposed mechanism (shading moves the edges keypoints sit on), now on 10,000 descents. It still hasn't been checked feature by feature **[Inference]**.
- **Only the stress series shows a rise with sun difference**, from a 4.67 m shift at 39° to 9.93 m at 112.6°, with no step at JPL's 100°. It holds the elevation at 25°, which is why the rise is visible there.
- **The tail is thin.** About 1 descent in 10 is more than 10 m off, about 1 in 80 more than 15 m, and 3 in 10,000 more than 20 m. Nothing with a fix got past 22.62 m, about half of r.
- **Where the series meet (39.0°, the same sun), they disagree:** median 6.35 m realistic, 4.84 m stress. That gap is the texture and the sourced errors together. This report can't split it, as the plan said.

## Caveats

- **Two series, never one curve:** they differ in texture, sourced errors and sample size (10,000 against 300 per sun), not only the sun.
- **Elevation and sun difference are confounded** within the realistic suns: five suns, three elevations. The elevation reading is from five points.
- **The 8.2° sun's shift can't be split** between the sun and the CTX offset residual, because every Fidelis sun sits in the west.
- **The exceedance curve is exploratory** and holds only under the sourced ranges, with the exposure range that runs bright (`exposure-brightness.md`).
- **Pretrained matcher only.** Fine-tuning was dropped (ADR 0006).
