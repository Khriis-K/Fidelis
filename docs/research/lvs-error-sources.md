# LVS error sources and failure modes

**Question:** when NASA/JPL validated Perseverance's Lander Vision System (LVS), which error sources did they model, how big were they, and what made LVS fail or degrade? The aim is to add realistic missing errors to Fidelis, sized only from real sources. The failure scan (`docs/benchmarks/failure-scan.md`) found 0/300 failures at r = 12 m, with errors of 1.5–4.7 m. Investigated 2026-11-11.

**Status:** research only. No code or config changed. Every number below is quoted from a source, with its page, or marked **[Computed]**, **[Inference]** (our reasoning), or **[Garbled]** (the PDF's text extraction scrambled a table or unit, so the reading is ours). Facts already covered in `edl-dispersions-for-scenario-sampler.md` (exposure, noise, blur, tilt, true-position dispersion, flown altitudes) aren't repeated here unless they bear on failures.

## Summary (plain English)

Today Fidelis gives its navigator a perfect world. The map and the descent image come from the **same picture**, so they agree on every rock and every shadow. The camera model is exact, and the attitude is the true one. That's why nothing fails. The real LVS was tested against a map made from *different* orbital photos, a hazy atmosphere, and an imperfectly calibrated camera.

Ranked by how likely each is to produce realistic failures (the ranking is **[Inference]** from the sources below):

1. **Map and descent image that don't look alike.** This is the only documented cause of LVS field-test misses. All 6–7 runs outside 40 m had a sun-angle difference between map and image above 100°. Fidelis has *no* mismatch today, because its map is a block average of the same orthoimage it renders from.
2. **Dust haze lowering contrast.** LVS stopped matching below about 4 DN of scene contrast. Pre-launch tests varied dust optical depth τ from 0.5 to 1. Fidelis renders no haze at all.
3. **Map distortion (local warping of the orbital map).** Pre-landing simulations found this the *largest* contributor to LVS position error. The requirement allowed up to 18 m (99%), which is bigger than Fidelis' 12 m failure line on its own. Not modelled.
4. **Camera calibration (focal length).** A ±1% focal-length error gave up to 16.9 m (99%) of error in the 2017 error budget, the largest single term at the time. It also exceeds 12 m alone. Not modelled. A post-calibration value wasn't found.
5. **Attitude knowledge (IMU) error.** Small: under 0.5° per axis at LVS start, and LVS tolerated 5°. Fidelis already measured little effect up to 3° (`docs/benchmarks/navigator.md`). Low priority for failures.
6. **Altitude and timing errors.** Small and well tolerated (altitude ±350 m tested fine; time sync worth ~1.8 m). Lowest priority.

**Push back: what counts as a failure.** Items 3 and 4 mostly make the fix *inaccurate*, not *lost*. At r = 12 m they would turn into "failures" without any matching going wrong. That's a valid accuracy failure, but it's a different failure mode from a lost match, and the results should say which one they count. A *global* map shift is worse still. The LVS map was allowed a 150–300 m absolute offset, which "essentially cancels out" because the hazard map is registered to the same map (Johnson et al. 2017 p.10). If Fidelis adds a global map tie error and measures error against the true position, every descent would fail. Measure against the map frame, as LVS did, or leave global tie error out.

## 1. Attitude knowledge and how LVS uses attitude

**Sizes.**
- **At LVS start (pre-flight design):** the spacecraft hands LVS an attitude whose "error is less than 0.5° per axis because it is based on DIMU propagation from an accurate attitude estimate prior to entry" (Johnson et al. 2017 p.8). The Monte Carlo drew it Gaussian, "less [than] 0.5° per axis" (p.12). **Sigma level not stated**: it reads as a bound, not a 1σ.
- The MRL design assumes "the lander surface relative attitude error is small (<0.5° in all axes)" so a small-angle solver works (Johnson et al. 2023, fine-phase RANSAC section).
- **Tolerance:** off-nadir and about-nadir attitude errors were swept from 0 to 5° in 0.5° steps. "LVS was able to take out all of these attitude errors" (Johnson et al. 2017 p.13, Table 3: operational limit 5°, requirement < 0.5°).
- **Attitude initialization before entry:** 0.15° pre-launch, cut to 0.075° from in-flight calibration. The reconstructed flight value was 33 millidegrees (Mohan et al. 2021 pp.13, 15, 17). Sigma level not stated; the error budget treats these as inputs to 99%-tile terms.
- **Flight:** LVS attitude error against the reconstruction was under 0.05° on X and Z, and 0.1° on Y, possibly an unmeasured LCAM–DIMU misalignment (Johnson et al. 2022 p.15).

**LCAM-to-IMU misalignment.** The camera is on the rover, the IMU (DIMU) on the descent stage, joined by a cup/cone interface that can slip. "Shock events, parachute dynamics and thermal drift can introduce relatively large errors (0.75°/axis)" in alignment knowledge. Simulated as Gaussian within ±0.75° per axis, effect on position "no more than 3.0m" (Johnson et al. 2017 p.11–12). The later simulator split it into a static misalignment per run plus a dynamic part scaled by accelerations and angular rates (Aaron et al. 2022 p.3). Flight showed dynamic attitude error under 0.025° (Mohan et al. 2021 p.16).

**Does LVS solve for attitude itself?** Yes, after a start from the IMU.
- The spacecraft seeds LVS with position, attitude, velocity and altitude (Johnson et al. 2017 p.2).
- **Coarse phase:** estimates *horizontal position only*, by batch least squares, with attitude taken from the DIMU-propagated state (Johnson et al. 2023, coarse batch estimator section; Johnson et al. 2022 p.3). The image is warped onto the map plane with a homography built from the current pose and the radar altitude (Johnson et al. 2023, step A).
- **Fine phase:** an Extended Kalman Filter "estimates LVS position, velocity, attitude and IMU biases" (Johnson et al. 2017 p.3). Full state estimation was a deliberate choice: "This allowed the LVS to compensate for errors in the initial attitude and velocity as well as misalignments" (Johnson et al. 2022 p.2). Each RANSAC hypothesis solves position and a small attitude correction from three matches (Johnson et al. 2023).

**For Fidelis [Inference].** The Fidelis navigator solves full pose with PnP, so attitude only enters through the north-up warp before matching. Sourced attitude errors (≤ 0.5°, misalignment ≤ 0.75°) sit well below the 3° where Fidelis saw no effect. Adding them is cheap realism but is unlikely to create failures.

## 2. Orbital map errors and map–image appearance differences

**Map requirements** (Cheng et al. 2021 Table 1, p.2):

| Requirement | Value |
|---|---|
| Appearance map pixel size | 6 m/pixel, from imagery of ≤ 6 m/pixel |
| Elevation map pixel size | 12 m/pixel, from DEMs of ≤ 20 m/pixel |
| Elevation–appearance co-registration | within 6 m (99%) across the 30 × 30 km map |
| Absolute horizontal tie error | < 300 m |
| Orientation error about the normal | < 1 mrad |
| Local horizontal distortion | 9 m (99%) under 120 m separation; 18 m (99%) from 120 m up to 12 km |
| Absolute elevation error | < 20 m (3σ) vs the MOLA aeroid |
| Relative pixel-to-pixel elevation | < 20 m (3σ) |
| Radiometry | Not enhanced, colour-balanced or smoothed |

The 2017 simulation used slightly tighter numbers: position error < 150 m horizontal and 20 m vertical, orientation < 1 mrad, local distortion < 6 m under 120 m and 12 m beyond, pixel-to-pixel elevation < 2 m (1σ) (Johnson et al. 2017 p.10). These differ from Cheng's Table 1, probably because the requirement was revised; the sources don't say why.

**Measured on the Jezero map** (Cheng et al. 2021 pp.19–20):
- Two independent JPL maps (A, B): dx/dy standard deviations 1.978 / 2.852 m; mean dy offset ~14 m; 99th-percentile distortion 5.376, 5.715, 5.583 m at 120, 1,200, 12,000 m.
- USGS vs JPL map B: 99th percentile 5.94, 6.17, 6.3 m at the same scales. USGS map vs HiRISE: 4.27, 4.29, 7.74 m.
- Final shift to align with HRSC: 17.96 m west, 20.25 m north. Orientation vs HRSC 0.8 mrad.
- Relative elevation: 1σ 4.35 m, 99th percentile 13.05 m between maps A and B.
- **Flight check:** an LCAM image projected onto the LVS map gave shift standard deviations "around 3m", close to pre-landing estimates, and about 1.5 m in the image centre (Johnson et al. 2022 p.17). Mohan et al. 2021 p.16 say the LCAM-to-map displacements ranged 0–9 m against the 18 m pre-launch value.

**What the errors cost.** Map distortion and rotation gave up to 3.4 m (99%) in the 2017 budget, against a 12 m allocation (Johnson et al. 2017 Table 2). By 2021: "Pre-landing simulations showed that local distortions in the map were the largest contributor to errors in the LVS position estimate" (Johnson et al. 2022 p.17). A global shift "will just shift the LVS position by the same amount" and cancels at system level (Johnson et al. 2017 p.11).

**How JPL simulated it.** The truth map was translated and rotated in 3D, then given "a random field of horizontal distortions that are correlated locally and pixel to pixel Gaussian elevation noise" (Johnson et al. 2017 p.9–10). The later simulator applied a global tie error, a rotation about nadir and local distortions, with a different distorted truth map per Monte Carlo run (Mohan et al. 2021 p.8–9). The correlation length of the distortion field: **not found**.

**Appearance differences.**
- **Sun angle between map and descent image:** the Jezero map images had to be within ±15° of the landing sun elevation and ±35° of its azimuth (Cheng et al. 2021 p.8). For Mars 2020 the LCAM-to-map sun difference "is less than 20°" (Johnson et al. 2020 p.6).
- **Map image dates and sun:** the CTX images were taken 2006–2017 (Cheng et al. 2021 Tables 3–4) with sun elevations 25.6°–51.9° and azimuths 222.9°–295.1° (Tables 5–6). CTX images at about 3 pm, and the team picked images from the same season (northern spring) as landing (Johnson et al. 2020 p.5).
- **Dust in map images:** required atmospheric opacity < 1.0; listed τ values 0.45–1.148, all but one under 1 (Cheng et al. 2021 p.8, Tables 5–6).
- **Two-map test:** JPL rendered descent images from one CTX-based map and matched them to another built from a different stereo pair. LVS worked everywhere; the worst case, a bland area, had 2 wrong coarse matches out of 5 but >100 fine matches (Johnson et al. 2017 p.13–14).
- **Simulation vs reality gap:** flight correlation peaks may have been narrower than simulated, probably because simulated images were rendered from the 6 m map itself (Johnson et al. 2022 p.16). That's the same limitation Fidelis has, one step worse: Fidelis renders and matches from the same orthoimage.

**For Fidelis [Inference].** The HiRISE ortho (rendered) vs CTX ortho (matched) split is exactly the JPL two-map test. The repo already has the CTX mosaic in `config/jezero.yaml` (`ctx_ortho`, "the LVS map itself, per USGS"). Matching rendered HiRISE views against the real CTX map would add real map distortion, real date and lighting differences, and real CTX noise in one step, with no invented numbers. It's the single most realistic change available.

## 3. Sun geometry and lighting sensitivity

- **Design range:** elevation 25°–55°, azimuth 240°–310° clockwise from north (Johnson et al. 2017 p.3; Johnson et al. 2020 p.1).
- **Expected at landing time:** elevation 39.6°–42.2°, azimuth 255.8°–257.2° (Cheng et al. 2021 p.8). Our earlier computation gave 39.8°, 257.3° for the flight (`edl-dispersions-for-scenario-sampler.md`).
- **Landing-day simulation sun:** Mohan et al. 2021 p.16 give "0.71148347 and 4.477205864 deg" for azimuth and elevation. **[Garbled]**: read as radians these are 40.8° and 256.5°, which matches Cheng's window if the order is elevation then azimuth. Treat as unconfirmed.
- **Sensitivity, simulation:** field testing and MSL data showed LVS "not sensitive to illumination differences within its operational envelope" (Johnson et al. 2017 p.13).
- **Sensitivity, field test** (Johnson et al. 2020 pp.6–7, 11):
  - Expanding the map-to-image sun difference from 20° to 100° (plus entropy 4→3, altitude to 5000 m, rates to 100°/s): all 393 runs still met 40 m, though some had low inlier counts.
  - Beyond that: "only 7 of the additional 263 runs" exceeded 40 m, and "All 6 of these" came from one late-afternoon Kelso Dunes run with sun differences above 100°. The 7-vs-6 discrepancy is in the source.
  - Coarse retries (a fault response when too few landmarks match): 20 runs, none inside the operational envelope. Causes: extreme illumination differences 7, clouds or cloud shadows 5, large relief plus illumination difference 3, saturation 3, large relief 1, map edge 1 (Table 5).
- **Shadows:** the JPL camera simulator ray-traced terrain shadows plus capsule and parachute shadows (Aaron et al. 2022 p.4). Fidelis' mesh renderer casts terrain shadows (`src/fidelis/render/mesh_render.py`).

**For Fidelis [Inference].** Fidelis already spans the 25–55° design box with five suns. Sources say low sun alone didn't break LVS. The *difference* between map lighting and image lighting did. So varying the render sun while the map keeps one fixed lighting is what matters, and today both carry the same baked HiRISE shading.

## 4. Camera calibration, timing and altitude

**Focal length.** The calibration estimate of LCAM focal length "can be as large as 1%". With ±1% added to the MRL camera parameters, "errors can be as large as 17m horizontally" when the camera looks off nadir: 16.9 m (99%), allocation 20 m, the largest term in the 2017 budget (Johnson et al. 2017 p.11, Table 2). The paper said more accurate calibration would be investigated. Monte Carlo drew it Gaussian within ±1% (p.12). LCAM was later calibrated with a dot-grid target measured by 3D metrology and a CAHVORE model (Johnson et al. 2023, LCAM section). **The post-calibration focal-length uncertainty: not found.**

**Lens distortion.** LCAM has 18% radial distortion at the image edge, corrected with a lookup table (Johnson et al. 2017 p.8, p.11). Fidelis renders a distortion-free pinhole camera. Calibration *residual* after correction: **not found**.

**Other LCAM errors** (detector noise, distortion, falloff, opposition effect) together gave about 1.1 m (Johnson et al. 2017 p.11).

**Timing.**
- The LVS-to-navigation-filter time sync was bounded by one 64 Hz sample (15.625 ms) times 115 m/s, giving 1.79 m pre-launch; 0.31 m post-landing (Mohan et al. 2021 pp.11, 13, 17).
- The camera simulator modelled command and image-transfer delays as normal distributions "with bias and standard deviation to simulat[e] timing uncertainty" (Aaron et al. 2022 p.3). **Their sizes: not found.** Camera-to-IMU latency is listed as a modelled term (Mohan et al. 2021 p.11) without a value.
- LCAM image latency ~100 ms (Johnson et al. 2017 Table 1).

**Altitude.** Radar altitude error < 100 m at LVS start. Swept ±500 m in 50 m steps; LVS worked except at +400 m (Johnson et al. 2017 p.12–13, Table 3: operational limit 350 m). Vertical position errors in the 2017 Monte Carlo were under 45 m (p.13). The coarse warp uses the radar altitude (Johnson et al. 2023, step A).

**Other initialization errors.** Velocity < 2 m/s per axis; horizontal position up to 3.2 km, tested to 5 km with no effect (Johnson et al. 2017 pp.8, 13).

## 5. Requirement and validation results

**Requirement.** LVS "generate a solution with less than 40m accuracy within 10s of initialization, 99% of the time" (Mohan et al. 2021 p.4). Slides give "40m 99% @ 2km" (Montgomery et al. 2019). So it's **99%, not 3σ**. There was also a 54 m-at-6-s requirement for late starts (Johnson et al. 2017 p.2), and ~200 m after coarse (Johnson et al. 2022 p.3). The touchdown requirement was 60 m radial, 99% (Mohan et al. 2021 p.2).

**Error budget, 2016** (Johnson et al. 2017 Table 2; each source alone at its maximum, 99%-tile):

| Source | Best estimate | Allocation |
|---|---|---|
| Algorithm | 0.6 m | 1 m |
| DIMU noise | 0.4 m | 1 m |
| LCAM image effects | 1.1 m | 5 m |
| LCAM focal length | 16.9 m | 20 m |
| LCAM–DIMU misalignment | 3.0 m | 15 m |
| Map distortion and rotation | 3.4 m | 12 m |
| **RSS** | **17.3 m** | **28.2 m** |

**Monte Carlo.**
- 2016, NE Syrtis, 256 POST trajectories: horizontal error under 14.8 m after 6 images; vertical under 45 m (Johnson et al. 2017 pp.12–13).
- Pre-launch, Jezero: 30.28 m (99%-tile) at 6 s after initialization, over the full envelope (Mohan et al. 2021 p.12). The final pre-flight Monte Carlo used 256 trajectories (Johnson et al. 2022 p.10, Fig. 8; results in a figure only).
- Post-landing rerun with landing-day conditions: 5.96 m (99%-tile) at 2200 m, 256 runs (Mohan et al. 2021 p.16).
- **How many Monte Carlo runs failed outright: not found.**

**Field test** (May 2019, helicopter, Mojave National Preserve and Death Valley; 17 flights, 656 runs; Johnson et al. 2020):
- 83 runs inside the flight envelope, 393 in an expanded envelope, 656 total.
- 99%-tile radial error **[Garbled]**, read from a scrambled table: at 10 s, 14.3 / 24.3 / 55.4 m (flight / expanded / all); at 6 s, 16.0 / 30.5 / 44.5 m; end of coarse 158.0 / 135.3 / 169.3 m (Table 4).
- All flight-envelope and expanded-envelope runs met 40 m. 6–7 of the rest didn't (section 3).
- The 2016 field test gave 16.0 m against the 17.3 m budget (Johnson et al. 2017 Table 2).

**Flight** (18 Feb 2021): 15/15 coarse landmarks matched; 137 m position shift found; first fine image 92 inliers from 150; all 42 fine images had ≥ 80 inliers; touchdown 5 m from the STS target (Johnson et al. 2022 pp.13–15). Post-landing budget bound 8.07 m (99%) (Mohan et al. 2021 p.19). Skies were clear with low dust (Johnson et al. 2022 p.13); dust τ was about 0.47 (Mohan et al. 2021 p.16).

## 6. Documented failure modes

From field test, sensitivity studies and fault-protection design:

| Failure mode | What happened | Source |
|---|---|---|
| Map–image sun difference > 100° | The only runs outside 40 m; most common coarse-retry cause | Johnson et al. 2020 pp.7, 11 |
| Low contrast (dust, bland terrain) | Matching worked down to entropy 2 (~4 DN contrast), not 1 (~2 DN). Bland terrain listed as a cause of low coarse inliers | Johnson et al. 2017 p.15; Johnson et al. 2023 fault section |
| Large terrain relief and slope | Coarse retries; relief 281 m, slope 23° in one example. Envelope 150 m relief, 15° slope | Johnson et al. 2020 pp.6, 11 |
| Clouds and cloud shadows | Coarse retries (Earth only) | Johnson et al. 2020 Table 5 |
| Saturation, bad exposure | Coarse retries; matching failed below 1/8 brightness (16 DN mean) | Johnson et al. 2020 Table 5; Johnson et al. 2017 p.14 |
| Altitude error +400 m | Only failing value in the ±500 m sweep | Johnson et al. 2017 p.13 |
| Initial position error > ~5 km | Not expected to work beyond the ±6 km coarse map | Johnson et al. 2017 p.13 |
| Heat shield ≤ 2 m away | Covers the image; rejected at ≥ 3 m | Johnson et al. 2017 p.15 |
| Map edge | One coarse retry | Johnson et al. 2020 Table 5 |
| Poor initial navigation state | Listed cause of low coarse inliers | Johnson et al. 2023 fault section |

Not a failure mode: attitude error up to 5° and exposure off by 4× both worked (Johnson et al. 2017 pp.13–14).

## Error sources compared with Fidelis

"Modelled today" checked against `config/jezero.yaml` (`scenario:`), `src/fidelis/simulation.py`, `src/fidelis/nav/navigator.py`, `src/fidelis/nav/orbital_map.py` and `src/fidelis/render/degrade.py` on 2026-11-11. The "expected effect" column is **[Inference]**, not measured.

| Error source | Real size (sourced) | Modelled in Fidelis today? | Expected effect on failures |
|---|---|---|---|
| Map–image appearance (different source imagery, date, lighting) | Sun difference < 20° for Mars 2020; field misses at > 100° | **No.** Map is a block average of the same orthoimage rendered | Highest. The documented cause of LVS misses |
| Dust haze (contrast loss) | τ 0.5–1 pre-launch; ~0.47 in flight; failure below ~4 DN contrast | **No.** No atmosphere in the renderer | High: lowers contrast directly |
| Map local distortion | ≤ 9 m / 18 m (99%) required; ~3 m sd measured in flight | **No** | Medium: shifts fixes; can cross 12 m alone at the requirement value |
| Map global tie error | < 300 m required (150 m in 2017 sim); 18/20 m shift applied | **No** | None if error is measured in the map frame; *every* descent fails if measured vs truth |
| Map orientation | < 1 mrad; 0.8 mrad measured vs HRSC | **No** | Small: Mohan et al. convert it to 8.35 m at the corner of a 12 km box and 1.77 m for the flown target |
| Map elevation error | < 20 m (3σ); 4.35 m 1σ between maps | **No.** Map DEM is the render DEM | Small to medium; not separately quantified in sources |
| Focal-length calibration | ±1% → up to 16.9 m (99%) in 2016 | **No.** Intrinsics exact | Medium: bias that can exceed 12 m off nadir; post-calibration size not found |
| Lens distortion residual | 18% distortion, LUT-corrected; residual not found | **No.** Pinhole | Unknown |
| Attitude knowledge at LVS start | < 0.5° per axis (bound, σ unstated); tolerated to 5° | **No.** True attitude | Low: Fidelis saw no effect to 3° |
| LCAM–IMU misalignment | ±0.75° per axis → ≤ 3.0 m | **No** | Low |
| Altitude (radar) | < 100 m; tolerated to 350 m | Not used: the navigator takes no altitude prior | None unless a prior is added |
| Timing / sync | 15.6 ms sample → 1.79 m; camera delay sizes not found | **No** | Low |
| Sun elevation and azimuth | 25–55°, 240–310° | **Yes**: five suns | Already in |
| Detector noise, exposure, blur | See `edl-dispersions-for-scenario-sampler.md` | **Yes** | Already in |
| Terrain relief and slope beyond envelope | > 150 m relief, > 15° slope caused retries | Real Jezero terrain | Already in (whatever Jezero has) |

## Not found

- The **sigma level** of the 0.5° per-axis attitude error and of the 0.15° / 0.075° attitude initialization. Sources say "less than" and use them in 99%-tile budgets.
- **Post-calibration LCAM focal-length uncertainty** and **lens-distortion residual** for the flight camera.
- **Sizes of the camera timing delays** (bias and standard deviation) in the LVSS LCAM model, and of camera-to-IMU latency.
- The **correlation length** (or spectrum) of the local map-distortion field JPL simulated.
- The **number or fraction of failed runs** in the pre-flight Jezero Monte Carlo (results are only in a figure, Johnson et al. 2022 Fig. 8).
- An **error budget breakdown** for the final pre-launch Jezero LVS figure (30.28 m). Mohan et al. give only the total; the per-term breakdown is in their Fig. 8, which didn't extract.
- The **sun angle of the HiRISE orthoimage** Fidelis renders from (needed to size Fidelis' own map-vs-render lighting difference).
- Any LVS test of **season** (frost, albedo change) apart from choosing same-season CTX images.

## Sources

1. Johnson, A., Aaron, S., Chang, J., Cheng, Y., Montgomery, J., Mohan, S., Schroeder, S., Tweddle, B., Trawny, N., Zheng, J. (2017). The Lander Vision System for Mars 2020 Entry Descent and Landing. AAS Guidance and Control Conference (AAS 17-038). Preprint read: https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/46186 ; NTRS https://ntrs.nasa.gov/citations/20190025658
2. Johnson, A., Villaume, N., Umsted, C., Kourchians, A., Sternberg, D., Trawny, N., Cheng, Y., Geipel, E., Montgomery, J. (2020). The Mars 2020 Lander Vision System Field Test. AAS 20-105. Preprint read: https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/52354 ; NTRS https://ntrs.nasa.gov/citations/20220000862
3. Johnson, A. E., et al. (2022). Mars 2020 Lander Vision System Flight Performance. AIAA SciTech 2022. https://doi.org/10.2514/6.2022-1214 ; preprint read: https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/56099
4. Johnson, A. E., Cheng, Y., Trawny, N., Montgomery, J. F., Schroeder, S., Chang, J., Clouse, D., Aaron, S., Mohan, S. (2023). A Map Relative Localization System for Planetary Landing. JPL preprint read: https://doi.org/10.48577/jpl.TWOU9P ; journal version (not read) https://doi.org/10.2514/1.G006780 . The preprint has no stable page numbers in extraction; cited by section.
5. Mohan, S., Johnson, A. E., Brugarolas, P., Cheng, Y., Otero, R., Trawny, N., Aaron, S., Serricchio, F., Montgomery, J., Casoliva, J., Chen, A. (2021). Assessment of M2020 Terrain Relative Landing Accuracy: Flight Performance vs. Predicts. AAS 21-541. Preprint read: https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/55252 ; NTRS https://ntrs.nasa.gov/citations/20230007014
6. Aaron, S., Cheng, Y., Trawny, N., Mohan, S., Montgomery, J., Ansari, H., Smith, K., Johnson, A., Goguen, J., Zheng, J. (2022). Camera Simulation for the Perseverance Rover's Lander Vision System. AIAA SciTech 2022. https://doi.org/10.2514/6.2022-0746 ; copy read: https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/56110
7. Cheng, Y., Ansar, A., Johnson, A. (2021). Making an Onboard Reference Map From MRO/CTX Imagery for Mars 2020 Lander Vision System. *Earth and Space Science* 8, e2020EA001560. https://doi.org/10.1029/2020EA001560 . Read from the copy in this folder.
8. Montgomery, J., et al. (2019). The Mars 2020 Lander Vision System: Architecture and V&V Results. IPPW 2019 slides. https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/51096
