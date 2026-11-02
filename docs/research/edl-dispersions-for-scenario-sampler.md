# EDL dispersions for the scenario sampler

**Question:** what do primary sources say about the per-descent random inputs that issue #6's scenario sampler draws? Those inputs are true horizontal position, altitude at the LVS handoff image, camera attitude (tilt off nadir, heading), image noise, motion blur, exposure, and sun angle. For each: what was the design envelope, what actually flew on 18 February 2021, and how sure are we? Investigated 2026-11-01.

**Status:** research only. Nothing in `config/jezero.yaml` or `src/fidelis/scenario.py` has been changed. Every number below is either quoted from a source (page given where I saw it), **[Computed]** from sourced numbers with the arithmetic shown, **[Read from plot]** (an approximate reading of a published figure), or **[Inference]** (our reasoning). Anything we couldn't check is marked **not verified**.

## Findings

| Topic | Number(s) | Source | Confidence |
|---|---|---|---|
| **1. Handoff altitude** | The rover chose its landing target from the last LVS fix before back-shell separation (BSS). BSS fired at **2276 m** above ground in flight (2275.6 m in the post-flight reconstruction). The fix STS used came from the 16th fine image, at "approximately 2200m". Design BSS altitude was **2300 m**. The LVS requirement was to keep working down to 2000 m or BSS. | Brugarolas et al. 2021 p.9; Way et al. 2022 Table 7; Johnson et al. 2022 p.13; Way et al. 2020 p.9; Johnson et al. 2017 p.3 | **High** for the flown values. **Not verified:** the spread of the BSS altitude across pre-flight Monte Carlo runs. |
| | LVS started at **4200 m** (design and flight) and declared a valid fix at **3.7 km** in flight. | Johnson et al. 2022 p.13; Way et al. 2022 Fig. 1, Table 7 | High |
| **2. Horizontal position** | *Knowledge error* before LVS: up to **3.2 km** (design bound, "99%" in JPL slides). In flight it was only **137 m**. | Johnson et al. 2017 p.1, p.8; Montgomery et al. 2019 slides; Johnson et al. 2022 p.13 | High |
| | *True-position dispersion*: the pre-flight 99% touchdown ellipse at Jezero was **7.1 × 6.5 km**, from 8000 Monte Carlo runs. Post-flight reports give 7.5 × 5.2 km and 7.4 × 5.1 km, both in one paper. In flight the vehicle reached the zero-divert point **1.13 km** from the pre-flight target, at 1.03σ. At BSS it was **0.78 km** from the landed rover (the tile centre) **[Computed]**. | Way et al. 2020 p.5; McGrew et al. p.1; Way et al. 2022 p.3, Tables 1 and 7 | High for the numbers. The 7.5/7.4 discrepancy is in the source itself. |
| **3. Attitude on the parachute** | LVS design envelope: off-nadir angle **≤ 45°** and attitude rate **≤ 50°/s**. Pre-flight POST trajectories averaged roughly **7–18° off nadir at 12–22°/s**; the MSL flight reconstruction was about 11° at 7°/s **[Read from plot]**. Field-test profiles called "flight like" sat near **20°** off nadir. In flight, pitch swung within about **−12° to +5°** with ~1–2 s oscillations. Heading (yaw) drifted **~70° over ~35 s** on the parachute **[Read from plot]**. | Johnson et al. 2017 p.3; Johnson et al. 2020 p.4 Fig. 3, p.6; Johnson et al. 2022 Fig. 14 p.15 | High for the envelope. **Medium** for the typical values (read from small figures). **Not verified:** any published distribution of heading. |
| **4. LCAM image quality** | Exposure: the requirement was **~1 ms**. Values stated later fall: ~400 µs (2016), 300 µs (2019), 200 µs (2021), 150 [µ]s (2023), **80 µs** (2022 camera-simulation paper). Motion blur is called "negligible" at 80 µs. | Johnson et al. 2017 Table 1; Montgomery et al. 2016, 2019 slides; Johnson 2021 slides; Johnson et al. 2023 Table 2; Aaron et al. 2022 p.3 | **Medium**: sources disagree; 80 µs is the latest. |
| | Noise: 32.6 e⁻ system noise, 139.4 e⁻/DN, 33,925 e⁻ full well, 8-bit output, "less than 2 DN of noise". Flight images averaged **~60 DN** against a 100 DN target. | Maki et al. 2020 Table 5 and LCAM section; Johnson et al. 2023 Table 2; Johnson et al. 2022 p.15 | High |
| | No automatic exposure. Exposure was set in advance from radiometric predictions, with a fallback re-exposure on coarse-match failure. LVS tolerated exposure **off by 4×** either way. | Johnson et al. 2017 p.14; Maki et al. 2020; Johnson et al. 2023 p.34 | High |
| **5. Sun geometry** | LVS design range: sun elevation **25–55°**, azimuth **240–310°** clockwise from north. Sun at LCAM time versus map imagery: **< 20°** apart. | Johnson et al. 2017 p.3; Johnson et al. 2020 p.1, p.6 | High |
| | Flight: elevation **≈ 39.8°**, azimuth **≈ 257.3°** (west-south-west) during the LVS window. Local mean solar time at touchdown was 15:53:26, which matches the press kit's 3:53 p.m. | **[Computed]** with the Mars24 algorithm (method below) | Medium–high (our computation, validated against GISS's worked example) |

**The one thing to push back on.** The brief says a nadir view at 2.2 km sees 4.4 km of a 9.49 km tile, leaving ~2.5 km each side for offset and tilt. That holds along the image's edges only. The sampler's own `check_fits_tile` uses the image **corner**, at 54.7° off axis, since heading is free. On that basis a nadir view at 2.2 km already reaches **3.1 km** from the camera, which leaves **1.6 km**, not 2.5 km. At 2.3 km, any tilt above **~9°** can see past the tile edge even with zero offset **[Computed]**. Flight-like tilts of 10–20° don't fit the 10 km tile under that check. See "What this means for the sampler" for options.

## 1. Handoff altitude

**The sequence.** The EDL navigation filter converges on landing-radar data at about 7 km above the ground. At 4200 m above ground level (AGL), LVS starts its coarse phase: three images, about 5 s. Then it takes about one fine image per second. The rover uses "the last valid LVS estimate" just before back-shell separation to pick a safe target (Johnson et al. 2017 p.3–4; Brugarolas et al. 2021 p.3). LVS kept processing down to 500 m in flight, but only fixes before BSS were used (Johnson et al. 2022 p.3, p.13).

**Design values.**
- LVS must work from 4200 m down to "2000m AGL or the spacecraft tells LVS that back shell separation has occurred" (Johnson et al. 2017 p.3). The field-test paper gives the same 4200–2000 m altitude envelope (Johnson et al. 2020 p.1).
- Mars 2020 raised the BSS altitude "from 1800 m to 2300 m" relative to MSL, to allow longer diverts (Way et al. 2020 p.9).
- The design concept of operations says BSS happens at "approx. 2100 m" (Brugarolas et al. 2021 p.3). The press kit says the powered descent vehicle separates at about 2.1 km (press kit p.22). These are round planning numbers.
- The pre-flight LVS Monte Carlo reported position error "at 2200m altitude" (Johnson et al. 2022 p.10, Fig. 8).

**Flown values.**

| Event | Altitude AGL | Source |
|---|---|---|
| LVS starts coarse matching | 4200 m (design threshold) | Johnson et al. 2022 p.13 |
| LVS valid | 3.7 km (3699 m in reconstruction) | Way et al. 2022 Fig. 1; Table 7 p.10 |
| Last LVS fix received by STS | received 0.90 s before BSS trigger | Brugarolas et al. 2021 p.9 (timestamps 666952874.85 and 666952875.75) |
| BSS triggered | 2276 m (navigation filter); 2275.6 m ("Wait for BSS", reconstruction) | Brugarolas et al. 2021 p.9; Way et al. 2022 Table 7 |
| Free fall from back shell | 2161 m | Way et al. 2022 Table 7 |

The reconstructed BSS altitude was a +0.47σ quantile of the pre-flight Monte Carlo (Way et al. 2022 Table 8, p.11). So the flight was typical, but the paper doesn't give the σ itself. **Not verified:** the pre-flight spread of BSS altitude.

**Altitude of the image behind the final fix [Inference + Computed].** The solution reached the rover 0.9 s before BSS. A fine image takes about 1.1 s from capture to updated state. The fine-processing allocations add up to that: 171.9 + 750.0 + 93.8 + 78.1 ms (Johnson et al. 2022 Table 2). So the image was exposed roughly 1–2 s before BSS. The vehicle was descending at about 81 m/s (Way et al. 2022 Table 7), which puts the exposure at **about 2.36–2.44 km**, above the 2.28 km BSS altitude. Johnson et al. 2022 round this to "approximately 2200m". The exact exposure altitude isn't published (**not verified**).

## 2. Horizontal position: knowledge error vs true position

These are two different things, and the sampler should keep them apart:

- **Knowledge error** is how wrong the spacecraft's own belief about its position is before LVS corrects it. It sets how far the matcher has to search.
- **True-position dispersion** is where the vehicle actually is, relative to the target or the tile centre. It sets what ground the camera sees.

**Knowledge error.**
- Before LVS, the MSL-heritage inertial navigation carries position from the last ground fix before entry. "the error on this position can be as large as 3.2km" (Johnson et al. 2017 p.1). The LVS requirement was to cut 3.2 km to 40 m within 10 s (p.2). JPL slides label this "3.2km 99%" (Montgomery et al. 2017 and 2019 slides) and "3km 3σ" (Montgomery et al. 2016 slides).
- In the LVS simulation, horizontal initial error was drawn Gaussian, bounded at ±3.2 km per axis. Vertical error was ±100 m, attitude error < 0.5° per axis, and velocity error < 2 m/s per axis (Johnson et al. 2017 p.12). LVS still worked with 5 km of initial error (p.13, Table 3).
- **Flight:** the coarse phase found "a 137m position shift" between LVS and the spacecraft estimate (Johnson et al. 2022 p.13). The STS reconstruction gives a correction of [125, −14.8] m, east and north (Brugarolas et al. 2021 p.9). So flight was far inside the 3.2 km design bound.

**True-position dispersion.**
- The project required a 10 km × 10 km landing ellipse (McGrew et al. p.1–2).
- Pre-flight, with the range trigger, 8000 Monte Carlo runs at Jezero gave a **99%-tile ellipse of 7.1 × 6.5 km** (Way et al. 2020 p.5, Fig. 5). Way et al. describe the back-shell separation points (the landing points without TRN) as approximately bivariate normal (Way et al. 2020 p.10, Fig. 17 discussion).
- Post-flight, McGrew et al. state the ellipse as "7.5 x 5.2 km" in the abstract and "7.4 km x 5.1 km" in the introduction (p.1). Treat it as about 7.5 × 5.2 km. The paper doesn't say what probability level it uses (**not verified**).
- **[Computed]** For a 2D Gaussian, the 99% ellipse's semi-axes are 3.03σ. So 7.1 × 6.5 km means σ ≈ **1.17 km** along the long axis and **1.07 km** along the short one, about the ellipse centre (the pre-entry target).
- **Flight:** the zero-divert point (where the vehicle would have landed without diverting) was 1.134 km south-east of the pre-flight target, a 1.03σ quantile (Way et al. 2022 p.3). From HiRISE, the landed rover was 1746 m from the pre-entry target (Way et al. 2022 Table 1).
- **[Computed]** Relative to the landed rover, which is the Fidelis tile centre, the reconstructed position was about 0.84 km west and 0.48 km north at "LVS valid", and 0.63 km west and 0.46 km north at BSS (0.78 km away). This uses Way et al. 2022 Table 7 and R = 3,396.19 km.

**Side finding.** Way et al. 2022 Table 1 (p.3) gives the landed rover at **18.44463 N, 77.45088 E** from HiRISE. That's a NASA primary source for the tile centre that `config/jezero.yaml` says it lacks (18.4447, 77.4508): the two agree to within about 5 m.

## 3. Attitude on the parachute

**Design envelope.** "Attitude rates can be as high as 50°/s and off nadir angles up to 45°" (Johnson et al. 2017 p.3; repeated in Johnson et al. 2020 p.1 and Johnson 2021 slides). LCAM's 90° field of view was sized to see enough map "even when the camera is pointed up to 45° off-nadir" (Johnson et al. 2017 p.5). During LVS, the flight software checks the off-nadir angle at 64 Hz before each image (Johnson et al. 2023 p.33).

**Typical values.** The field-test gimbal profiles were built from POST Monte Carlo trajectories. The team extracted "statistics on off nadir angle, attitude rates and frequency and amplitude of oscillations" (Johnson et al. 2020 p.3–4).
- Fig. 3 of that paper plots mean off-nadir angle against mean attitude rate. The POST points cluster at about **7–18° and 12–22°/s**, and the MSL flight reconstruction sits near **11° and 7°/s** **[Read from plot]**.
- The text says many gimbal profiles "were flight like with an off-nadir angle near 20°" (p.6).
- The field test's operational envelope repeats 0–45° and 0–50°/s (Table 3, p.10).

**Flight.** Johnson et al. 2022 Fig. 14 (p.15) plots LVS attitude estimates, labelled as showing "the attitude oscillations of parachute descent". **[Read from plot]**:
- Pitch oscillated with a period of roughly 1–2 s and amplitude of a few degrees, within about −12° to +5° on the parachute.
- Roll stayed within about 20° of ±180°. The frame convention isn't stated, so this is probably near-level about that axis, but that reading is uncertain.
- **Yaw drifted from about −90° to −20° over the ~35 s before MRL start**, then turned back to about −35° by BSS.

So the vehicle slowly rotates about the vertical on the parachute. There's no reason to fix the image heading.

**Heading in sources.** No source gives a heading or yaw distribution. LVS estimated "about nadir" angle errors, and its tests varied that error up to 5° (Johnson et al. 2017 Table 3). That concerns knowledge error, not the true heading. **Not verified:** any statement that heading is uniformly random.

## 4. LCAM image quality

**Camera.** LCAM is an On Semiconductor Python 5000 global-shutter CMOS sensor, summed 2×2 to 1024 × 1024. Images are "always summed 2 × 2 and sent with 8-bit pixel depth". Other values from Table 5 and the LCAM section of Maki et al. 2020:
- full well 33,925 e⁻
- system noise 32.6 e⁻
- conversion gain 139.4 e⁻/DN
- maximum SNR 45 dB
- dynamic range 60 dB
- pixel scale 1.67 mrad

The SNR requirement was ">80:1" at about 1 ms (Maki et al. 2020). Johnson et al. 2017 Table 1 gives SNR 80 "at half full well depth". Johnson et al. 2023 Table 2 gives 70, and says the design's "10 bit images binned 2x2 results in less than 2 DN of noise" (p.14).

**Noise budget [Computed].** Assume the 139.4 e⁻/DN gain applies to the 8-bit output. That fits: 33,925 / 139.4 ≈ 243 DN. Then:

| Mean signal | Shot noise | Read noise | 8-bit quantisation | Total | As a fraction of 255 | SNR |
|---|---|---|---|---|---|---|
| 60 DN (flight mean) | 0.66 DN | 0.23 DN | 0.29 DN | 0.75 DN | 0.30% | ~80 |
| 100 DN (target) | 0.85 | 0.23 | 0.29 | 0.92 | 0.36% | ~108 |
| 243 DN (full well) | 1.32 | 0.23 | 0.29 | 1.37 | 0.54% | ~177 |

That fits the "< 2 DN" design statement. The simulator also modelled fixed-pattern noise and pixel non-uniformity, measured in the lab; dark current, which varies only 1.5 DN over temperature; vignetting; and bad pixels (Aaron et al. 2022 p.9–10). The 2017 simulator used column-to-column variation under 0.5% and 10 hot pixels (Johnson et al. 2017 p.9).

**Contrast matters more than noise.** Jezero imagery is low-contrast. The flight images' largest local standard deviation in 64-pixel windows was 10.9 DN (Aaron et al. 2022 Fig. 12, p.12). Coarse-match interest scores were lower than in the terrestrial field tests, because the site "is covered with dust and low contrast" (Johnson et al. 2022 p.16). The rendering model reproduced flight images with a dust optical depth τ₀ = 0.47 (Aaron et al. 2022 p.11). LVS was tested down to 4 DN of scene contrast (Johnson et al. 2017 p.15). **[Inference]** A noise floor of under 1 DN against 5–11 DN of local contrast is the regime the sampler should reproduce. Noise far above that tests a camera Perseverance didn't fly.

**Exposure time.** The sources disagree, probably because the design changed over time:
- "~1ms" was the requirement (Johnson et al. 2017 Table 1; Maki et al. 2020).
- "~400µs" nominal (Montgomery et al. 2016 slides).
- "300us" (Montgomery et al. 2019 slides).
- "200us" (Johnson 2021 slides p.3).
- "150 s", with the unit glyph lost in the PDF text and almost certainly µs (Johnson et al. 2023 Table 2 p.14).
- "the short camera exposure time of 80 µs makes this effect negligible" (Aaron et al. 2022 p.3, on motion blur). This is the most recent, written after landing by the team that rendered flight images.

**Not verified:** the exact exposure commanded on landing day.

**Motion blur [Computed].** LCAM's pixel is 1.67 mrad. Rotation smears an image by (rate × exposure) / IFOV:

| Rate | 80 µs | 150 µs | 1 ms (requirement) |
|---|---|---|---|
| 10°/s | 0.008 px | 0.016 px | 0.10 px |
| 20°/s (typical) | 0.017 px | 0.031 px | 0.21 px |
| 50°/s (envelope max) | 0.042 px | 0.078 px | 0.52 px |

Translation blur is smaller still: 15 m/s × 80 µs at 2.3 km is about 0.0003 px. So in flight, blur was effectively zero. Even at the 1 ms requirement and the 50°/s maximum, it stays around half a pixel.

**How exposure was set.** "The LCAM does not have automatic exposure control" (Johnson et al. 2017 p.14). Exposure was computed in advance from predicted signal levels. The VCE can change it if images look too bright or too dark (Maki et al. 2020). The flight software retries with an adjusted exposure only when coarse matching fails *and* a bad exposure is detected (Johnson et al. 2023 p.34).
- Sensitivity: LVS matched from full brightness (mean 128 DN) down to 1/8 (16 DN). The team concluded exposure "could be off by a factor of 4 either too long or too short" (Johnson et al. 2017 p.14).
- Flight: mean image brightness "hovered around 60 DN", below the 100 DN target. Matching wasn't affected, and the likely cause was "uncertainty in the Mars radiometry" (Johnson et al. 2022 p.15).

## 5. Sun geometry

**Design range.** "Sun elevations can be between 25° and 55° above the horizon and sun azimuths can be between 240° and 310° clockwise from north" (Johnson et al. 2017 p.3). The spread comes from variation in landing date, time of day and latitude. The same range appears in Johnson et al. 2020 p.1 and Johnson 2021 slides p.4. The difference between the sun direction at LCAM time and at map-image time "is less than 20°" for Mars 2020 (Johnson et al. 2020 p.6).

**Landing-day sun [Computed].**
- Inputs:
  - Touchdown at **20:43:49 UTC** spacecraft event time, 18 February 2021 (Abilleira et al. 2021, NTRS abstract). Entry was at 20:36:50 UTC, so LVS ran from about 20:42:29 (E+5:39, valid) to 20:42:48 (E+5:58, BSS), using Way et al. 2022 Fig. 1.
  - Site 18.44463 N, 77.45088 E (Way et al. 2022 Table 1).
  - TT − UTC = 69.184 s (37 leap seconds + 32.184 s).
- Method: the Mars24 algorithm, steps A–D, from NASA GISS (Allison & McEwen 2000). Elevation = 90° − Z from step D-5. Azimuth from step D-6 using atan2, which gives a bearing clockwise from north.
- Check: the code reproduces GISS's worked example 1 to 10⁻⁴°. It matches MST, Ls, EOT, δs, subsolar longitude, Z and azimuth.
- Scratch script: not committed. The numbers are the ones it printed.

| Time (UTC, spacecraft) | LMST | LTST | Ls | Sun elevation | Sun azimuth (clockwise from N) |
|---|---|---|---|---|---|
| 20:42:29 (LVS valid) | 15:52:08 | 15:14:13 | 5.64° | 39.8° | 257.3° |
| 20:42:48 (BSS) | 15:52:27 | 15:14:32 | 5.64° | 39.8° | 257.3° |
| 20:43:49 (touchdown) | 15:53:26 | 15:15:31 | 5.65° | 39.5° | 257.4° |

Three cross-checks:
- The press kit gives touchdown at "3:53 p.m. local mean solar time" (p.14, p.24). The computation agrees to the minute.
- Way et al. 2020 give the arrival season as Ls = 5.6 (p.3). That matches.
- Flight falls inside the 25–55°, 240–310° design box.

The computation ignores local terrain slope and the planetocentric/planetographic latitude difference, both well under 1°. Over the 19 s of LVS the sun moves less than 0.1°. Aaron et al. 2022 likewise call the change "negligible" (p.5).

## What this means for the sampler (#6)

`src/fidelis/scenario.py` (in progress, uncommitted) draws each input from a bounded distribution and refuses configs whose worst case could see past the tile. Below, **From sources** means a number a source gives. **Fidelis choice** means a modelling decision we're making that no source dictates.

### The tile is the binding constraint, and it bites harder than the brief assumes

`check_fits_tile` assumes the worst case: the highest altitude, the image corner (54.7° off axis for a 90° square view) and the full tilt, all lined up with the largest offset. That's correct when heading and tilt direction are both free, as they are here. With the east–west half-width of 4743 m (5000 m × cos 18.4447°) and no relief allowance, **[Computed]**:

| Altitude | Corner reach at nadir | Spare for offset at nadir | Max tilt with zero offset | Max tilt if only edges counted |
|---|---|---|---|---|
| 2200 m | 3111 m | 1632 m | 10.4° | 20.1° |
| 2300 m | 3253 m | 1490 m | 9.4° | 19.1° |
| 2450 m | 3465 m | 1278 m | 7.9° | 17.7° |

Adding terrain relief, as `load_config` does, shrinks these further. So with the current check and tile, the sampler can't combine flight-like tilts (10–20°) with offsets of even a few hundred metres. A larger HiRISE tile isn't available: the largest valid square centred on the site is 10.14 km (`descent-altitude-and-tile-size.md`). A coarser CTX outer ring could widen it to about 26 km of ground; see `ctx-terrain-for-a-wider-tile.md`. The choices, all **Fidelis choices**:

1. **Keep the worst-case check and narrow the inputs.** Use small tilt (≤ ~5°) and small offset (≤ ~0.8 km). Simple and provable, but the attitude is less realistic than flight.
2. **Check each draw's actual footprint.** Use its heading, tilt direction and the DEM, and reject or redraw draws that leave the tile. That allows realistic tilts, but it biases the joint distribution: large-tilt + large-offset + unlucky-heading draws get removed. The rejection rate must be reported, and the failure estimate is then *conditional on the view fitting the tile*.
3. **Couple heading to tilt direction** so the corner never lines up with the lean. That's only defensible if a source shows the swing plane is fixed relative to the camera axes, and **none does** (not verified).

My recommendation is (2), with the rejection rate logged. Option (1) quietly removes the attitude realism that is part of what LVS was tested for.

### Suggested distributions

| Input (`Scenario` field) | From sources | Suggested starting distribution (**Fidelis choice** unless stated) |
|---|---|---|
| `altitude_m` | BSS at 2276 m flown; design 2300 m; final-fix image ~2.36–2.44 km **[Inference]**; "approximately 2200m" (Johnson et al. 2022). BSS spread **not verified**. | Uniform 2200–2450 m. It brackets every sourced handoff number. The lower end gives more tile margin. |
| `offset_east_m`, `offset_north_m` | True dispersion σ ≈ 1.1–1.2 km about the ellipse centre (from the 7.1 × 6.5 km 99% ellipse); flight BSS point 0.78 km from tile centre. | Truncated normal, mean 0, sd 1100 m, truncated to whatever the tile allows (under the current check, about ±1200 m at nadir; less with tilt). Centring on the tile centre rather than the pre-entry target is a choice. The truncation is a tile limit, not physics, and should be said so in results. |
| `tilt_deg` | Envelope 0–45°; POST means ~7–18° **[Read from plot]**; "flight like ... near 20°"; flight pitch within ~−12° to +5° **[Read from plot]**. | With option (2): truncated normal, mean ~12°, sd ~6°, bounds 0–25°. With option (1): uniform 0–5°. The 45° envelope edge is a stress case, not a typical value. |
| `tilt_azimuth_deg` | No source. | Uniform 0–360°. |
| `heading_deg` | Yaw drifted ~70° in ~35 s on the parachute **[Read from plot]**; no distribution published. | Uniform 0–360°. |
| `noise_sd` (fraction of full scale) | ~0.75 DN at 60 DN mean, < 2 DN by design → 0.3–0.8% of 255 **[Computed]**. | Uniform 0.002–0.008. Anything higher is a stress test, not LCAM. Consider also modelling scene contrast (dust), which mattered more in flight. |
| `blur_px` | ≤ 0.04 px at 80 µs; ≤ 0.52 px at the 1 ms requirement and 50°/s **[Computed]**. | Uniform 0–0.5 px, bounding the 1 ms requirement. Flight-faithful would be ~0; the range only exists as a sensitivity knob. |
| `blur_angle_deg` | No source. | Uniform 0–360°. |
| `exposure_gain` | Flight was 0.6× target (60 vs 100 DN); LVS tolerates 1/4× to 4× (tested down to 1/8 brightness). | Log-uniform-ish over [0.5, 1.5], or uniform 0.5–1.5, centred near the flight value. The source tolerance [0.25, 4] is a stress range. The current module has no log-uniform; uniform is fine to start. |
| `sun_index` | Flight ≈ (257.3° az, 39.8° el) **[Computed]**; design box el 25–55°, az 240–310°; LCAM-to-map sun difference < 20°. | Index 0 = flight sun (257°, 40°). The other four span the design box, e.g. (240°, 25°), (310°, 25°), (240°, 55°), (310°, 55°). The orthoimage texture already carries HiRISE-time shadows, so the rendered-sun vs texture-sun difference should stay modest. |

### Not in the scenario, but worth a decision

**Knowledge error.** The `Scenario` dataclass has no field for the navigator's *prior* position error, which is design ≤ 3.2 km at 99% but was 137 m in flight. If the Fidelis matcher uses a prior to crop its search window, as LVS did, that error is an input in its own right. If the matcher searches the whole tile, it doesn't matter. Flagging this without building anything.

## Sources

1. Johnson, A., Aaron, S., Chang, J., Cheng, Y., Montgomery, J., Mohan, S., Schroeder, S., Tweddle, B., Trawny, N., Zheng, J. (2017). The Lander Vision System for Mars 2020 Entry Descent and Landing. AAS Guidance and Control Conference, Breckenridge, CO. NTRS lists it as AAS 17-036; Aaron et al. 2022 cite AAS-17-038. NTRS record https://ntrs.nasa.gov/citations/20190025658 (no PDF there). Preprint read from JPL's open repository: https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/46186
2. Johnson, A., Villaume, N., Umsted, C., Kourchians, A., Sternberg, D., Trawny, N., Cheng, Y., Geipel, E., Montgomery, J. (2020). The Mars 2020 Lander Vision System Field Test. AAS 20-105, AAS Guidance and Control Conference. Preprint: https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/52354 ; NTRS https://ntrs.nasa.gov/citations/20220000862
3. Johnson, A. E., Aaron, S., Ansari, H., et al. (2022). Mars 2020 Lander Vision System Flight Performance. AIAA SciTech 2022 Forum. https://doi.org/10.2514/6.2022-1214 ; preprint read: https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/56099
4. Johnson, A. E., Cheng, Y., Trawny, N., Montgomery, J. F., Schroeder, S., Chang, J., Clouse, D., Aaron, S., Mohan, S. (2023). A Map Relative Localization System for Planetary Landing. JPL preprint: https://doi.org/10.48577/jpl.TWOU9P . Published as "Implementation of a Map Relative Localization System for Planetary Landing", *J. Guidance, Control, and Dynamics*, https://doi.org/10.2514/1.G006780 (journal version not read; page numbers are the preprint's).
5. Aaron, S., Cheng, Y., Trawny, N., Mohan, S., Montgomery, J., Ansari, H., Smith, K., Johnson, A., Goguen, J., Zheng, J. (2022). Camera Simulation for the Perseverance Rover's Lander Vision System. AIAA SciTech 2022 Forum. https://doi.org/10.2514/6.2022-0746 ; copy read: https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/56110
6. Mohan, S., Johnson, A. E., Brugarolas, P., Cheng, Y., Otero, R., Trawny, N., Aaron, S., Serricchio, F., Montgomery, J., Casoliva, J., Chen, A. (2021). Assessment of M2020 Terrain Relative Landing Accuracy: Flight Performance vs. Predicts. AAS 21-541, AAS/AIAA Astrodynamics Specialist Conference, Big Sky, MT. Preprint: https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/55252 ; NTRS https://ntrs.nasa.gov/citations/20230007014
7. Brugarolas, P., Casoliva, J., Way, D. W., Dutta, S. (2021). Mars 2020 Perseverance EDL GNC Safe Target Selection Reconstruction. AAS 21-599, AAS/AIAA Astrodynamics Specialist Conference. Preprint: https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/55194
8. Way, D. W., Dutta, S., Zumwalt, C. H., Blette, D. J. (2022). Assessment of the Mars 2020 Entry, Descent, and Landing Simulation. AIAA SciTech 2022 Forum. https://doi.org/10.2514/6.2022-0421 ; PDF read from NTRS: https://ntrs.nasa.gov/citations/20210024480
9. Way, D., Dutta, S., Zumwalt, C., Santini De León, S. (2020). EDL Simulation Results for the Mars 2020 Landing Site Safety Assessment. IEEE Aerospace Conference. https://doi.org/10.1109/AERO47225.2020.9172525 ; PDF read from NTRS: https://ntrs.nasa.gov/citations/20200002978
10. McGrew, L. C., Killeen, B. J., Mendeck, G. F. Entry Guidance Design and Post-Flight Performance of the Mars 2020 Mission. AAS/AIAA preprint (NTRS file labelled January 2025). https://ntrs.nasa.gov/citations/20240015538
11. Abilleira, F., et al. (2021). Mars 2020 Perseverance Trajectory Reconstruction and Performance from Launch through Landing. AAS/AIAA Astrodynamics Specialist Conference. NTRS abstract only (for the touchdown time 20:43:49 SCET UTC): https://ntrs.nasa.gov/citations/20220000775
12. Maki, J. N., et al. (2020). The Mars 2020 Engineering Cameras and Microphone on the Perseverance Rover: A Next-Generation Imaging System for Mars Exploration. *Space Science Reviews* 216, 137. https://doi.org/10.1007/s11214-020-00765-9 ; open access https://pmc.ncbi.nlm.nih.gov/articles/PMC7686239/ (LCAM section, Table 5)
13. NASA/JPL (2021). *Mars 2020 Perseverance Landing Press Kit*, February 2021. https://www.jpl.nasa.gov/news/press_kits/mars_2020/download/mars_2020_landing_press_kit.pdf (pp.14, 21–22, 24)
14. JPL presentation slides, all by the LVS team; the PDF has no page numbers, so citations are by slide content:
    - Montgomery, J., et al. (2016). The Mars 2020 Lander Vision System. IPPW 2016. https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/46727
    - Montgomery, J., et al. (2017). The Mars 2020 Lander Vision System: Architecture and I&T Results. IPPW 2017. https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/48016
    - Montgomery, J., et al. (2019). The Mars 2020 Lander Vision System: Architecture and V&V Results. IPPW 2019. https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=hdl:2014/51096
    - Johnson, A. (2021). Mars 2020 Lander Vision System, Workshop for Lunar Mapping for Precision Landing, 2 March 2021. https://lsic.jhuapl.edu/uploadedDocs/presentations/764-1.09_Johnson_20210302_LuNaMaps_LVS_v0.pdf
15. Allison, M., McEwen, M. (2000). A post-Pathfinder evaluation of areocentric solar coordinates with improved timing recipes for Mars seasonal/diurnal climate studies. *Planetary and Space Science* 48, 215–235. https://doi.org/10.1016/S0032-0633(99)00092-6 . Algorithm as published by NASA GISS (Mars24), used for the sun computation: https://www.giss.nasa.gov/tools/mars24/help/algorithm.html
16. Cheng, Y., Ansar, A., Johnson, A. (2021). Making an Onboard Reference Map From MRO/CTX Imagery for Mars 2020 Lander Vision System. *Earth and Space Science* 8, e2020EA001560. https://doi.org/10.1029/2020EA001560 . Not read beyond its abstract (publisher returned 403). Its "as large as 3.2 km" figure is the same one stated in source 1.

## Open questions / not verified

- The pre-flight Monte Carlo spread of BSS altitude, and of altitude at the last LVS image. Way et al. 2022 give only the flight's quantile (+0.47σ).
- The exact LCAM exposure time on landing day. Sources range from 80 µs to ~1 ms.
- Any published heading or yaw distribution on the parachute, and whether the swing plane is fixed relative to the camera.
- The probability level of McGrew et al.'s post-flight 7.5 × 5.2 km ellipse, and why the paper gives two slightly different sizes.
- The typical attitude values above were read from small published figures (Johnson et al. 2020 Fig. 3; Johnson et al. 2022 Fig. 14). Treat them as ±2–3°.
- The noise budget assumes the 139.4 e⁻/DN gain applies to the 8-bit output. The full-well arithmetic (≈ 243 DN) supports that, but no source states it.
