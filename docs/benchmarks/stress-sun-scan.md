# Stress-sun scan: does a map lit from over 100° away make the navigator miss? (issue #43)

**A diagnostic, not a failure rate.** Nothing in this file is ever reported as the realistic failure rate (ADR 0003, decisions 1 and 7). Every record and estimate it produces is labelled as a stress case.

The CTX map scan found 0 failures in 300 (`docs/benchmarks/ctx-map-scan.md`), with every Fidelis sun at most 39.0° from the map's. JPL's field misses all came from runs with a map-to-image sun difference above 100°. This scan moves the simulated sun past 100° and asks whether misses appear, as a dose-response. It renders from the de-shaded texture (#19, `docs/benchmarks/deshade.md`), so the simulated sun is the main lighting in the image rather than an extra layer on top of the map's baked one.

This file fixes what will be measured **before** the run. Results go under the plan afterwards, and any departure from the plan is stated there with its reason.

## Plan

**Status: pre-registered 2026-11-13, before any descent was rendered from the de-shaded texture.** The suns and the swapped controls were approved by the user. Nothing in this section is a measurement.

### The suns

"Sun difference" is the angle between the two sun directions in 3D, as in `docs/research/ctx-and-hirise-sun-angles.md`, section 6. The map's sun over the tile is CTX `J03_045994_1986`: bearing 267.37°, elevation 42.42° (the research note, section 2). The HiRISE texture over the tile centre (`ESP_045994_1985`) was taken 14 s later and is lit the same way.

| Role | Bearing | Elevation | Sun difference from the map [Computed] |
|---|---|---|---|
| Anchor (realistic sun 3, run on the de-shaded texture) | 310° | 25° | 39.0° |
| Stress 1 | 357.4° | 25° | 73.4° |
| Stress 2 | 37.4° | 25° | 98.3° |
| Stress 3 | 87.4° | 25° | 112.6° |

**Why one elevation.** Holding the elevation at 25° means only the bearing changes along the dose axis, so cast-shadow length and overall shading strength stay the same. 25° is also the bottom of LVS's design box, and the anchor is already one of the five realistic suns.

**Why the top dose is only 112.6°.** At 25° elevation, the sun directly opposite the map's is 112.6° away: 180° − 42.42° − 25°. At the map's own elevation the opposite sun is only 95.2° away. Going past about 113° needs a lower sun (87.4° / 15° would be 122.6° away). That sun is left out because it adds long cast shadows, which de-shading doesn't remove, as a second change at once.

**Physical plausibility isn't the test.** Stress 1 and 2 put the sun almost due north and north-east at 25° elevation. At Jezero's latitude (18.4° N, with Mars's sun never more than about 25° north of the equator) that sky position probably never occurs **[Inference]**. This is a diagnostic of the matcher, not a scenario that happens in practice. The alternative is to mirror them to the south side (bearings 177.4° and 137.4° give the same differences), which is no more realistic at 25°.

### What runs

- **Navigator.** The real one: ALIKED + LightGlue, true attitude, matching against the CTX map (`matching_map.source: ctx`), offset +1.89 m east, −1.49 m north removed, as in the CTX map scan.
- **Renderer.** The mesh renderer, `texture_deshaded.png`.
- **Seeds.** 40,000–40,299 (300 descents), **the same seeds under every sun**. Taken so far: 0–299, 10,000–10,299, 20,000–20,299, 30,000–30,003. Sharing seeds means a seed's position, altitude, tilt, heading, noise, blur and exposure are the same at every dose; only the sun changes. That makes the per-seed change along the dose axis the main comparison, not just four separate counts.
- **Sun.** Pinned per run, from a `stress_suns` list in `config/jezero.yaml` that the realistic sampler can't draw (the ticket's acceptance criteria). The anchor is pinned from the realistic `suns` list but rendered with the de-shaded texture, so its records are labelled stress-case too.
- **Hardware.** One NVIDIA A40 (`docs/carc.md`). All five runs go in one `sbatch` job on `debug`, so they share a node and a GPU type.

### Controls

1. **The anchor (zero dose).** The 39.0° sun on the same texture and seeds. It separates "the de-shaded texture plus a 25° sun" from "the sun difference". If the anchor already fails, the texture is the cause, not the sun.
2. **Photo texture at Stress 3 (what #19 changed).** The 112.6° sun on the original `texture.png`, same seeds. The #19 laptop check predicts this stays closer to the map than the de-shaded render (correlation proxy, median 0.823 vs 0.623). If it does worse, the prediction behind building #19 was wrong, and that gets written up.

**Departure from the ticket: no `--matching-map scene` control.** The ticket asked for each sun to also run against the old map, to separate "an odd sun alone" from "an odd sun against the real map". That no longer works. The old map is the photo texture averaged to 6 m, and over the tile centre the photo carries the same MRO pass's lighting as the CTX map. So both maps disagree with a stress sun in the same way, and the comparison would mostly measure CTX vs HiRISE detail and elevation, which the CTX map scan already sized (median error ×1.8). The anchor answers the question that's left. The user can still ask for this control.

### What counts as a failure (fixed by ADR 0003, not chosen here)

- **r = 40 m in the map frame.** A descent fails if it gets no fix, or its fix is more than 40 m from the truth after the offset is removed.
- **No-fix and beyond-r failures are counted separately.** Under a stress sun a lost match (no fix) is the more likely first sign, and it's a different failure from an inaccurate fix.
- r isn't tuned, whatever the count turns out to be.

### What gets reported

1. **By sun, in dose order:** failures out of 300 (no-fix and beyond-r separately) with the 95% Wilson interval; map-frame error median, 90th percentile and maximum; inlier median and minimum.
2. **Error against sun difference:** the four de-shaded runs as a curve, plus the per-seed change from the anchor at each dose.
3. **Against the photo-texture control:** the same row for the 112.6° sun on `texture.png`, and the per-seed difference from Stress 3.
4. **Exploratory, labelled as such:**
   - how far each view reaches into the east strip (`ESP_048842_1985`, baked sun 219.2°, which de-shading barely corrected) and into the CTX ring (CTX's own baked suns, not de-shaded), against error. Every view at these altitudes reaches past the tile centre to some degree, so this is a split, not a filter;
   - no-fix descents: which seeds, and whether they repeat across doses;
   - match counts, and the map's keypoint count.

### Prediction from the sources

Steps of the literature null-case test (`CLAUDE.md`).

1. **Conditions and denominator.** Johnson et al. 2020, via `docs/research/lvs-error-sources.md`, section 3: inside a 100° sun difference, all 393 runs in the expanded envelope met 40 m. Beyond that envelope, "only 7 of the additional 263 runs" exceeded 40 m, and all 6 the source traces came from one late-afternoon Kelso Dunes run with sun differences above 100°. The 263 also varied other things (entropy, altitude, rates), and the source doesn't say how many of them sat above 100°. So no rate per run conditioned on sun difference can be read from it.
2. **Where we sit on the source's axis.** Anchor 39.0°, Stress 1 73.4°, Stress 2 98.3°: all inside the source's no-miss zone. Stress 3 at 112.6° is outside it.
3. **The prediction.**
   - **Anchor, Stress 1, Stress 2: 0 failures expected** at 40 m, from the source's 393 of 393 inside 100°.
   - **Stress 3: the source doesn't cover our conditions** well enough to give a number. Its misses came from one field run on Earth dunes with a real camera; ours is a relit render of Mars terrain. It says misses become *possible* above 100°, not how often.
   - **Our own proxy, not a source:** the #19 laptop check gives render-to-map similarity under the 112.6° sun of median 0.623, 10th percentile 0.403, against 0.93–0.99 under the realistic suns (high-passed correlation over 256 m windows, the photo standing in for the map). That predicts fewer inliers and larger errors at Stress 3 than at the anchor. It doesn't predict whether they cross 40 m.

### What each outcome means

- **0 failures at every dose.** Inconclusive, stated plainly: not "LVS is immune to sun difference". The render is still a relit photo. Shading finer than the 4 m DTM, cast shadows and the east strip still carry baked light, and the matcher may be relying on albedo, which doesn't change with the sun. The error-vs-dose curve and the inlier counts are then the result. The 95% upper bound per sun would be about 1.3%.
- **Failures only at Stress 3.** Fits the source's 100° line. Check the anchor is clean and see whether the misses cluster by view position (east strip, ring) before reading them as a sun effect.
- **Failures at Stress 1 or 2 (inside 100°).** Disagrees with the source. The first suspect is the renderer, not the matcher: a 25° de-shaded render may be less like a real image than JPL's field images were like their maps. Check the anchor first.
- **Failures at the anchor.** The texture or the 25° sun is the cause, not the sun difference. The scan doesn't measure what it set out to, and that gets written up before anything else.
- **Photo-texture control fails less than Stress 3.** Expected; it's #19 doing its job. If it fails more, #19's prediction was wrong.

None of these outcomes changes the realistic failure rate or tunes r.

### Rough cost (an estimate, not a measurement)

Five runs × 300 descents = 1,500 descents. The CTX map scan ran 600 descents in 5 min 23 s on an A40 (about 129 SUs). At that rate this is roughly 13–15 minutes and 300–360 SUs, inside the debug partition's 1-hour limit. It costs more if the de-shaded scene, or the low sun, slows rendering. Not submitted until the user says so (`CLAUDE.md`, compute guardrail).

### Known before the run

- **No descent had been rendered from the de-shaded texture when this plan was written.** `terrain_scene` accepted `texture=`, but neither `fidelis-plain-mc` nor the simulator passed it. Building that, the `stress_suns` list and the stress labelling came first. Laptop smoke descents on seeds outside 40,000–40,299 check the plumbing and are reported only as a smoke test.
- **De-shading is weak outside the tile centre.** In the east strip the fit explains a median of 5% of local brightness (`deshade.md`), and the CTX ring isn't de-shaded at all. A view's lighting there is still partly baked.
- **The 0.623 proxy is a stand-in** for ALIKED + LightGlue, with the photo standing in for the map.

## How to run it (added after the plan was committed)

The plan above was committed as `745b240`. This section adds the build and the commands. Nothing in the plan changed.

**What was built (#43).** `stress_suns` and `ctx_map_sun` in `config/jezero.yaml`. `fidelis-plain-mc --stress-sun <i>` pins a stress sun, and `--texture texture_deshaded.png` picks the de-shaded texture (`src/fidelis/stress.py`). Every record carries `sun_list` and `texture`. Every estimate carries `stress_case` and `pinned_sun`: the one sun the run used, with its angle from the map's sun (39.0° for the anchor). Its summary begins `STRESS CASE, not a failure rate`. A run on the de-shaded texture must pin its sun; otherwise it is refused before anything renders. `estimate()` refuses to pool stress records with realistic ones, two textures, or two stress suns. The stress suns are indexed 0–2: Stress 1 is `--stress-sun 0`, Stress 3 is `--stress-sun 2`.

**On CARC**, in the clone, with this branch checked out and `uv sync` run (the CTX map scan's first job failed without it, `docs/carc.md`). First the de-shaded texture (CPU, about 25 s on the laptop; it isn't part of the default build):

    uv run --no-sync snakemake --profile profiles/carc data/jezero/mesh/texture_deshaded.png

Then all five runs as one batch job, so they share a node and wait in the queue without holding a terminal:

    sbatch --account=irahbari_1147 --partition=debug --gres=gpu:a40:1 -c 8 --mem=32G -t 45 -J stress_sun_scan -o results/stress_sun_scan_%j.log --wrap "module purge && module load ver/2607 gcc/15.3.0 cuda/13.0.3 && R='uv run --no-sync fidelis-plain-mc config/jezero.yaml data/jezero' && S='--fidelity mesh --first-seed 40000 --samples 300' && \$R results/stress_sun/anchor \$S --sun-index 3 --texture texture_deshaded.png && \$R results/stress_sun/stress1 \$S --stress-sun 0 --texture texture_deshaded.png && \$R results/stress_sun/stress2 \$S --stress-sun 1 --texture texture_deshaded.png && \$R results/stress_sun/stress3 \$S --stress-sun 2 --texture texture_deshaded.png && \$R results/stress_sun/stress3_photo \$S --stress-sun 2"

The log ends with one `STRESS CASE` line per run. Each folder has `records.jsonl` and `estimate.json`. The 45-minute limit is about three times the cost estimate above; at 24 SUs a minute it caps the job at 1,080 SUs [Computed].

## Smoke test on the laptop (after the plan was committed; not a result)

Run on 2026-11-13 to check the plumbing end to end. Seeds 30,004–30,005 (outside the scan's, and now used), mesh renderer, RTX 5070 Laptop GPU, the map **thinned to its 8,000 highest-scoring keypoints** (`--fidelity mesh-k8000`), because the laptop can't hold the full map's matching. So this is not the real navigator, not the pinned GPU, and two descents per run: it says the code runs, and nothing about a failure rate.

| Run | Seed 30004: inliers / map-frame error | Seed 30005: inliers / map-frame error |
|---|---|---|
| Anchor (39.0°, de-shaded) | 293 / 5.47 m | 247 / 7.68 m |
| Stress 3 (112.6°, de-shaded) | 272 / 11.77 m | 189 / 10.44 m |
| Stress 3 (112.6°, photo) | 290 / 8.00 m | 253 / 5.81 m |

- **All six got a fix.** Every record and estimate carried its labels, and each summary began `STRESS CASE`.
- **The renders changed with the sun and the texture, not just the labels.** The same seeds gave different inlier counts and errors in each run. The direction matches the prediction: fewer inliers and larger errors at 112.6° on the de-shaded texture than at the anchor or on the photo. Two seeds on a thinned map can't say more than that.

## Results

Not run.
