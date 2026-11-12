# CTX map scan: what matching against the real LVS map does (issue #36)

Sourced error 1 of 4 (ADR 0003, decision 6). Until now the navigator matched against a map made from the same orthoimage the scene renders, so map and descent image agreed on every rock and shadow, and the failure scan found 0 failures in 300 (`docs/benchmarks/failure-scan.md`). From #36 the navigator matches against the real LVS map: the USGS CTX orthomosaic. This file fixes what will be measured **before** the run. Results go under the plan afterwards, and any departure from the plan is stated there with its reason.

## What changed in the navigator's map

- **The image is CTX, and only CTX.** `fidelis-lvs-map` (`src/fidelis/terrain/lvs_map.py`) copies the raw CTX mosaic's own pixels under the scene's footprint. It isn't resampled onto the scene's pixel grid, and it isn't brightness-matched to HiRISE. The scene's ring texture is CTX scaled to HiRISE's mean and spread, so that isn't used either. The builder is given only the CTX file, the footprint and the east–west scale, and a test builds the map with every HiRISE-derived raster deleted.
- **Built on the laptop, 2026-11-11 (CPU, no GPU time):** 4,340 × 4,350 pixels at 6 m, 0 nodata pixels, over map x 4,577,803–4,603,903 m and y 1,080,293–1,106,333 m.
- **The old map stays.** `matching_map.source` in `config/jezero.yaml` picks `ctx` (now the default) or `scene`. `fidelis-plain-mc --matching-map scene` reruns on the old map, with its zero offset and the scene's elevation. Every record now names its map.

### Elevation: the CTX DTM (decided here)

**The LVS map's keypoints get their height from the CTX DTM everywhere, not from the scene's HiRISE DTM.**

- **Why.** LVS's onboard map was an appearance map plus an elevation map, both made from CTX (`docs/research/lvs-error-sources.md`, section 2: 6 m appearance, 12 m elevation "from DEMs of ≤ 20 m/pixel", co-registered within 6 m; Cheng et al. 2021 Table 1, read through the research note, since the paper itself isn't in the repo). The CTX ortho was also projected onto the CTX DTM, so the two belong together. Giving the navigator HiRISE heights would hand it a better elevation map than LVS had, made from the same data as the truth. With CTX heights the navigator holds nothing HiRISE-derived.
- **What it costs.** This step now adds two real map errors at once: appearance and elevation. Over the tile, CTX minus HiRISE heights have a median of +0.32 m and an sd of 3.11 m (`docs/benchmarks/wide-scene.md`). The scan can't separate the two effects. The control below, and `--matching-map scene`, only tell the old map from the new one.
- **Not the flight product.** USGS's CTX DTM is 20 m/pixel. LVS's elevation map was required at 12 m/pixel, and JPL built its own. So this is the closest public product, not the onboard map.
- **Truth is unchanged.** The true camera is still placed above the scene's own terrain. Before this change the truth read the ground height from the navigator's map, which was harmless only while the two were the same terrain.

### The offset that is removed

`map_offset` is now live: **+1.89 m east, −1.49 m north** in ground metres (2.41 m), from `docs/benchmarks/map-offset.md` (#40). It is subtracted from every fix's error before failures are counted, and every record carries it.

**Caveat, known before the run.** That offset was measured over the HiRISE tile, and it is exact only there. In the ring around the tile the scene renders CTX itself, so map and scene agree and the true offset is zero. A view that reaches the ring mixes the two, and inside the tile's 200 m blend band the texture eases from HiRISE to CTX. Removing 2.41 m from a fix that leaned on ring features over-corrects by up to that much. It is 6% of r, and it stays in the error like any other local map distortion.

## Plan

**Status: pre-registered 2026-11-11, before any descent was run against the CTX map.** Nothing in this section is a measurement.

### What runs

- **Scan.** The real navigator on the mesh renderer (`--fidelity mesh`), matching against the CTX map (`matching_map.source: ctx`).
- **Seeds.** 20,000–20,299 (300 descents). Unused so far: 0–299 and 10,000–10,299 are taken.
- **Scenario.** Drawn from `config/jezero.yaml` as is. `--sun-index` is left off, so each seed draws one of the five suns. The navigator still gets the true attitude.
- **Hardware.** One debug job on one NVIDIA A40 (`docs/carc.md`), as the failure scan.
- **Command.**

      srun --account=irahbari_1147 --partition=debug --gres=gpu:a40:1 -c 8 --mem 32G -t 30 bash -c "module purge && module load ver/2607 gcc/15.3.0 cuda/13.0.3 && uv run --no-sync fidelis-plain-mc config/jezero.yaml data/jezero results/ctx_map_scan --fidelity mesh --first-seed 20000 --samples 300"

- **Control (recommended, the user's call).** The same 300 seeds on the old map, same GPU type:

      srun --account=irahbari_1147 --partition=debug --gres=gpu:a40:1 -c 8 --mem 32G -t 30 bash -c "module purge && module load ver/2607 gcc/15.3.0 cuda/13.0.3 && uv run --no-sync fidelis-plain-mc config/jezero.yaml data/jezero results/ctx_map_scan_control --fidelity mesh --first-seed 20000 --samples 300 --matching-map scene"

  The renders are the same seed for seed, so any difference is the map. It is also the null case for this change: on the old map the new code should look like the failure scan (0 failures, per-sun medians of 1.6–4.8 m). If it doesn't, the refactor broke something and the scan isn't trusted. If the control isn't run, the results say so.
- **Rough cost (an estimate, not a measurement).** The failure scan's mesh descents took a median 0.479 s on an A40, so about 2.5 minutes of descents plus loading per run. Each job reserves 1 A40, 8 CPUs and 32 GB, which is 24 SUs per minute (`docs/carc.md`): at most 720 SUs each at the 30-minute limit, and roughly 100–150 SUs each if it takes about 5 minutes.

### What counts as a failure (fixed by ADR 0003, not chosen here)

- **r = 40 m in the map frame.** A descent fails if it gets no fix, or its fix is more than 40 m from the truth after the offset above is removed.
- **No-fix and beyond-r failures are counted separately.**
- r is not tuned, whatever the count turns out to be.

### What gets reported

1. **The headline:** failures out of 300, no-fix and beyond-r separately, with the 95% Wilson interval `fidelis-plain-mc` prints, and the offset used.
2. **By sun:** failures, and the map-frame error's median, 90th percentile and maximum, for each of the five suns.
3. **Against the control, if run:** the same table on the old map, and the per-seed change in error.
4. **Exploratory, labelled as such:**
   - failures and errors at 12 m (the old provisional line);
   - horizontal and vertical error separately, since CTX heights can add a height bias;
   - the mean true-frame error vector, next to the offset. This is a check only: the offset is never set from it;
   - errors against tilt, blur, altitude and distance from the tile centre (views that reach the ring);
   - match and inlier counts, and the map's keypoint count.

### What each outcome means

- **0 failures.** The real LVS map alone doesn't push fixes past 40 m. The 95% upper bound on P(fail) is then about 1.3%. Report how much the errors grew. #37 (dust haze) is next either way.
- **A few failures.** Report which kind and where they cluster (by sun first). Nothing is tuned in response.
- **Many failures, or most descents failing.** Check the control first. If the control is clean, the map is the cause, and the honest reading is **not** "LVS would fail this often". Fidelis's descent images are HiRISE textures relit by a simple sun, with HiRISE's own shadows baked in, compared with a CTX photo taken on other days under other light. That gap may be wider than the one between a real LCAM image and the CTX map. JPL kept the map-to-image sun difference under 20° for Mars 2020, and its only field misses came above 100° (`docs/research/lvs-error-sources.md`). Fidelis's suns reach azimuth 310° against CTX images taken at 222.9–295.1°, and neither orthoimage's sun angle is recorded in the repo. So a high rate would be reported with that caveat, and the render-versus-reality gap would become the next thing to size, before any more errors are added.
- **No-fix failures.** Reported separately. They are a lost match, a different failure from an inaccurate fix.

### Known before the run

- **The map gives fewer keypoints than the old one.** ALIKED finds 58,860 on the CTX map on the CPU and 58,827 on the laptop's RTX 5070, against 70,749 and 70,726 on the old map (measured 2026-11-11, extraction only, no matching). The A40's count for the CTX map isn't known yet. It gave the CPU's count on the old map.
- **No descent has been matched against the CTX map yet**, on any machine. The laptop can't hold the full map's matching (`docs/benchmarks/plain-mc.md`). A handful of smoke descents on a thinned map may be run on the laptop after this plan is committed, on seeds outside the scan's, to check the plumbing. They are not results and are reported as a smoke test only.

## A loose end closed: 70,749 or 70,726 map keypoints

`failure-scan.md` and `fidelity-sweep.md` record 70,749 keypoints on the old map, against 70,726 in `descent-profile.md`, `plain-mc.md` and `docs/carc.md`. **It is the GPU model, not the map or the code.** The same map file, with today's code, gives 70,726 on the laptop's RTX 5070 and 70,749 on the laptop's CPU, the A40's number (measured 2026-11-11). The 70,726 was measured on the laptop on 2026-11-02 and later quoted for the cluster runs. The A40 records are the ones that count 70,749.

Where the difference comes from: the map is extracted in 25 tiles, each capped at 4,096 keypoints. Ten tiles hit the cap on both devices. The other fifteen keep however many points pass ALIKED's detector, and twelve of them differ between the two devices by 1 to 11 keypoints (some up, some down, 23 in all). That fits last-digit differences in the network's arithmetic tipping borderline points over the detection threshold, in line with ADR 0001 (GPU results repeat at image level, not bit for bit). The exact mechanism inside ALIKED wasn't traced. Whether the cluster's map file is byte-identical to the laptop's also wasn't checked. It doesn't need to be for this explanation, since one file gives both numbers.

What it means: keypoint counts, and so fixes, differ slightly between GPU models. That's one more reason costs and results are pinned to one GPU type.

## Results

Not run yet.
