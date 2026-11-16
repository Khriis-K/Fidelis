# Fidelis

**How often does a Mars lander's vision-based navigation fail, and how does lighting change that?**

Fidelis (Latin for "faithful") simulates the last stretch of a Mars descent over Jezero Crater, where Perseverance landed in 2021. It builds the terrain from public NASA/USGS data and renders what the descent camera would see. A learned image matcher then locates the lander against the same kind of orbital map Perseverance's Lander Vision System (LVS) used. Ten thousand randomized descents on a GPU cluster give a certified bound on how often that fails.

> **Status: complete** ([ADR 0006](docs/adr/0006-close-out.md)). Every number below comes from a committed write-up in [`docs/benchmarks/`](docs/benchmarks/), which also carries its caveats.

## Results

- **1 failure in 10,000 descents, so under 0.057% at 95% confidence** ([`certified-bound.md`](docs/benchmarks/certified-bound.md)). JPL's requirement was 1%. The one failure was an image 65% clipped to white, overexposed beyond anything LVS was tested on. With only its exposure lowered, the same descent locates. The bound holds under the sourced error ranges, two of which are labelled ranges rather than flight values.
- **A low sun, not a distant one, drives the error** ([`sun-degradation.md`](docs/benchmarks/sun-degradation.md)). Under the two 25° suns the median horizontal error is 6.35–6.84 m; under the 55° suns it's 3.77–4.18 m. At every sun the error is mostly a shift directly away from the sun, with 3.1–3.4 m of scatter around it. 10.6% of descents land more than 10 m off, and 3 in 10,000 more than 20 m.
- **Past JPL's 100° line, the error doubles but nothing fails** ([`stress-sun-scan.md`](docs/benchmarks/stress-sun-scan.md)). JPL's field-test misses all came with the sun more than 100° from the map's. Fidelis pushed it to 112.6°: the median error rose from 5.0 to 10.6 m, with no step at 100°, and 0 of 1,200 stress descents missed by more than 40 m. A diagnostic, not a failure rate.
- **Multifidelity Monte Carlo doesn't pay here** ([`sun-sweep.md`](docs/benchmarks/sun-sweep.md)). A free formula built from each descent's own inputs already cuts plain Monte Carlo's variance to about 0.2×. The cheapest navigator variant still costs 61% of the real one, so it can't beat the formula.

## How it works

1. **Scene.** A 10 km tile around the landing site from the USGS HiRISE elevation model and orthoimage, meshed at 4 m. A coarser 20 m CTX ring around it makes a scene about 26 km across, curved to Mars's radius ([`terrain-mesh.md`](docs/benchmarks/terrain-mesh.md), [`wide-scene.md`](docs/benchmarks/wide-scene.md)).
2. **Render.** Mitsuba 3 path-traces the descent camera's view from any pose under any sun. A Gaussian splat trained from mesh renders matches held-out views to at worst 32.6 dB PSNR ([`splat.md`](docs/benchmarks/splat.md)), but saved only 21% per descent, because matching, not rendering, dominates the cost ([`descent-profile.md`](docs/benchmarks/descent-profile.md)).
3. **Randomize.** Each descent draws position, altitude, tilt, heading, noise, blur, exposure, sun, dust haze, focal-length error and attitude error. Every range is taken from a JPL source ([`docs/research/`](docs/research/)), and each descent reruns from its seed.
4. **Navigate.** ALIKED keypoints are matched by LightGlue against the USGS CTX mosaic, the public stand-in for LVS's map. PnP with RANSAC then solves for position.
5. **Judge.** A descent fails if it gets no fix, or if its horizontal error is more than 40 m, LVS's requirement, measured in the map's frame as LVS was ([ADR 0003](docs/adr/0003-failure-is-40m-in-the-map-frame.md)).
6. **Bound.** Plain Monte Carlo with a Wilson interval gives the failure rate's 95% upper bound. Multifidelity estimation with NASA Langley's [MXMCPy](https://github.com/nasa/MXMCPy) was tested on mean error as a go/no-go and came back no ([ADR 0004](docs/adr/0004-reframe-when-failures-dont-happen.md)).

Every experiment was planned in writing before it ran, with a prediction from the sources. Each write-up then compares the result against that prediction, including the predictions that missed.

## What was dropped, and why

- **Fine-tuning the matcher.** With 1 failure in 10,000, there was no failure rate it could measurably improve. The lighting shift it would target stays well inside the 40 m line.
- **Repeated multifidelity estimator trials.** The go/no-go said no, so there was nothing to validate.
- **The web viewer.** Dropped to finish the project at a sensible size.

Full reasoning is in [ADR 0006](docs/adr/0006-close-out.md).

## Limitations

The simulator can't produce real camera images, cast shadows from features under 4 m, sky light, or suns more than 113° from the map's. The renders are a relit HiRISE photo, and the map is the public CTX product, not JPL's onboard one. Fidelis judges one image, where LVS had about 10 seconds and several images to converge. The full list is in [`docs/EXPLAINER.md`](docs/EXPLAINER.md#honest-limitations).

## Reproducing

Requires a CUDA GPU. The full CTX map needs more GPU memory than a laptop has; the headline runs used one NVIDIA A40 (48 GB).

```bash
uv sync
uv run snakemake --profile profiles/local jezero jezero_deshaded_texture
```

That downloads about 8.7 GB of public data, checks it against pinned checksums ([`config/jezero.yaml`](config/jezero.yaml)), and builds the scene. The descent runs are one command per batch:

```bash
uv run fidelis-plain-mc config/jezero.yaml data/jezero results/certified_bound/batch0 --fidelity mesh --first-seed 80000 --samples 2500
```

Batches 1–3 start at seeds 82,500, 85,000 and 87,500. The stress runs are listed in [`stress-sun-scan.md`](docs/benchmarks/stress-sun-scan.md). Then rebuild the sun report from the saved records:

```bash
uv run snakemake sun_degradation
```

Cluster job scripts for USC CARC (SLURM) are in each write-up. Rendering on a GPU jitters by about a millionth of a pixel, so a rerun reproduces each descent's image but can move its error by up to about 0.8 m ([`plain-mc.md`](docs/benchmarks/plain-mc.md)).

## Stack

| Area | Tools |
|---|---|
| Terrain and rendering | rasterio, Mitsuba 3 (CUDA) |
| Gaussian splatting | gsplat |
| Navigation | ALIKED, LightGlue, OpenCV |
| Uncertainty quantification | MXMCPy, NumPy, SciPy |
| Orchestration | Snakemake with the SLURM executor, local GPU or USC CARC |
| Tooling | Python with uv, pytest, ruff |

## Related work

- **Mars 2020 LVS.** Johnson et al., "The Lander Vision System for Mars 2020 Entry Descent and Landing," AAS 17-038 (2017); Johnson et al., "The Mars 2020 Lander Vision System Field Test," AAS 20-105 (2020); Johnson et al., "Mars 2020 Lander Vision System Flight Performance," AIAA SciTech 2022, [doi:10.2514/6.2022-1214](https://doi.org/10.2514/6.2022-1214); Mohan et al., "Assessment of M2020 Terrain Relative Landing Accuracy: Flight Performance vs. Predicts," AAS 21-541 (2021); Aaron et al., "Camera Simulation for the Perseverance Rover's Lander Vision System," AIAA SciTech 2022, [doi:10.2514/6.2022-0746](https://doi.org/10.2514/6.2022-0746). JPL verified LVS with field tests and high-fidelity Monte Carlo. Fidelis does the same at small scale, with a learned matcher, and without IMU fusion or flight dynamics.
- **The LVS map.** Cheng, Ansar & Johnson, "Making an Onboard Reference Map From MRO/CTX Imagery for Mars 2020 Lander Vision System," *Earth and Space Science* 8 (2021), [doi:10.1029/2020EA001560](https://doi.org/10.1029/2020EA001560).
- **Multifidelity Monte Carlo.** Peherstorfer, Willcox & Gunzburger on MFMC and optimal model management; Gorodetsky et al. on approximate control variates; Bomarito et al., "Multi Model Monte Carlo with Python (MXMCPy)," NASA/TM–2020–220585. Elfverson, Hellman & Målqvist, "A multilevel Monte Carlo method for computing failure probabilities," *SIAM/ASA JUQ* (2016), on why failure indicators are hard for multilevel methods.
- **Feature matching over pose regression.** Sattler et al., "Understanding the Limitations of CNN-based Absolute Camera Pose Regression," CVPR 2019. Matching landmarks also mirrors the real LVS.

Fidelis *applies* certified Monte Carlo bounds and a multifidelity go/no-go to simulated, vision-based Mars descent navigation. It doesn't claim to be the first to do so.

## Data and credits

- **USGS Astrogeology:** Mars 2020 TRN HiRISE DTM mosaic ([doi:10.5066/P9REJ9JN](https://doi.org/10.5066/P9REJ9JN)), HiRISE orthorectified image mosaic ([doi:10.5066/P9QJDP48](https://doi.org/10.5066/P9QJDP48)), and CTX orthomosaic ([doi:10.5066/P9GV1ND3](https://doi.org/10.5066/P9GV1ND3)). Fergason et al. (2020), "Mars 2020 Terrain Relative Navigation Flight Product Generation: Digital Terrain Model and Orthorectified Image Mosaics," 51st LPSC, abstract 2020.
- **HiRISE** (NASA/JPL-Caltech/University of Arizona). McEwen et al. (2007), *J. Geophys. Res.* 112, E05S02, [doi:10.1029/2005JE002605](https://doi.org/10.1029/2005JE002605). Public domain; credit requested.
- **CTX** (NASA/JPL-Caltech/Malin Space Science Systems). Malin et al. (2007), *J. Geophys. Res.* 112, E05S04, [doi:10.1029/2006JE002808](https://doi.org/10.1029/2006JE002808).
- **Software:** LightGlue (Lindenberger, Sarlin & Pollefeys, ICCV 2023; Apache-2.0), ALIKED (Zhao et al.; BSD-3-Clause), gsplat (Apache-2.0), Mitsuba 3 (BSD-3-Clause), MXMCPy (Apache-2.0), PyTorch, OpenCV, rasterio.

No SuperPoint code or weights are used: LightGlue runs with its ALIKED weights.

## License

[MIT](LICENSE). Every runtime dependency is permissively licensed. The public data keeps its own terms (see Data and credits).
