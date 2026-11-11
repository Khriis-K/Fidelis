# Fidelis

**How often does a Mars lander's vision-based navigation fail, and how do you estimate that affordably and prove the estimate is right?**

Fidelis (Latin for "faithful") is a Gaussian-splat-based Mars descent simulator. It estimates the failure probability of a vision-based terrain-relative navigation (TRN) system at Jezero Crater, using multifidelity Monte Carlo to keep the estimate affordable, and validates it against a brute-force reference.

> **Status: in development.** Built and measured so far: the Jezero scene, a path-traced renderer, the navigator, a Gaussian splat stand-in for the renderer, and plain Monte Carlo on a local GPU and the USC CARC cluster. The latest run found **no failures in 300 simulated descents** ([`docs/benchmarks/failure-scan.md`](docs/benchmarks/failure-scan.md)). That's because the simulation is kinder than flight: its map and its descent images come from the same photo. Realistic error sources, starting with the real LVS map, are being added one at a time ([ADR 0003](docs/adr/0003-failure-is-40m-in-the-map-frame.md)). Not built yet: cheaper fidelity levels that work (map thinning didn't; [`docs/benchmarks/fidelity-sweep.md`](docs/benchmarks/fidelity-sweep.md)), multifidelity estimation, validation, and the viewer. Steps below describe the planned system unless they say otherwise. Every number lives in [`docs/benchmarks/`](docs/benchmarks/) with its caveats. Progress is tracked in [GitHub Issues](https://github.com/Khriis-K/Fidelis/issues) and [milestones](https://github.com/Khriis-K/Fidelis/milestones).

## The problem

During landing, a Mars lander can estimate its position by matching descent-camera images against an orbital map. This is terrain-relative navigation, and Perseverance used it at Jezero Crater in 2021. Before a landing, mission teams need to know how likely that matching is to fail.

You can't test-land on Mars, so reliability has to come from thousands of simulated descents. Realistic simulation is expensive, because every descent means rendering photoreal terrain and running a navigation pipeline on it. Cheap approximate simulations are fast, but they're biased, and the bias has to be corrected rigorously.

## The approach

1. **Build the scene.** Turn public USGS HiRISE elevation data and orthoimagery of Jezero into a textured 3D terrain mesh, rendered with physically correct shadows.
2. **Make it fast to render.** Train Gaussian splat scenes from the mesh renders, one per sun angle. The splat is a faster stand-in for the mesh renderer.
3. **Fly simulated descents.** Randomize the true position, altitude, camera attitude, image noise, blur, exposure, and sun angle. Every descent is reproducible from its random seed.
4. **Navigate.** ALIKED keypoints, LightGlue matching against the orbital map, and PnP with RANSAC give a position estimate. A descent fails if it gets no fix, or if its fix is more than *r* = 40 m off, Perseverance's LVS requirement, measured against the map as LVS was ([ADR 0003](docs/adr/0003-failure-is-40m-in-the-map-frame.md)). Every error source in the random scenario has to be one a real landing faces, with its size taken from a source.
5. **Estimate efficiently.** Fidelity levels are variants of the navigator on a shared splat render, since matching, not rendering, dominates a descent's cost ([ADR 0002](docs/adr/0002-navigator-fidelity-ladder.md)). Combine many cheap low-fidelity runs with a few expensive high-fidelity ones using NASA Langley's [MXMCPy](https://github.com/nasa/MXMCPy). Compare multifidelity Monte Carlo (MFMC), multilevel Monte Carlo (MLMC), and approximate control variates (ACV) against plain Monte Carlo at equal compute.
6. **Prove it.** Check every estimator against a brute-force high-fidelity reference over repeated independent trials, and compare MXMCPy's predicted variance with the observed spread.
7. **Show it.** An interactive web viewer lets visitors fly over the splat terrain, replay individual descents from the descent camera's point of view with the navigator's keypoint matches drawn live, and watch landing-estimate dispersion build up.

## Why the result should be trustworthy

- **Jezero is unseen.** The matcher is fine-tuned only on six other Mars regions (Gale, Holden, Eberswalde, Mawrth Vallis, Gusev, Elysium Planitia), with a geographic buffer around Jezero excluded. A test enforces the exclusion.
- **Validated, not just estimated.** Each estimator is judged against a brute-force reference across repeated trials, not a single run.
- **Honest costs.** Per-sample costs are measured on real hardware with the GPU type recorded, and every cost in one comparison comes from the same GPU type.
- **Reproducible.** The pipeline will rebuild from raw public data with one workflow command.

## Planned stack

| Area | Tools |
|---|---|
| Terrain and rendering | rasterio, Mitsuba 3 (CUDA) |
| Gaussian splatting | gsplat |
| Navigation | ALIKED, LightGlue, OpenCV PnP |
| Matcher fine-tuning | glue-factory |
| Uncertainty quantification | MXMCPy |
| Orchestration | Snakemake with the SLURM executor, local GPU or USC CARC |
| Viewer | TypeScript, React, Spark |
| Tooling | Python with uv and ruff |

## Roadmap

| Milestone | Delivers |
|---|---|
| M1: Smoke tests | Every core library runs on a local GPU and on the cluster |
| M2: End-to-end plain Monte Carlo | One command goes from a Jezero tile to a failure probability with a confidence interval, plus a match visualization for any descent |
| M3: Fidelity ladder and estimator comparison | Cheaper fidelity levels (navigator variants), a pilot study, and MFMC, MLMC, and ACV vs. plain Monte Carlo. The first showable result. |
| M4: Fine-tuning | Fine-tuned vs. pretrained matcher failure probability |
| M5: Lighting and validation | Multiple sun angles, brute-force reference, predicted vs. observed variance |
| M6: Viewer and write-up | Public replay viewer centered on the descent-camera view with live match overlay, highlight video, full write-up |

## Out of scope

Flight dynamics and guidance (the descent is scripted), hazard detection, live in-browser simulation, and validation against real descent-camera imagery.

## Data and credits

- **USGS Astrogeology:** Mars 2020 TRN HiRISE DTM mosaic ([doi:10.5066/P9REJ9JN](https://doi.org/10.5066/P9REJ9JN)) and HiRISE orthorectified image mosaic ([doi:10.5066/P9QJDP48](https://doi.org/10.5066/P9QJDP48)). Fergason, R. L., Hare, T. M., Mayer, D. P., Galuszka, D. M., Redding, B. L., Smith, E. D., Shinaman, J. R., Cheng, Y., Otero, R. E. (2020), "Mars 2020 Terrain Relative Navigation Flight Product Generation: Digital Terrain Model and Orthorectified Image Mosaics," 51st Lunar and Planetary Science Conference, abstract 2020.
- **HiRISE** (NASA/JPL-Caltech/University of Arizona): DTMs and imagery. McEwen, A. S., et al. (2007), "Mars Reconnaissance Orbiter's High Resolution Imaging Science Experiment (HiRISE)," *J. Geophys. Res.* 112, E05S02, [doi:10.1029/2005JE002605](https://doi.org/10.1029/2005JE002605). HiRISE products are public domain; credit requested.
- **CTX** (NASA/JPL-Caltech/Malin Space Science Systems): sets the matching-map resolution, via the USGS Mars 2020 TRN CTX orthomosaic, the LVS map ([doi:10.5066/P9GV1ND3](https://doi.org/10.5066/P9GV1ND3)). Malin, M. C., et al. (2007), "Context Camera Investigation on board the Mars Reconnaissance Orbiter," *J. Geophys. Res.* 112, E05S04, [doi:10.1029/2006JE002808](https://doi.org/10.1029/2006JE002808).
- **LVS reference map:** the 6 m/pixel matching-map resolution comes from Cheng, Y., Ansar, A., & Johnson, A. (2021), "Making an Onboard Reference Map From MRO/CTX Imagery for Mars 2020 Lander Vision System," *Earth and Space Science* 8, [doi:10.1029/2020EA001560](https://doi.org/10.1029/2020EA001560), Table 1.

Rebuild the Jezero scene from these public files with `uv run snakemake --profile profiles/local jezero`. That's about 8.7 GB of downloads. The source URLs, pinned checksums, and the facts behind the tile and map resolution are in [`config/jezero.yaml`](config/jezero.yaml).
- **MXMCPy:** Bomarito et al., "Multi Model Monte Carlo with Python (MXMCPy)," NASA/TM–2020–220585, NASA Langley Research Center.
- **LightGlue** (Lindenberger, Sarlin, Pollefeys, ICCV 2023), **ALIKED**, **gsplat**, **Mitsuba 3**, and **Spark**.

## License

A permissive license will be added before the first release. All planned dependencies are permissively licensed.
