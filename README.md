# Fidelis

**How often does a Mars lander's vision-based navigation fail, and how do you estimate that affordably and prove the estimate is right?**

Fidelis (Latin for "faithful") is a Gaussian-splat-based Mars descent simulator. It estimates the failure probability of a vision-based terrain-relative navigation (TRN) system at Jezero Crater, using multifidelity Monte Carlo to keep the estimate affordable, and validates it against a brute-force reference.

> **Status: early development.** Nothing below is built or measured yet. This README describes the planned system. There are no results to report, and numbers will appear here only once they're measured. Progress is tracked in [GitHub Issues](https://github.com/Khriis-K/Fidelis/issues) and [milestones](https://github.com/Khriis-K/Fidelis/milestones).

## The problem

During landing, a Mars lander can estimate its position by matching descent-camera images against an orbital map. This is terrain-relative navigation, and Perseverance used it at Jezero Crater in 2021. Before a landing, mission teams need to know how likely that matching is to fail.

You can't test-land on Mars, so reliability has to come from thousands of simulated descents. Realistic simulation is expensive, because every descent means rendering photoreal terrain and running a navigation pipeline on it. Cheap approximate simulations are fast, but they're biased, and the bias has to be corrected rigorously.

## The approach

1. **Build the scene.** Turn public USGS HiRISE elevation data and orthoimagery of Jezero into a textured 3D terrain mesh, rendered with physically correct shadows.
2. **Make it fast to render.** Train Gaussian splat scenes from the mesh renders, one per sun angle. Derive three fidelity levels from each by varying Gaussian count, render resolution, and spherical-harmonic degree.
3. **Fly simulated descents.** Randomize the true position, altitude, camera attitude, image noise, blur, exposure, and sun angle. Every descent is reproducible from its random seed.
4. **Navigate.** ALIKED keypoints, LightGlue matching against the orbital map, and PnP with RANSAC give a position estimate. Failure means the navigation error at handoff exceeds a threshold *r*.
5. **Estimate efficiently.** Combine many cheap low-fidelity runs with a few expensive high-fidelity ones using NASA Langley's [MXMCPy](https://github.com/nasa/MXMCPy). Compare multifidelity Monte Carlo (MFMC), multilevel Monte Carlo (MLMC), and approximate control variates (ACV) against plain Monte Carlo at equal compute.
6. **Prove it.** Check every estimator against a brute-force high-fidelity reference over repeated independent trials, and compare MXMCPy's predicted variance with the observed spread.
7. **Show it.** An interactive web viewer lets visitors fly over the splat terrain, replay individual descents, and watch landing-estimate dispersion build up.

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
| M2: End-to-end plain Monte Carlo | One command goes from a Jezero tile to a failure probability with a confidence interval |
| M3: Fidelity ladder and estimator comparison | Three fidelity levels, a pilot study, and MFMC, MLMC, and ACV vs. plain Monte Carlo. The first showable result. |
| M4: Fine-tuning | Fine-tuned vs. pretrained matcher failure probability |
| M5: Lighting and validation | Multiple sun angles, brute-force reference, predicted vs. observed variance |
| M6: Viewer and write-up | Public replay viewer, highlight video, full write-up |

## Out of scope

Flight dynamics and guidance (the descent is scripted), hazard detection, live in-browser simulation, and validation against real descent-camera imagery.

## Data and credits

- **USGS Astrogeology:** Mars 2020 TRN HiRISE DTM and orthoimage mosaics.
- **HiRISE** (NASA/JPL-Caltech/University of Arizona): DTMs and imagery. HiRISE products are public domain; credit requested.
- **MXMCPy:** Bomarito et al., "Multi Model Monte Carlo with Python (MXMCPy)," NASA/TM–2020–220585, NASA Langley Research Center.
- **LightGlue** (Lindenberger, Sarlin, Pollefeys, ICCV 2023), **ALIKED**, **gsplat**, **Mitsuba 3**, and **Spark**.

## License

A permissive license will be added before the first release. All planned dependencies are permissively licensed.
