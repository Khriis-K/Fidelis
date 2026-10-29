# Fidelis — Spec

Fidelis (Latin for "faithful") is a Gaussian-splat-based Mars descent simulator that estimates how often a vision-based terrain-relative navigation (TRN) system fails, using multifidelity Monte Carlo to make that estimate affordable, and proves the estimate is right against a brute-force reference.

## Problem Statement

Before a Mars landing, mission teams need to know how likely their vision-based navigation is to fail. Terrain-relative navigation estimates the lander's position by matching descent-camera images against an orbital map; Perseverance used this approach at Jezero Crater. You cannot test-land on Mars, so reliability has to be estimated from thousands of simulated descents. Realistic simulation is expensive: every simulated descent requires rendering photoreal terrain and running a navigation pipeline on it. Plain Monte Carlo at the required sample sizes is slow and costly, and cheap approximate simulations are only useful if their bias can be corrected rigorously.

From the developer's perspective, there is also a portfolio problem: demonstrating, in one coherent project, real competence in 3D scene reconstruction (Gaussian splatting), deep learning (learned feature matching), and uncertainty quantification (multifidelity UQ, reliability analysis) on a problem that NASA, JPL, and aerospace/defense teams recognize as their own, using public NASA data and NASA's own UQ tooling.

## Solution

Fidelis builds a photoreal, renderable model of Jezero Crater from public USGS HiRISE terrain data, simulates many scripted descents under randomized operating conditions (true position and altitude, camera attitude, image quality, sun angle), runs a learned feature-matching navigator on each rendered descent image, and records the navigation error at handoff.

Failure is defined as navigation error at handoff exceeding a threshold r. Fidelis estimates a smoothed failure probability (plus mean and variance of navigation error) using three Gaussian-splat render fidelity levels and NASA Langley's MXMCPy library, comparing multifidelity Monte Carlo (MFMC), multilevel Monte Carlo (MLMC), and approximate control variates (ACV) against plain Monte Carlo at equal compute. Estimates are validated against a brute-force reference value with repeated independent trials, and predicted variance is compared with observed variance.

The navigator is fine-tuned only on terrain from six other Mars regions, so Jezero is a genuinely unseen test site. Results are presented in an interactive web replay where visitors fly over the splat terrain, replay individual simulated descents, and watch landing-estimate dispersion ellipses build up, plus a short highlight video.

## User Stories

1. As the developer, I want a single smoke-test workflow that confirms every core library runs on my local GPU and on the cluster, so that environment problems surface before I build on them.
2. As the developer, I want to download the USGS Mars 2020 TRN HiRISE DTM and orthoimage mosaics for Jezero reproducibly, so that anyone can rebuild the test scene from public data.
3. As the developer, I want to convert a DTM and orthoimage into a textured terrain mesh, so that I have a ground-truth reference scene that can be rendered from any viewpoint.
4. As the developer, I want to render the terrain mesh from many viewpoints under a chosen sun angle with physically accurate shadows, so that splat training and matcher training have realistic, correctly lit images.
5. As the developer, I want shadows already baked into the orthoimage to be reduced before relighting, so that relit terrain does not show double shadows.
6. As the developer, I want to train one Gaussian splat scene per sun angle from mesh renders, so that sun angle can vary across simulated descents even though splats bake in lighting.
7. As the developer, I want to derive three fidelity levels from each splat scene by varying Gaussian count, render resolution, and spherical-harmonic degree, so that I have cheap approximations that track the expensive render.
8. As the developer, I want the cost of each fidelity level measured on real hardware and recorded with the GPU type, so that estimator sample allocation uses real costs rather than guesses.
9. As the developer, I want a scripted descent trajectory over Jezero, so that the simulation isolates navigation performance without needing a flight-dynamics simulator.
10. As the developer, I want each simulated descent to draw random values for true position, altitude, attitude, image noise, blur, exposure, and sun angle, so that the failure estimate reflects variation in operating conditions.
11. As the developer, I want every simulated descent to be reproducible from its random seed, so that any individual result can be regenerated and inspected.
12. As the developer, I want a navigator that detects ALIKED keypoints, matches them to the orbital map with LightGlue, and solves position with PnP, so that the system under test mirrors how landmark-matching TRN works.
13. As the developer, I want the navigator to report diagnostics such as match count and inlier count alongside its position estimate, so that failures can be explained rather than just counted.
14. As the developer, I want to render matcher training data from meshes of six held-out training regions with known camera poses, so that training labels (ground-truth correspondences) come for free.
15. As the developer, I want Jezero and its surroundings fully excluded from matcher training, so that the failure estimate is not inflated by data leakage.
16. As the developer, I want to fine-tune LightGlue with ALIKED features on the training-region renders, so that the matcher learns Mars terrain in general.
17. As the developer, I want the pretrained, non-fine-tuned matcher evaluated as a baseline, so that I can show how much fine-tuning changed failure probability.
18. As the developer, I want a single simulation interface that takes random inputs and a fidelity level and returns a navigation error, so that the UQ layer and the tests use the same boundary.
19. As the developer, I want a pilot study that runs a modest number of samples at all fidelity levels, so that I can measure costs, correlations, and typical error magnitudes.
20. As the developer, I want the failure threshold r chosen from the pilot study so failure occurs roughly 2–10% of the time, so that a brute-force reference is affordable and the estimator comparison is meaningful.
21. As the developer, I want failure measured with a smoothed (sigmoid) indicator around r, so that correlation between fidelity levels stays high and multifidelity estimators deliver real savings.
22. As the developer, I want to report the bias introduced by smoothing as a function of sigmoid steepness, so that the bias-variance tradeoff is explicit.
23. As the developer, I want mean and variance of navigation error estimated alongside failure probability, so that I have well-behaved supporting results.
24. As the developer, I want MXMCPy to compute optimal sample allocations and estimates for MFMC, MLMC, and ACV, so that I can compare estimators fairly at equal compute.
25. As the developer, I want plain Monte Carlo run at the same compute budget as each multifidelity estimator, so that speedups are measured against an honest baseline.
26. As the developer, I want a brute-force reference failure probability from a large number of high-fidelity samples, so that every estimator can be judged against a known target.
27. As the developer, I want each estimator repeated independently as many times as fit within a stated GPU-hour budget, so that I can measure actual estimator spread rather than relying on a single run.
28. As the developer, I want MXMCPy's predicted variances compared with the empirically observed variances, so that I can report whether the optimizer's promises hold for splat-based fidelity levels.
29. As the developer, I want failure probability reported as a function of sun angle, so that the effect of lighting mismatch between orbital map and descent image is visible.
30. As the developer, I want failure probability reported across a range of r values, with the Mars 2020 Lander Vision System's 40 m position-knowledge target marked as a reference line, so that readers see how reliability changes as the requirement tightens and where a real mission requirement falls.
30a. As the developer, I want the orbital map the navigator matches against to be built at CTX-like resolution (downsampled from the HiRISE orthoimage), while descent renders stay HiRISE-derived, so that matching mirrors the real LVS, whose reference map came from the coarser CTX camera.
31. As the developer, I want the whole pipeline orchestrated by a workflow manager that reruns only missing or failed steps, so that a cluster failure halfway through does not force a restart.
32. As the developer, I want the same workflow to run locally on my GPU or on the cluster by switching a profile, so that I can develop small and scale up without rewriting.
33. As the developer, I want Monte Carlo samples batched into cluster jobs sized from measured timing, so that scheduler overhead and queue limits do not dominate.
34. As the developer, I want all pipeline outputs needed by the viewer written in a documented, versioned results format, so that the Python side and the web side can evolve independently.
35. As a visitor, I want to fly around the Jezero terrain in my browser, so that I can see the scene the simulation runs in.
36. As a visitor on a laptop or phone, I want the terrain to stream at a level of detail my device can handle, so that the viewer stays smooth.
37. As a visitor, I want to pick an individual simulated descent and watch it replay with the true and estimated positions shown, so that I understand what the navigator does.
38. As a visitor, I want to see landing-estimate dispersion ellipses build up over the terrain, so that I can see how navigation error spreads.
39. As a visitor, I want to switch between precomputed scenario presets such as different sun angles and noise levels, so that I can explore how conditions affect failure without waiting for live simulation.
40. As a visitor, I want headline numbers (failure probability with confidence bounds, estimator speedups) shown in the viewer, so that I get the result without reading the full write-up.
41. As a recruiter or reviewer who will not click through, I want a short highlight video, so that I can see the project in under a minute.
42. As a NASA or JPL reviewer, I want the README to state the failure definition, threshold rule, random inputs, fidelity levels, and validation method precisely, so that I can judge whether the results are trustworthy.
43. As a NASA or JPL reviewer, I want the data sources and NASA tools credited and cited, so that provenance is clear.
44. As a reviewer, I want to see that training regions were held out geographically from the test site, so that I can trust the failure estimate is not leaked.
45. As a reviewer, I want every published component to carry a permissive license, so that the code can be reused without licensing surprises.
46. As a future contributor, I want to reproduce the full pipeline from raw public data with one workflow command, so that I can verify or extend the results.
47. As the developer, I want fast tests that run without a GPU, so that I can check the statistics code and results format on any machine.
48. As the developer, I want stretch goals tracked separately from the core milestones, so that the core result ships before optional work begins.

## Implementation Decisions

**Framing**

- The project is a reliability study. The headline contribution is an efficiently estimated, validated failure probability. The navigator only needs to be reasonable; it is the system under test, not the object of innovation.
- The quantity of interest is navigation position error at the handoff point of a scripted descent. Failure means that error exceeds r. There are no flight dynamics, guidance, or hazard avoidance.
- The headline statistic is a smoothed failure probability using a sigmoid ramp around r, reported with its smoothing bias. Mean and variance of navigation error are supporting statistics.
- r is set from the pilot study so failure occurs roughly 2–10% of the time. Results are also reported across a range of r.

**Data**

- The test site is Jezero Crater, built from the USGS Mars 2020 TRN HiRISE DTM and orthoimage mosaics. The USGS Mars Sample Return TRN mosaic is an alternative if wider coverage is needed. HiRISE products are public domain; credit is requested.
- Training regions are six official HiRISE stereo DTMs from former or actual landing sites: Gale Crater, Holden Crater, Eberswalde Crater, Mawrth Vallis, Gusev Crater, and Elysium Planitia. Exact DTM product IDs are confirmed at download time.
- Only official stereo-derived DTMs are used, so DTM production method is consistent between training and test. Shape-from-shading DTMs (such as some Oxia Planum products) are excluded.
- A geographic buffer around Jezero is excluded from training, including Nili Fossae, Northeast Syrtis, and the wider MSR mosaic area.
- The orbital map used for matching is the HiRISE orthoimage downsampled to CTX-like resolution, mirroring the real Mars 2020 LVS reference map, which was built from MRO Context Camera (CTX) imagery. Descent renders remain HiRISE-derived, so the map is coarser than the descent views, as in flight. The same downsampling applies to training-region maps used for fine-tuning.

**Terrain ingestion module**

- Reads GeoTIFF DTMs and orthoimages with rasterio (GDAL-based) and produces textured terrain meshes.
- Applies a shadow-reduction step to orthoimage texture before relighting to mitigate double shadows.

**Rendering module**

- Mitsuba 3 (CUDA variant) renders meshes with path tracing so sun shadows are physically correct. BlenderProc is the fallback if Mitsuba proves unworkable.
- Used for two purposes: multi-view renders for splat training (Jezero) and descent-view renders with known camera poses for matcher training (training regions).
- Sun angle is a render parameter. Jezero is rendered under 4–6 sun angles.

**Splat scene module**

- gsplat trains one splat scene per Jezero sun angle from mesh renders. Sun angle is therefore a discrete random input.
- Three fidelity levels are derived per scene by varying Gaussian count (pruning), render resolution, and spherical-harmonic degree. All levels share one navigator; low-resolution renders are resized to the navigator's input size.
- If profiling shows the navigator dominates per-sample cost, the fidelity ladder may additionally shrink the navigator. This is a fallback, not the default.
- Splat scenes are exported in a format the browser renderer reads (PLY or a compressed equivalent).

**Scenario sampling module**

- Draws per-descent random inputs: true position offset, altitude, attitude, image noise, motion blur, exposure, and sun-angle index. Every draw is derived from a recorded seed.
- The descent trajectory is scripted. A pendulum-swing attitude model is a v1.1 enhancement, replacing independent random tilts.

**Navigation module**

- Detects ALIKED keypoints and matches descent images to the orbital map with LightGlue, then solves camera position with OpenCV PnP and RANSAC.
- Feature matching is chosen over direct pose regression because pose regression behaves more like image retrieval than structure-based localization and does not consistently beat a retrieval baseline (Sattler et al., CVPR 2019). Landmark matching also mirrors the real LVS.
- Returns a position estimate plus diagnostics: match count, inlier count, and solve status. A failed solve is a recorded outcome, not a crash.
- SuperPoint is not used in published code or weights because its license restricts distributing derivatives. It may be used privately as an unpublished comparison.

**Matcher training module**

- Fine-tunes LightGlue with ALIKED features on mesh renders from the six training regions, using ground-truth correspondences available from known render poses. glue-factory, LightGlue's companion training library, is the expected tool.
- The pretrained matcher is evaluated as a baseline.

**Simulation interface (primary seam)**

- One boundary: given a scenario sample (random inputs plus seed) and a fidelity level, return a record containing navigation error vector and magnitude, navigator diagnostics, wall-clock cost, and GPU type.
- This is the model interface the UQ module consumes and the boundary the tests exercise.

**UQ module**

- MXMCPy computes sample allocations, estimates, and predicted variances for MFMC, MLMC, and ACV from pilot-study costs and covariances. MFMC is the headline method in the README; the three-way comparison is the deeper result.
- Plain Monte Carlo is run at equal compute as the baseline.
- Validation uses a brute-force high-fidelity reference value and repeated independent trials per estimator. The number of repeats is set by a stated GPU-hour budget after the pilot study, not fixed in advance.
- Reports predicted versus observed estimator variance, failure probability versus sun angle, failure probability versus r (with the LVS 40 m target marked for reference; r itself is still set from the pilot study), smoothing bias versus steepness, and fine-tuned versus pretrained failure probability.
- Selective refinement (Elfverson, Hellman & Målqvist) is a planned comparison to sigmoid smoothing: run the high-fidelity level only for samples whose cheaper-level error lands near r. It is scheduled as a stretch item (see M7), not part of the core milestones.

**Orchestration**

- Snakemake with the SLURM executor plugin. The same workflow runs locally or on USC CARC by switching profiles. Per-step GPU resources are declared in the workflow.
- Monte Carlo samples are batched into jobs, with batch size set from measured timing.
- Hardware use: local RTX 5070 for development and the demo; CARC A100 and L40S for splat training, fine-tuning, and cost measurement; any available CARC GPU for Monte Carlo batches. All costs fed to MXMCPy for one comparison must come from the same GPU type.

**Results format (boundary between Python and the viewer)**

- A documented, versioned format containing descent trajectories, per-run records (true and estimated positions, errors, diagnostics), aggregate statistics with confidence bounds, scenario-preset metadata, and references to splat assets.
- The Python side only writes it; the viewer only reads it.

**Viewer**

- TypeScript and React, rendering splats with Spark (World Labs, MIT license, three.js-based, with streamed level of detail). React integration via react-three-fiber, pending verification of compatibility with Spark.
- Replays precomputed results only; there is no live GPU backend. Scenario presets (sun angle by noise level) give interactive exploration without live simulation.
- A short highlight video is cut from the viewer.

**Languages and tooling**

- Python for all computation, managed with uv and linted with ruff. TypeScript only for the viewer. Roughly 80% Python, 20% TypeScript.
- C++ is not used in the core. A flight-style C++ implementation of the navigation step, cross-verified against the Python version on identical inputs, is a stretch goal.

**Licensing**

- All published components use permissive licenses: gsplat (Apache-2.0), LightGlue (Apache-2.0), ALIKED (BSD-3-Clause), Mitsuba 3 / Dr.Jit (BSD-3-Clause), Spark (MIT). MXMCPy and HiRISE data are NASA/public.
- BlenderProc is GPL-3.0. If the fallback is used, its effect on the project license must be reviewed before publishing.

## Testing Decisions

**What makes a good test here:** tests check external behavior at a seam, not internal implementation. Rendering, splat training, and matching are tested through the simulation seam rather than individually, so their internals can change without rewriting tests.

**Seam 1: Simulation interface (primary)**

- The same scenario sample and seed produce the same navigation error.
- A perfect-conditions scenario (no image noise or blur, nadir camera, sun angle matching the orbital map) produces navigation error below a small tolerance.
- Measured per-sample cost increases with fidelity level.
- Outputs at different fidelity levels for the same scenario samples are positively correlated.
- A navigation solve failure returns a valid failure record rather than an exception.
- These tests require a GPU and are marked so they run locally or on CARC, not in GPU-less CI.

**Seam 2: Estimator interface**

- Tested against toy models with analytically known means, variances, and correlations standing in for the simulator.
- Each estimator's mean over many repeats falls within a confidence interval of the known true value.
- Predicted variance matches empirically observed variance on the toy problems within statistical tolerance.
- Multifidelity estimators beat plain Monte Carlo at equal cost when toy models are highly correlated.
- The smoothed failure indicator converges to the hard indicator as steepness increases.
- When the selective refinement stretch item starts: on toy models, selective refinement returns the same failure probability as always using the highest level, within statistical tolerance, at lower average cost.
- These tests run fast and without a GPU.

**Seam 3: Results format**

- Every file the pipeline writes validates against the versioned schema.
- The viewer loads a small checked-in sample fixture and renders a replay without errors.

**Prior art:** the repository is new, so there are no existing tests to follow. The toy-problem testing approach mirrors how MXMCPy's own examples and test suite exercise estimators on models with known properties.

**Deferred seam:** if the C++ navigation module stretch goal starts, it gets its own seam (matches and camera parameters in, position out) so Python and C++ outputs can be compared on identical inputs.

## Out of Scope

- Flight dynamics, guidance, control, or 6-DOF simulation; the descent is scripted.
- Hazard detection or landing-site selection.
- Live, in-browser simulation or a GPU backend for the viewer.
- Validation against real descent-camera imagery.
- Training on any Jezero-area terrain.
- Publishing SuperPoint weights or any SuperPoint derivative.
- Relightable Gaussian splats (future work).
- Rare-event estimation methods such as importance sampling (stretch goal).
- Deep ensembles and map/DTM error as random inputs (stretch goals).
- A pendulum-swing attitude model in v1 (planned for v1.1).
- The C++ flight-style navigation module in the core milestones (stretch goal).
- Commercial use.

## Further Notes

**Milestones and definitions of done.** Each ticket should be assigned to the milestone whose definition of done it serves. Milestones are vertical slices: each one leaves a working end-to-end system.

- **M1 — Smoke tests.** Mitsuba (CUDA), gsplat, LightGlue with ALIKED, and MXMCPy each run a minimal example on both the local RTX 5070 and at least one CARC GPU type, and a Snakemake workflow submits at least one job through SLURM. Any incompatibility is fixed or documented with a workaround.
- **M2 — End-to-end plain Monte Carlo ("skateboard").** One command runs one Jezero tile through splat scene, scripted descent, pretrained matcher, and PnP to navigation error, and outputs a plain Monte Carlo failure probability with a confidence interval from a small sample.
- **M3 — Fidelity ladder and estimator comparison ("scooter").** Three fidelity levels exist with measured per-sample costs. The pilot study produces correlations, r is set from it, and MXMCPy produces MFMC, MLMC, and ACV estimates with predicted variances, compared against plain Monte Carlo at equal cost. M1–M3 together form the first showable result.
- **M4 — Fine-tuning ("bicycle").** The matcher is fine-tuned on mesh renders of the six training regions, with no Jezero data in training, and failure probability is reported for fine-tuned versus pretrained.
- **M5 — Lighting and validation ("motorcycle").** 4–6 sun-angle splat scenes exist; a brute-force reference value is computed; estimators are repeated within a stated GPU-hour budget; predicted versus actual variance is reported, along with failure probability versus sun angle and versus r (with the LVS 40 m reference line).
- **M6 — Viewer and write-up ("car").** The Spark replay viewer is deployed publicly with precomputed scenario presets, plus a highlight video and a README with headline results and cited NASA sources.
- **M7 — Stretch.** Each item is its own ticket, started only after M6. Rough priority: selective refinement comparison against sigmoid smoothing, pendulum-swing attitude model, C++ flight-style navigation module, map/DTM error, deep ensembles, rare-event methods. Relightable splats remain future work.

**Known risks to check early (M1).**

- The RTX 5070 is a Blackwell-generation GPU and needs recent PyTorch and CUDA builds; Mitsuba, gsplat, and LightGlue must all be confirmed on it.
- MXMCPy's PyPI release is v1.0 from the Python 3.6 era; older dependency pins may conflict with a modern environment. Fixes are candidates for upstream contributions.
- V100 GPUs may not benefit from LightGlue's FlashAttention acceleration; prefer A100 or L40S for timing measurements.
- Smoothing must preserve enough cross-fidelity correlation for multifidelity estimators to beat plain Monte Carlo; the pilot study confirms this.
- The CTX-like map may be too coarse to match against sharp HiRISE-derived descent renders, even in perfect conditions. A blur sweep on the perfect-conditions render (sharp map to CTX-like) shows where matching breaks. If blur is the cause, fallbacks in order: restrict map matching to the altitude band where LVS used it, where descent and map resolution are closer; check whether the fine-tuned matcher closes the gap; as a last resort, use a less-coarse map, recorded in an ADR and listed as a limitation.
- If the coarse map, not render quality, limits navigation accuracy, the three fidelity levels may give nearly identical errors. Correlations then look excellent while the high level stops being meaningfully higher-fidelity, which inflates apparent speedups. The pilot study reports whether the levels differ in error, judged against a criterion written down before the pilot runs. If they don't differ, that is reported as a finding (cheap renders suffice when the map is the bottleneck), and the ladder may be rebuilt around what still affects matching, such as lighting and shading quality. The brute-force validation remains the backstop against a paper-only speedup.

**Open items to verify at implementation time.**

- Exact HiRISE DTM product IDs for each training region.
- CTX ground resolution and the actual LVS reference-map resolution, to set the downsampling factor for the matching map.
- react-three-fiber compatibility with Spark.
- Details of JPL's F Prime flight software framework before relying on it for the C++ stretch goal.

**Related work.** A light literature pass, not an exhaustive review.

- *Mars 2020 Lander Vision System.* Johnson et al. (JPL), "The Lander Vision System for Mars 2020 Entry Descent and Landing" (AAS GNC 2017) and "Mars 2020 Lander Vision System Flight Performance"; Cheng, Ansar & Johnson, "Making an Onboard Reference Map From MRO/CTX Imagery for Mars 2020 Lander Vision System" (Earth and Space Science, 2021). LVS matched descent-image landmarks to a CTX-derived map and fused them with IMU data, reducing position error from up to 3.2 km to under 40 m; touchdown was 5 m from target. JPL's verification combined field tests with high-fidelity Monte Carlo EDL simulation, which Fidelis mirrors at small scale. Fidelis differs by omitting IMU fusion and flight dynamics, and by studying a learned matcher.
- *Multilevel and multifidelity failure probabilities.* Elfverson, Hellman & Målqvist, "A multilevel Monte Carlo method for computing failure probabilities" (SIAM/ASA JUQ, 2016), introduces selective refinement for the indicator-function problem. Related work includes multifidelity importance sampling and multifidelity probability estimation via fusion of estimators. Williamson et al. (Strathclyde, 2024) applied the MLBLUE multifidelity estimator to re-entry landing-location uncertainty, the closest landing-related precedent found; it is physics-based re-entry, not vision-based navigation.
- *Multifidelity Monte Carlo foundations.* Peherstorfer, Willcox & Gunzburger on MFMC and optimal model management; Gorodetsky et al. on approximate control variates; Bomarito et al. on MXMCPy.
- *Gaussian splatting for planetary surfaces.* Moon-GS (IEEE TGRS, 2025) and Casadesus Vila, Dai & Gao (arXiv, 2026) use 3DGS for lunar rover-scale reconstruction and mapping. AstroSplat (arXiv, 2026) integrates planetary reflectance models into Gaussian splatting, which is the closest prior work to Fidelis's relightable-splats future work.
- *Pose regression vs. structure-based localization.* Sattler et al., "Understanding the Limitations of CNN-based Absolute Camera Pose Regression" (CVPR 2019).

**Novelty wording.** This pass found no prior use of Gaussian-splat render levels as a multifidelity ladder for descent-navigation reliability, but a light pass cannot prove absence. README and résumé wording should say Fidelis *applies* multifidelity UQ to splat-rendered, vision-based Mars descent navigation, not that it is the first to do so.

**Sources to credit and cite.**

- USGS Astrogeology: Mars 2020 TRN HiRISE DTM and orthoimage mosaics, and the MSR TRN HiRISE mosaic.
- HiRISE (NASA/JPL-Caltech/University of Arizona) for DTMs and imagery.
- Bomarito et al., "Multi Model Monte Carlo with Python (MXMCPy)," NASA/TM–2020–220585, Langley Research Center.
- LightGlue (Lindenberger, Sarlin, Pollefeys, ICCV 2023), ALIKED, gsplat, Mitsuba 3, and Spark.