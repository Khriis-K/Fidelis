# The Jezero splat scene behind the simulation interface (issue #9)

One Gaussian splat scene of the wide Jezero scene, trained from mesh renders under the landing-day sun (azimuth 257°, elevation 40°; `splat.sun_index: 0`), checked against held-out mesh renders, and run through plain Monte Carlo beside the mesh on the same seeds. Trained and run on 2026-11-02 on CARC, on NVIDIA L40S nodes (b17-16), from commit `2313be8`, config SHA-256 `05bbc8fb…` (recorded in every record and in `quality.json`).

    uv run --no-sync snakemake --profile profiles/carc splat plain_mc_splat plain_mc_mesh_pinned

## What was built

- **Training views** (`fidelis-splat-views`): 300 training and 20 held-out views, each the true descent camera of a scenario drawn from the config (up to 18° tilt, ±2.75 km offsets), on seed ranges at and above 1e9 so no Monte Carlo descent (seeds below 1e9) renders a pose the splat trained on. Path-traced at 256 spp, LCAM's 1024 × 1024 px at 90°. Job 13682196: 30 s.
- **Training** (`fidelis-splat-train`): one Gaussian per mesh vertex to start (6,275,025 tile + 1,709,126 ring), one grey channel, gsplat's `DefaultStrategy` and simple_trainer's learning rates, loss 0.8 L1 + 0.2 (1 − SSIM), 30,000 steps. Job 13682227: 610 s of training (from its log), 12 min 49 s for the whole job. It ended with **7,715,069 Gaussians**: densification pruned more than it grew. Checkpoint 354 MB, browser PLY 501 MB, both under `data/jezero/splat/sun0/` on `/project2`.
- **Renderer**: `SplatRenderer` serves fidelity `"splat"` through `Simulator.run`, like `"mesh"`.

## Held-out image quality

Each of the 20 held-out views rendered by the splat against the mesh's render of the same pose. PSNR and SSIM are relative to the camera's full scale (the mesh render's 99.9th percentile, where `degrade.expose` puts full scale). Gate (provisional, `splat.quality`): every view ≥ 30 dB and ≥ 0.9, and every constant-image view must fail.

| | Worst PSNR | Worst SSIM |
|---|---|---|
| Splat vs mesh | **32.56 dB** | **0.9775** |
| Constant image (each view's mean brightness) | 17.83 dB | 0.3587 |
| Gate | ≥ 30 dB | ≥ 0.9 |

**Passed**, and the constant image is rejected. Worst-view numbers only: the per-view list and medians are in `quality.json` on CARC and weren't copied here. The PSNR margin is 2.6 dB over a threshold we chose; 32.56 dB is an RMS error of about 2.4% of full scale, still 3–12× the camera noise the sampler draws (0.2–0.8%). The references are 256-spp renders with their own sampling noise, which caps what any splat can score against them.

## Plain Monte Carlo: mesh vs splat, same seeds, same sun

Seeds 0–19, every descent under sun 0 (`--sun-index 0`), same config, same GPU type. `plain_mc_mesh_pinned` (mesh, 256 spp) and `plain_mc_splat`.

| | Mesh | Splat |
|---|---|---|
| Failures at the provisional r = 12 m | 0 / 20 | 0 / 20 |
| 95% Wilson interval | 0–16.1% | 0–16.1% |
| Median error | 2.37 m | 2.20 m |
| Error range | 1.09–3.21 m | 1.02–2.83 m |
| Median cost per descent | 0.234 s | 0.184 s |

Paired over the 20 seeds (computed from the two `records.jsonl`):

- **Errors track each other closely: Pearson r = 0.933, Spearman 0.920.** Inlier counts differ by 0.2–3.3% per seed (1,560–2,434 on the splat).
- **The splat's errors are slightly lower**: splat − mesh averages −0.145 m (sd 0.222 m, range −0.70 to +0.16 m), lower on 15 of 20 seeds, Wilcoxon signed-rank p = 0.011. Why isn't measured. One candidate: the splat renders without the path tracer's sampling noise.
- **Cost: the splat is only 21% cheaper per descent** (median ratio 0.79; mean saving 0.050 s over seeds 1–19). Seed 0 is the first descent in each process and costs more (mesh 0.579 s, splat 2.232 s, which likely includes gsplat loading its compiled extension; not checked). With about 0.05 s saved by swapping the renderer, the rest of each descent, about 0.18 s, appears to be the camera model and the navigator, whichever renderer runs. That's an inference from totals: no per-stage timing exists yet.

## Seam 1 on the splat

`tests/test_simulation_splat.py` on an L40S: **7 of 7 pass** (job 13682975, 10.96 s, from `8e2070c`). The first run (job 13682689, 10.6 s, from `2313be8`) passed 6 of 7: Perfect conditions under the trained sun, a tilted/turned/noisy descent, repeatability of the camera image, held-out-style quality at Monte Carlo poses, splat cheaper than mesh per sample, and a sun with no splat as a caller error.

The seventh, the failure-record test, was wrong as written: it tilted the camera 120° to see empty sky, and a view that reaches the horizon has no nadir image, which the navigator refuses on purpose with an error (the sampler's 18° cap never gets there). It now puts the camera 500 m under the ground looking down, so every Gaussian is behind it and the image is black (checked on a toy splat: exactly 0). It passed on the rerun.

## What this does and doesn't show

- One sun. The four others are #20's.
- 20 seeds, 0 failures on either fidelity: the paired comparison says the splat tracks the mesh, not how either fails. r = 12 m is still provisional.
- The renders and the map still share the orthoimage, as in every earlier benchmark, which flatters matching.
- **For the fidelity ladder:** cheaper splat levels (fewer Gaussians, lower resolution, lower SH degree) can only cut into the ~0.05 s render share of each descent. If the ~0.18 s rest is the navigator, the spec's fallback, shrinking the navigator too, may be what makes the levels' costs differ enough. A per-stage timing should come first.
