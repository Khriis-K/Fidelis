# First plain Monte Carlo run (issue #8)

The first end-to-end run of the simulation interface: 20 seeded descents through the Jezero mesh renderer, the camera model and the navigator, then P(fail) with a 95% Wilson interval. It shows that the pipeline runs end to end. **It isn't a failure estimate:** r is provisional, N is tiny, and the descents are easier than flight (see Caveats).

    uv run snakemake --profile profiles/local plain_mc
    uv run snakemake --profile profiles/local results/plain_mc/matches/seed_14.png

Setup: seeds 0–19, config `config/jezero.yaml` (SHA-256 `f764e270…`, recorded on every record), fidelity `mesh` at 256 spp, LCAM 1024 × 1024 px at 90°, the 6 m CTX-like map, five suns. The navigator gets the true attitude. NVIDIA GeForce RTX 5070 Laptop GPU, 2 November 2026.

## Result

**P(fail) = 0/20 at PROVISIONAL r = 12 m, 95% Wilson interval [0, 0.161].** Every descent got a fix.

| Seed | Sun (az, el) | Error (m) | Inliers / matches | Cost (s) |
|---|---|---|---|---|
| 0 | 257, 40 | 2.64 | 2024 / 2037 | 0.917 |
| 1 | 257, 40 | 2.78 | 2235 / 2254 | 0.402 |
| 2 | 240, 25 | 5.45 | 2085 / 2108 | 0.474 |
| 3 | 310, 55 | 3.16 | 2300 / 2307 | 0.500 |
| 4 | 240, 55 | 1.75 | 2266 / 2273 | 0.561 |
| 5 | 240, 55 | 2.24 | 2032 / 2050 | 0.483 |
| 6 | 240, 55 | 1.91 | 2248 / 2254 | 0.466 |
| 7 | 310, 55 | 2.68 | 2047 / 2059 | 0.489 |
| 8 | 240, 55 | 2.58 | 1915 / 1922 | 0.382 |
| 9 | 240, 55 | 1.59 | 2203 / 2225 | 0.433 |
| 10 | 240, 55 | 2.29 | 1833 / 1849 | 0.492 |
| 11 | 257, 40 | 2.96 | 1777 / 1791 | 0.403 |
| 12 | 257, 40 | 3.26 | 2030 / 2048 | 0.379 |
| 13 | 257, 40 | 2.82 | 1802 / 1816 | 0.444 |
| 14 | 310, 25 | 6.12 | 1806 / 1847 | 0.582 |
| 15 | 240, 25 | 5.85 | 2127 / 2171 | 0.452 |
| 16 | 240, 55 | 2.02 | 2004 / 2013 | 0.431 |
| 17 | 240, 55 | 1.38 | 2344 / 2364 | 0.456 |
| 18 | 240, 25 | 4.92 | 1969 / 1997 | 0.415 |
| 19 | 257, 40 | 2.79 | 1710 / 1722 | 0.461 |

The table is generated from the second run's `results/plain_mc/records.jsonl`.

- **Error:** median 2.73 m, range 1.38–6.12 m. All four descents under the 25° suns had the four largest errors (4.92–6.12 m); the 40° sun gave 2.64–3.26 m and the 55° suns 1.38–3.16 m. That's an observation from 20 samples, not a tested trend, but it's the direction you'd expect: long shadows that the map's photo doesn't have.
- **Cost per sample** (render + camera model + navigation, wall clock): median 0.459 s over all 20 (0.456 s without seed 0), 0.379–0.582 s except seed 0 at 0.917 s, the first render in the process. That outlier is probably one-off GPU kernel compilation (not checked). Scene loading (1.5–1.9 s per sun, `docs/benchmarks/mesh-render.md`) is outside the timed window: the first run, before that fix, put it inside, and the first descent under each sun cost 1.85–2.73 s (below).

## Same seed, same error? Not exactly, on the GPU

The run was made twice (the second after the cost fix, which changes nothing about what's rendered or matched). Against the first run's values (below, as printed, two decimals), per seed the error moved by a median 0.08 m and at most 0.54 m (seed 10: 1.75 → 2.29 m); 8 of 20 moved more than 0.1 m, and inlier counts shifted by a few.

### Why: a tiny render jitter, amplified by the navigator

`uv run python scripts/check_gpu_repeatability.py data/jezero` separates the stages (seed 10, two processes, then one process; numbers from one run of it):

- **The render jitters a little.** Seed 10's radiance differed between the two processes by at most 1.34e-6 (median pixel about 0.32), and its camera image by at most 2.03e-6, in 944,625 of 1,048,576 pixels. That's Mitsuba's CUDA film adding samples in no fixed order. It's slightly above the 1e-6 that `docs/benchmarks/mesh-render.md` states (measured at 64 spp on other views).
- **The navigator doesn't.** One saved image, located in each process: 1.7437 m and 1,830 inliers in both. The same matched pairs solved five times: zero spread. So ALIKED, LightGlue and PnP + RANSAC repeat exactly on the same input.
- **But it amplifies.** That saved image with 1e-6 Gaussian noise added (four draws) located at 1.8648, 2.2893, 1.8151 and 1.8581 m, against 1.7437 m without. A change far below anything a camera could record moves which keypoints and matches survive, and the fix with them.
- **So it happens within one process too.** Three runs of a seed in one process: seed 10 2.2857 / 2.2876 / 2.2857 m, seed 15 5.8622 / 5.9472 / 5.6190 m, seed 0 2.6346 / 2.5088 / 2.5088 m, seed 4 1.7223 / 1.7639 / 1.7474 m, with camera images at most 4.17e-7 apart.
- **Exact GPU renders aren't on offer.** Mitsuba's `samples_per_pass = 1` (one sample per pixel per launch, so no two samples race for a pixel) still differed across processes by up to 7.75e-7, and took 0.59 s against 0.257 s at 256 spp. Mitsuba also marks the option deprecated.

What it means: the same seed regenerates the same camera image to within about 1e-6 on the GPU, but not the same error, which moves by up to about 0.6 m. On the CPU the whole pipeline repeats exactly within one process (across processes isn't checked). The GPU seam test now checks the camera images (within 1e-5, which a wrongly seeded noise draw, about 1e-3, would fail) instead of the errors. Against r in the tens of metres the jitter is small, but it makes samples near r partly random, and it bounds how closely two fidelity levels' errors can agree: the pilot study should measure it per level before r is set.

## Caveats

- **r = 12 m is a placeholder** (two 6 m map pixels), not the pilot study's r. No descent came near it, so P(fail) = 0 says only that these 20 descents all landed within 6.12 m.
- **N = 20.** The Wilson upper bound, 16%, is the honest summary: anything from 0 to about 16% is consistent with this run.
- **Easier than flight:** tilt capped at 3° and offsets at ±400 m by the tile (#33), true attitude, 2 m-detail renders matched against a map from the same photo, no sky light.
- One machine, one GPU, two runs.

## First run, as printed

The first run's `records.jsonl` was overwritten by the second, so its values survive only as printed
during the session, by a one-off script reading that file, rounded to two decimals. They're copied here
so the comparison above can be checked; the first run's costs include the scene load (before the fix).

| Seed | Error (m) | Cost (s) |
|---|---|---|
| 0 | 2.54 | 2.73 |
| 1 | 2.72 | 0.38 |
| 2 | 5.41 | 1.85 |
| 3 | 3.06 | 2.09 |
| 4 | 1.75 | 2.15 |
| 5 | 2.14 | 0.39 |
| 6 | 1.88 | 0.40 |
| 7 | 2.68 | 0.39 |
| 8 | 2.78 | 0.37 |
| 9 | 1.67 | 0.40 |
| 10 | 1.75 | 0.42 |
| 11 | 2.99 | 0.40 |
| 12 | 3.23 | 0.40 |
| 13 | 2.95 | 0.41 |
| 14 | 6.12 | 2.01 |
| 15 | 5.73 | 0.38 |
| 16 | 2.02 | 0.37 |
| 17 | 1.27 | 0.38 |
| 18 | 5.05 | 0.43 |
| 19 | 2.79 | 0.43 |
