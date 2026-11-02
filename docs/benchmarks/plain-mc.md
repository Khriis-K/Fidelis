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
- **Cost per sample** (render + camera model + navigation, wall clock): median 0.459 s, 0.379–0.582 s, except seed 0 at 0.917 s, the first render in the process. That outlier is probably one-off GPU kernel compilation (not checked). Scene loading (1.5–1.9 s per sun, `docs/benchmarks/mesh-render.md`) is outside the timed window: the first run, before that fix, put it inside, and the first descent under each sun cost about 2 s.

## Same seed, same error? Not exactly, on the GPU

The run was made twice (the second after the cost fix, which changes nothing about what's rendered or matched). Per seed, the error moved by a median 0.08 m and at most 0.54 m (seed 10: 1.75 → 2.29 m); 8 of 20 moved more than 0.1 m, and inlier counts shifted by a few. Within one process the GPU seam test repeats the error within its 0.05 m tolerance. Across processes it doesn't. The renderer repeats to 1e-6 per pixel (`docs/benchmarks/mesh-render.md`), so the drift is probably in matching (`docs/benchmarks/navigator.md` already saw one result move 0.8 m between runs), but that isn't confirmed. On the CPU the whole pipeline repeats exactly (tests/test_simulation.py).

What it means: a record regenerates from its seed to within about half a metre, not bit for bit. Against r in the tens of metres that's small; near r it could flip a sample. The pilot study should rerun a few seeds to measure the spread before r is set.

## Caveats

- **r = 12 m is a placeholder** (two 6 m map pixels), not the pilot study's r. No descent came near it, so P(fail) = 0 says only that these 20 descents all landed within 6.12 m.
- **N = 20.** The Wilson upper bound, 16%, is the honest summary: anything from 0 to about a sixth is consistent with this run.
- **Easier than flight:** tilt capped at 3° and offsets at ±400 m by the tile (#33), true attitude, 2 m-detail renders matched against a map from the same photo, no sky light.
- One machine, one GPU, two runs.
