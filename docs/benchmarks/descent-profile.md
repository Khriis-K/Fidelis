# Where a splat descent's time goes (issue #10)

Before building cheaper fidelity levels, this checks which part of a descent they could make cheaper. The splat Monte Carlo run cost 0.184 s per descent against the mesh's 0.234 s (docs/benchmarks/splat.md), so swapping the renderer saved only about 0.05 s. Cheaper splat levels (fewer Gaussians, lower resolution, lower SH degree) can only cut the render stage. This measures how big that stage is.

`scripts/profile_descent.py` (commit `b7fc789`) runs the real `Simulator.run` on the splat and times each stage in place. Each stage has a CUDA synchronize on both sides, so GPU work lands in the stage that launched it. "other" is the record's `cost_s` minus the six stages: tensor uploads, moving keypoints back to camera pixels, and similar glue. One untimed descent runs first, so gsplat's load and the GPU's warm-up aren't counted.

Run on 2026-11-10 on CARC, debug partition, one **NVIDIA A40**. The node and job ID weren't recorded. Splat scene from #9 under sun 0 (7,715,069 Gaussians), seeds 1–30, every descent under sun 0.

    srun --account=irahbari_1147 --partition=debug --gres=gpu:a40:1 -c 8 --mem 32G -t 30 bash -c "module purge && module load ver/2607 gcc/15.3.0 cuda/13.0.3 && export TORCH_EXTENSIONS_DIR=\$HOME/.cache/torch_extensions_a40 && uv run --no-sync python scripts/profile_descent.py config/jezero.yaml data/jezero --samples 30 --out results/profile_descent/a40.json"

## Result

| Stage | Runs on | Median s | Share of median cost |
|---|---|---|---|
| Render (splat) | GPU | 0.0070 | 2% |
| Camera model (blur, exposure, noise) | CPU | 0.0204 | 6% |
| Nadir warp | CPU | 0.0022 | 1% |
| ALIKED extract | GPU | 0.0354 | 11% |
| **LightGlue match** | GPU | **0.2389** | **75%** |
| PnP | CPU | 0.0008 | 0% |
| Other | both | 0.0090 | 3% |
| Whole descent (`cost_s`) | | 0.3168 | |

All 30 descents got a fix (status `ok`). The stage medians are taken separately, so they don't sum exactly to the median cost. Only the printed table was copied here; the per-seed times are in `results/profile_descent/a40.json` on CARC.

## What it shows

- **Matching is three-quarters of every descent; the render is 2%.** Even a free splat would cut per-sample cost by about 2% on this GPU. Splat-only fidelity levels would all cost almost the same. That's the null case for multifidelity: with near-equal costs, MXMCPy has nothing cheap to spend samples on, so MFMC/ACV can't beat plain Monte Carlo.
- **This triggers the spec's fallback for #10: shrink the navigator too.** LightGlue matches each descent image against all 70,726 keypoints of the wide map (14,290 on the old 10 km tile, docs/benchmarks/plain-mc.md), and its cost grows with both keypoint counts. Keypoint count is therefore the obvious lever. Which one to pull, and how much each saves, isn't measured yet.

## What this doesn't show

- **One GPU type, and not the one costs will be pinned to.** The A40 was the only usable free GPU that evening (docs/carc.md). Shares will differ on an L40S or A100: the CPU stages don't change with the GPU, the GPU stages do. A 75% vs 2% gap is unlikely to flip, but that's an inference, not a measurement.
- **The A40 descent cost isn't comparable with the L40S plain MC run's 0.184 s.** Different GPU, and this run's syncs add a little time to each stage.
- **Splat fidelity only.** The mesh renderer wasn't profiled. Its render stage is larger (0.234 s total on the L40S), but the navigator's stages are the same code.
