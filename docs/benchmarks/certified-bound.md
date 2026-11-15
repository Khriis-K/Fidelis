# 10,000-descent realistic run: certified bound and brute-force reference (issue #47)

ADR 0004 decision 1 and ADR 0005 decision 6. One plain Monte Carlo run on the full realistic model, which gives two results: the certified upper bound on P(fail at 40 m), and the brute-force reference for mean map-frame error that #21's estimator trials are judged against. This file fixes the run **before** it happens. Results go under the plan afterwards, and any departure from the plan is stated there with its reason.

**What it measures, worded with care.** The failure rate **under the scenario model's sourced ranges**, exactly as the combined scan ran it (`combined-scan.md`): five realistic suns, the CTX map, dust haze over τ 0.5–1, focal error over the pre-calibration ±1%, attitude and misalignment errors over their bounds. Two of those are labelled ranges, not flight values. The bound inherits them.

## Plan

**Status: written 2026-11-15, before any of these seeds ran (committed at `4d31985`); the results are under it.** Nothing in this section is a measurement, except numbers marked **[Computed]** from existing records.

### Step 0: none needed, the cost is already measured

`CLAUDE.md` asks for one cheap measurement before a big run. Here it's a calculation on records we have, not a new job. The combined scan ran 900 descents on this config, on an A40 on `debug` (job 14258755): the full-model run's median was 0.503 s per descent, 99th percentile 0.545 s, max 0.786 s. The whole job took 483 s for 436 s of descents, so loading costs about 16 s per run **[Computed]**. #48's step 0 gave 0.513 s on 30 more seeds. ADR 0004's 0.45 s came from runs before haze and the sourced errors were added.

### What runs

- **Model:** `config/jezero.yaml` as is, no flags. The real navigator on the mesh renderer (`--fidelity mesh`), the CTX map, `--sun-index` off. Every record must carry config SHA-256 `0e7ccb36…`, the combined scan's, which is the committed file at `ce08d0b` (the laptop copy hashes differently only because of CRLF line endings).
- **Seeds: 80,000–89,999**, 10,000 in all, all fresh. 70,000–70,299 stay reserved for #48's unrun sweep; 70,300–70,329 were #48's step 0.
- **Four batches of 2,500**, one `sbatch` job each, on `debug`, one A40 each: batch b gets seeds 80,000 + 2,500·b to 80,000 + 2,500·b + 2,499.
- **One at a time** (`--dependency=singleton`). Two jobs on one node would share its CPUs and memory bandwidth, so their costs might not compare with the earlier runs. Serial costs about 90 minutes of wall time instead of about 25.
- **Hardware:** A40 only. The analysis refuses records from any other GPU model.

**Departure from the issue.** #47 says to run on `gpu` in batched jobs once #12 (batched Monte Carlo jobs) is done. At 0.5 s per descent, a 2,500-descent batch takes about 21 minutes and fits under `debug`'s 1-hour limit, so four hand-submitted jobs do the batching that #12 would. Every cost so far was measured on `debug` A40s too, so this keeps the GPU pool the same. #12 stays open for anything bigger.

### Commands (the user runs them, from the clone on `/project2`)

    cd /project2/irahbari_1147/ckuizon/fidelis
    git pull
    sha256sum config/jezero.yaml      # expect 0e7ccb36...
    for b in 0 1 2 3; do s=$((80000 + 2500*b)); sbatch --account=irahbari_1147 --partition=debug --gres=gpu:a40:1 -c 8 --mem=32G -t 40 -J certified_bound --dependency=singleton -o results/certified_bound_b${b}_%j.log --wrap "module purge && module load ver/2607 gcc/15.3.0 cuda/13.0.3 && uv run --no-sync fidelis-plain-mc config/jezero.yaml data/jezero results/certified_bound/batch$b --fidelity mesh --first-seed $s --samples 2500"; done

Each batch writes `records.jsonl` and `estimate.json` under `results/certified_bound/batch<b>/`, on `/project2`, so a `/scratch1` purge can't take them.

- **Rough cost [Computed], from the measured medians:** 2,500 × 0.503 s + 16 s ≈ 21.2 minutes a batch, 85 A40-minutes in all, so **about 2,050 SUs** at 24 SUs a minute. The 40-minute limit caps each batch at 960 SUs, all four at 3,840.
- **Why 40 minutes:** records are written only when a batch finishes, so a batch that hits its limit loses all 2,500. 40 minutes is about 1.9× the estimate. Even at the slowest descent seen (0.786 s), a batch takes 33 minutes.
- **If a batch fails,** rerun that batch alone, same seeds. A partial record set is never pooled.

### What counts as a failure (fixed by ADR 0003 and #46, not chosen here)

- **r = 40 m, horizontal, in the map frame.** A descent fails if it gets no fix, or if its fix is more than 40 m from the truth horizontally once the map offset (+1.89 m east, −1.49 m north) is removed.
- No-fix and beyond-r failures are counted separately. r is not tuned, whatever the count.

### What gets reported

1. **Checks before any number:** 10,000 records, seeds 80,000–89,999 each exactly once, one GPU model (A40), one config SHA (`0e7ccb36…`), the realistic sun list, `stress_case` false.
2. **The headline:** failures out of 10,000, no-fix and beyond-r separately, with two 95% upper bounds: the Wilson interval `fidelis-plain-mc` prints, and the rule of three (3/n), with the exact one-sided Clopper–Pearson value beside it.
3. **The reference for #21:** mean map-frame horizontal error with its standard error, plus the median, 99th percentile and maximum. The full records stay on `/project2` and are copied to the laptop. None are ever reused as pilot samples (#47).
4. **By sun:** descents, failures, and the error's median, 99th percentile and maximum for each of the five suns.
5. **What the simulator can't produce,** stated next to the bound: real cameras (the focal and attitude errors are labelled ranges on a model camera), cast shadows (the mesh renderer doesn't cast them), and suns more than 113° from the map's sun (the farthest stress sun).
6. **Exploratory, labelled as such:** failures and errors at 12 m; cost per descent (median, 99th percentile) per batch, with node names.

### Prediction from the sources

The only source on our own conditions is the combined scan (300 descents, seeds 60,000–60,299, same config, same navigator, same GPU type). The JPL requirement (LVS within 40 m 99% of the time, Mohan et al. 2021 p.4) sets r but says nothing about our failure rate: LVS had about 10 s and several images to converge, and Fidelis judges one image. So the source doesn't cover our conditions for P(fail), and every prediction below comes from the combined scan.

- **Failures at 40 m: 0 of 10,000.** The combined scan's worst was 17.45 m, under half of r, and no descent lost its fix (fewest inliers 698). The bounds would then be 0.038% (Wilson) and 0.030% (rule of three; exact 0.0300%) **[Computed]**.
- **Mean horizontal error inside 5.31–6.04 m**, the combined scan's 95% interval for the mean (5.68 m, sd 3.21 m) **[Computed]**. With 10,000 samples its standard error should come out near 0.03 m (3.21/√10,000).
- **Median about 5.3 m, 99th percentile about 16 m** (combined scan: 5.34 m and 16.31 m).
- **Worst error 20–30 m [Inference].** With 33 times as many draws the tail reaches further than 300 did. The combined scan's 17.45 m came from a high-|e| focal draw on a low sun; the focal term alone tops out at 14.3 m over 10⁶ draws (`combined-scan.md`), and the control's worst on the old map was about 10 m. Lined up exactly, those add to about 24 m.
- **At 12 m (exploratory): 260–730 of 10,000.** The combined scan's 13 of 300 gives a Wilson interval of 2.5–7.3% **[Computed]**.
- **Cost: median 0.50–0.52 s per descent** in every batch.

**A surprise would be** any lost fix; any error over 30 m; a mean outside 5.31–6.04 m; or a batch whose median cost is outside 0.48–0.55 s. Each is reported against the prediction it breaks.

### What each outcome means

- **0 failures (predicted).** Under the sourced ranges, P(fail at 40 m) is under 0.038% at 95% confidence. That's the number for the pitch, stated with its ranges and with what the simulator can't produce.
- **1–3 failures.** The bound is still small: the Wilson upper bound is 0.057%, 0.073% and 0.088% at k = 1, 2 and 3 **[Computed]**. Report each failure's seed, kind and draws (sun, τ, e, tilt, attitude), and look for the shared cause. Nothing is tuned in response.
- **More, or any lost fix.** The combined scan's 300 seeds missed a tail. Check the error distribution against the combined scan first: if the bulk matches and only the tail differs, the 300 were too few to see it. If the bulk moved, something changed in the code or data since `2a252e9`, and that gets found before the number is used.
- **A batch fails the checks** (wrong GPU, wrong SHA, missing seeds): that batch is rerun, never pooled.

### Known before the run

- The combined scan already ran this config on 300 seeds with 0 failures. This run tests the same model on 33 times as many draws. It adds no new error source.
- The cost figures come from one `debug` node (b11-09). If the batches land on another A40 node, the record's cost shows whether that matters.

## Results

**Run on 2026-11-15.** SLURM jobs 14267261–14267264, debug partition, all on node b11-09, one NVIDIA A40 each, one after another as planned. All `COMPLETED`, in 21 min 10 s to 21 min 24 s each (`sacct`), 85 minutes in all: about **2,040 SUs** at 24 a minute **[Computed]**, against the plan's 2,050. **No departures from the plan.**

The numbers below come from the records with a throwaway analysis script (not committed), using `fidelis.plain_mc.read_records`, `is_failure`, `wilson_interval` and `fidelis.map_frame.horizontal_error_m`. "Error" is the map-frame horizontal error.

### 1. Checks

All passed: 10,000 records, seeds 80,000–89,999 each exactly once, every record on an NVIDIA A40, every one with config SHA-256 `0e7ccb36…`, the realistic sun list, `stress_case` false in every batch.

### 2. The headline

**1 failure in 10,000 at r = 40 m: 1 no-fix, 0 beyond r.** P(fail) = 0.01%, with a **95% Wilson upper bound of 0.057%** and an exact one-sided Clopper–Pearson bound of 0.047% **[Computed]**. The rule of three applies to 0 failures only, so it isn't quoted. Every fixed descent was under 40 m by a wide margin; the worst was 22.62 m.

This is the rate **under the sourced ranges**, and the one failure comes from a part of those ranges that sits outside what the sources tested (section 4). It is not the flight rate.

### 3. The reference for #21

Over the 9,999 descents that got a fix: **mean 5.807 m, standard error 0.032 m** (sd 3.227 m), median 5.325 m, 99th percentile 15.44 m, max 22.62 m (seed 84415). The records stay at `results/certified_bound/batch{0..3}/records.jsonl` on `/project2` and on the laptop. None are to be reused as pilot samples.

| Sun | Az, el | Descents | Failures | Median | 99th pct | Max |
|---|---|---|---|---|---|---|
| 0 | 257°, 40° | 2,024 | 0 | 4.94 m | 14.64 m | 18.76 m |
| 1 | 240°, 25° | 1,998 | 0 | 6.84 m | 16.20 m | 20.80 m |
| 2 | 240°, 55° | 2,032 | 0 | 4.18 m | 13.85 m | 22.62 m |
| 3 | 310°, 25° | 1,941 | 0 | 6.35 m | 16.37 m | 19.56 m |
| 4 | 310°, 55° | 2,005 | 1 | 3.77 m | 14.19 m | 19.25 m |

### 4. The one failure: an overexposed image

**Seed 85247**, sun 4 (310°, 55°), exposure gain 1.494 (the 99.4th percentile of the draws, near the 1.5 top), tilt 17.2°, τ 0.75, focal error −0.17%. It matched 1 point and kept 0 inliers (status `too_few_matches`). The next-lowest of the 10,000 kept 329, so this is a cliff, not the edge of a slope.

**The low tail is an exposure tail.** Of the 10 descents with the fewest inliers, 9 drew a gain of 1.42 or more, all 9 under a high sun (2 or 4, 55°). The worst error, 22.62 m (seed 84415), drew a gain of 1.496 under sun 2. Over all 10,000, gain barely ranks with inliers (Spearman −0.07). It matters only at the top of its range.

**Re-rendering the seed on the laptop, changing only the gain** (the thinned map `mesh-k8000`, because the full map doesn't fit the laptop's GPU; the image is the same render) **[Computed]**:

| Gain | Mean brightness | Pixels clipped | Result (thinned map) |
|---|---|---|---|
| 1.494, as drawn | 248 DN | 65% | No fix, 0 inliers (as on the A40) |
| 1.25 | 221 DN | 11% | Fix, 135 inliers |
| 1.0 | 178 DN | 0% | Fix, 142 inliers |
| 0.6 | 107 DN | 0% | Fix, 143 inliers |

So the image saturated, and two thirds of it was pure white. The haze lifts every pixel towards the brightest ground before exposure, and a 55° sun leaves little shading, so the histogram sits high and narrow. Our exposure model puts the 99.9th-percentile pixel at gain × full scale (`fidelis.render.degrade.expose`), so above a gain of 1 the bulk of a narrow histogram clips.

**Where this sits on the source's axis.** The sources set exposure against the **mean** brightness: a 100 DN target, a flight mean of about 60 DN, and LVS tested from 128 DN down to 16 DN, which is where "off by 4× either way" comes from (Johnson et al. 2017 p.14, Johnson et al. 2022 p.15; `docs/research/edl-dispersions-for-scenario-sampler.md`). Nothing brighter than 128 DN was tested. This image's mean was 248 DN, about twice the brightest tested and about 4 times the flight mean. Saturation is a failure the sources do report: 3 of the 20 coarse retries in JPL's Monte Carlo (Johnson et al. 2020 Table 5, `lvs-error-sources.md`). So the mechanism is a real one, and the draw that triggered it lies outside the sourced envelope.

**What that says about the exposure range** (one seed, so a lead, not a measurement). On this seed our gain of 0.6 gives 107 DN, about the source's 100 DN target, and our gain of 1.0 already gives 178 DN. If that holds across scenes, the config's range of 0.5–1.5 runs from about the flight target up to about 2.5 times it, not "centred near the flight value" as the config comment says. That needs the mean brightness measured over many seeds before anything changes. Nothing is re-tuned here, and the failure stays in the count.

### 5. Exploratory

- **At 12 m:** 483 of 9,999 fixed descents.
- **Cost:** median 0.4989–0.5022 s per descent across the four batches; 99th percentile 0.535–0.541 s; max 0.608 s. All on b11-09.

## Against the prediction

| Prediction (before the run) | Result | Held? |
|---|---|---|
| 0 failures of 10,000 | 1 (a lost fix) | **No.** A surprise, as the plan defined it |
| Surprise line: any lost fix | Seed 85247, an overexposed image | Crossed |
| Mean inside 5.31–6.04 m | 5.807 m | Yes |
| Standard error near 0.03 m | 0.032 m | Yes |
| Median about 5.3 m, 99th percentile about 16 m | 5.33 m, 15.44 m | Yes |
| Worst error 20–30 m **[Inference]** | 22.62 m | Yes |
| 260–730 over 12 m | 483 | Yes |
| Batch median cost 0.50–0.52 s | 0.4989–0.5022 s | Three batches a hair under the range; the 0.48–0.55 s surprise line held |

**Why the prediction missed.** It rested on the combined scan's 300 seeds, where the fewest inliers was 698. A lost fix at the corner of three draws (gain above about 1.4, a high sun, and enough haze to narrow the histogram) showed up once in 10,000, too rare for 300 descents to be likely to show it. The plan's own "more failures" branch named this: "the 300 were too few to see it". The bulk of the distribution matched the prediction everywhere.

## What this means

- **The certified number is P(fail at 40 m) under 0.057% at 95% confidence**, under the sourced ranges. That's still about 17 times under JPL's rough 1%, with the same caveat as before: LVS had about 10 s and several images, and Fidelis judges one.
- **The one failure is the model's exposure range, not the navigator's matching or geometry.** It comes from an image twice as bright as anything LVS was tested on. A flight-like exposure on the same seed works. So the bound is, if anything, pessimistic about exposure, as long as the gain anchor question in section 4 is open.
- **Not settled:** how the model's gain maps to mean DN across scenes. If it really runs to about 2.5 times the flight target, re-anchoring the exposure to mean brightness would remove draws like this one. Removing them would need a fresh pre-registered run, never an edit to this count.
- **For #21:** the reference mean is 5.807 m ± 0.032 m (one standard error) over 9,999 fixed descents.
