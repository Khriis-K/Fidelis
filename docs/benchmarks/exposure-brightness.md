# How bright are our descent images? (follow-up to #47)

The one failure in the 10,000-descent bound was an image that averaged 248 DN, about twice the brightest LVS was tested on (`certified-bound.md`, section 4). The sources set exposure against **mean** brightness: a 100 DN target, a flight mean of about 60 DN, and LVS tested from 16 to 128 DN (Johnson et al. 2017 p.14, Johnson et al. 2022 p.15; `docs/research/edl-dispersions-for-scenario-sampler.md`). Our model sets it from the 99.9th-percentile pixel (`fidelis.render.degrade.expose`). This note measures where our images sit on the source's axis. It's a diagnostic: nothing in the config changes here.

## Plan

**Status: written 2026-11-15, before any render; the results are under it.**

- **Seeds:** 80,000–80,499, the first 500 of the #47 run, so each image can be set against that seed's inliers on the A40. The scenario draws are the same; the render is re-done on the laptop GPU (Mitsuba jitter is about 1e-6, `plain-mc.md`).
- **Per seed:** render, haze, blur, then the image the navigator saw (`camera_image`). Record its mean in DN (×255) and the fraction of pixels clipped at 1.0. Also the mean at gain 1 before clipping and noise, m₁. Since `expose` is linear, the gain that would put a seed's mean at the 100 DN target is 100 / m₁.
- **No matching,** so this runs on the laptop with no cluster time.

## Prediction from the sources

The sources don't say how our gain maps to DN. The only measurement is seed 85247: m₁ ≈ 178 DN.

- **If every scene had m₁ = 178 DN,** an image goes above 128 DN once the gain is over 0.72, which is 78% of uniform draws on 0.5–1.5 **[Computed from that assumption]**. Scenes differ, so the prediction is **50–85% of images above 128 DN [Inference]**.
- **High suns are brighter than low suns** at the same gain: less shading makes the histogram narrower, so its mean sits closer to its 99.9th percentile **[Inference]**.
- **The images with the fewest #47 inliers are the brightest, most clipped ones.**

**A surprise would be** a median m₁ below 128 DN (the range is mostly inside the tested envelope, and seed 85247 was an outlier), or no link between brightness and inliers.

## Results

**Run on 2026-11-15** on the laptop GPU, 500 seeds (80,000–80,499), about 8 minutes, with a throwaway script (not committed) that calls `scenario_pose`, the mesh renderer, `scenario_transmittance` and `camera_image` as `Simulator.run` does, without the navigator. Inliers are the A40's from `results/certified_bound/batch0/records.jsonl`. **No departures from the plan.**

### Where our images sit on the source's axis

| | Min | 10th pct | Median | 90th pct | Max |
|---|---|---|---|---|---|
| Mean brightness, as drawn | 51 DN | 83 DN | 139 DN | 201 DN | 240 DN |
| Mean at gain 1, m₁ | 96 DN | 119 DN | 141 DN | 164 DN | 182 DN |
| Gain that gives the 100 DN target | 0.5 | 0.6 | 0.7 | 0.8 | 1.0 |

- **59% of images average above 128 DN,** the brightest LVS was tested on. 78% are above the 100 DN target, and none is below 16 DN.
- **Our gain of about 0.7 is the source's target,** so our range of 0.5–1.5 runs from about 0.7× to about 2.1× the target **[Computed]**, from the median m₁. Flight's 0.6× (60 DN) would be our gain of about 0.42, **below the bottom of the drawn range**. The config's comment, "the range around the flight value", doesn't hold: the range sits around 1.4× the target, more than twice as bright as flight.
- **Clipping is rare but real:** 4.8% of images have more than 10% of their pixels clipped, and 0.2% (1 of 500) more than half.
- **High suns are brighter,** as predicted: median m₁ is 150 and 146 DN for the 55° suns (4 and 2), 129–139 DN for the others.

### Brightness and matching

| Mean brightness | Images | Inliers (A40), median | Fewest |
|---|---|---|---|
| Under 128 DN | 205 | 1,371 | 699 |
| 128–200 DN | 242 | 1,385 | 836 |
| Over 200 DN | 53 | 1,221 | 631 |

Brightness alone barely matters (Spearman with inliers −0.02); clipping matters a little (−0.10). Above the tested 128 DN the navigator does as well as below it, until the image starts to clip: the brightest bin loses about 11% of inliers, and it holds the fewest-inlier image of the 500 (240 DN, 53% clipped). Low inlier counts also come from dark images (65 and 70 DN among the 10 lowest), so brightness isn't the only way into the tail.

## Against the prediction

| Prediction (before the run) | Result | Held? |
|---|---|---|
| 50–85% of images above 128 DN **[Inference]** | 59% | Yes |
| High suns brighter than low suns | 146–150 DN against 129–139 DN (median m₁) | Yes |
| The fewest-inlier images are the brightest, most clipped | The fewest is (240 DN, 53% clipped); 3 of the 10 fewest are over 200 DN, against 11% of all images. The others are mid or dark | Partly |
| Surprise: median m₁ under 128 DN | 141 DN | Not crossed |

Seed 85247's m₁ of 178 DN was near the top of the range (the 500's maximum is 182 DN), not typical.

## What this means

- **Our exposure range runs bright.** Measured against the source's own mean-brightness target, the drawn range is about 0.7–2.1× the target, and flight (0.6×) sits below it. Most images are brighter than anything LVS was tested on.
- **The navigator is mostly indifferent to that,** until pixels clip. Heavy clipping happens only at the top of the range: high gain, a high sun, haze. That is the corner where the #47 failure came from.
- **So the #47 bound is pessimistic about exposure, not optimistic.** A range anchored to the source's target, such as 0.5–1.5× of 100 DN, would be our gain of about 0.35–1.05. On seed 85247 a gain of 1.0 left 0% of pixels clipped and the image located (`certified-bound.md`). Whether the failure would disappear under a re-anchored range is **[Inference]**: it needs a fresh pre-registered run, and the #47 count isn't edited.
- **Not settled here:** a laptop render, not the A40's (the difference is render jitter of about 1e-6, `plain-mc.md`), and 500 seeds, not 10,000.
