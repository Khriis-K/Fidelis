# 4. What Fidelis estimates when the navigator doesn't fail

Date: 2026-11-13. Status: accepted (the user accepted every recommendation as written, 2026-11-13). Each decision keeps its reasoning below.

## Context

Every scan so far gives 0 failures at r = 40 m in the map frame (ADR 0003):

- **The realistic set so far** (sourced error 1, the CTX map): 0 in 300, 95% Wilson interval [0, 0.0126]. Worst descent 10.02 m (`docs/benchmarks/ctx-map-scan.md`).
- **The stress case** (#43): 0 in 300 at sun differences of 39.0°, 73.4°, 98.3° and 112.6° on the de-shaded texture, and at 112.6° on the photo texture. Worst descent 20.57 m (`docs/benchmarks/stress-sun-scan.md`).

A failure indicator that is always 0 is a constant output. Every multifidelity estimator looks perfect on a constant: zero variance at any budget, whatever the cheaper levels do. That is the null case in `CLAUDE.md`. So Fidelis can't show MFMC or ACV beating plain Monte Carlo on P(fail at 40 m). A "speedup" measured there would mean nothing.

The error itself isn't constant. In the stress scan it rises steadily with sun difference: median 5.02 → 10.55 m, with no step at JPL's 100° line. Most of that growth is a mean shift pointing roughly away from the sun (4.67 → 9.93 m). The scatter around it grows only 2.32 → 3.55 m. The realistic suns show the same pattern on a smaller scale: per-sun medians of 1.71–5.05 m on the old map (`ctx-map-scan.md`).

ADR 0003 still holds. r stays at 40 m and isn't tuned to make failures appear (decision 5). Stress results are never reported as the realistic rate (decision 1). No number is the realistic rate until #37–#39 have run (decision 6).

## Two things the handoff got wrong

Re-checked against the committed write-ups before drafting:

- **The stage-by-stage profile already exists.** `docs/benchmarks/descent-profile.md` (2026-11-10, A40, splat) found LightGlue matching at 75% of a descent and the render at 2%. The handoff's "about 0.18 s looks like the navigator (inferred from totals)" comes from `splat.md`, which was written before that profile. ADR 0002 already acted on it. Only the mesh renderer hasn't been profiled.
- **The fidelity sweep ran under one sun.** `docs/benchmarks/fidelity-sweep.md` ran every level with `--sun-index 0`, against the old map. It found that map thinning re-randomizes the metre-level error instead of tracking it. Its best usable level, k32000 (ρ = 0.832), would still give MFMC 1.33× the variance of plain MC. Its own diagnosis: correlation needs a shared cause, and those 300 descents had none. Under one sun, the sun-driven shift, the biggest shared cause we've since found, couldn't show up. This matters for decision 2.

## Decisions to make

### 1. Report P(fail) as a certified upper bound

With 0 failures in n descents, the 95% upper bound is about 3/n (the rule of three). 10,000 plain-MC descents would put it at about 0.03%.

- **Rough cost (an estimate, not a measurement):** 0.45 s per descent (the stress scan's median was 0.447–0.478 s on an A40) × 10,000 ≈ 75 A40-minutes. At 24 SUs a minute that's about 1,800 SUs. It doesn't fit the debug partition's 1-hour limit, so it would run on `gpu` in batched jobs.
- **Recommendation: yes, but only after #37–#39.** A bound only means something for the scenario model it was measured on. Run it now, and it certifies a model that leaves out haze, calibration and attitude error, so ADR 0003 decision 6 would make us run it again. Running it once on the full realistic set costs the same 1,800 SUs and gives the number that goes in the pitch.
- **Failure mode:** a bound of 0.03% says nothing about the cases the simulator can't produce (real cameras, cast shadows, sun differences beyond 113°). That limit is stated next to the number, every time it's quoted.

### 2. Show the estimators on a continuous measure

The quantity of interest (QoI) for the MFMC/ACV demo becomes **the mean map-frame error** over the realistic scenario distribution. It's still validated the way `CLAUDE.md` requires: against brute force, across repeated independent trials, with MXMCPy's predicted variance compared to the observed variance (#21).

- **Why the mean, not the 99th percentile.** MFMC and ACV estimate expectations. A quantile isn't an expectation, so it needs a different estimator and a different variance formula. The 99th-percentile error stays in the report, from plain MC, as a description rather than an estimator demo. **[Inference]** MXMCPy has no quantile estimator. Check before relying on that.
- **This means re-running the level selection under varying suns.** The fidelity sweep's rule stands, but it gets run again with the sun drawn from the realistic list, against the CTX map. Pre-registered as before.
- **Prediction from our own data (not a source):** the per-sun shift is shared by every level that matches the same map under the same sun, so correlation should rise above the one-sun sweep's. How far isn't predicted. The sources don't cover navigator-fidelity correlation.
- **Failure mode, and how we'd know:** if the sun explains nearly all of the error, a lookup table indexed by sun would correlate as well as a thinned navigator. MFMC would then be exploiting the sun variable, not the cheap navigator. The check: report how much of the real navigator's error variance the sun alone explains (per-sun means), next to each level's ρ. If a level's ρ is no better than the sun alone gives, the write-up says so.
- **Recommendation: yes.** It's the only QoI we have that isn't constant, so it's the only one that can test the estimators at all.

### 3. Point fine-tuning (#17, #18) at the error

The fine-tuning target becomes shrinking the map-frame error, especially the sun-driven shift. #18 is retitled from "Fine-tuned vs pretrained failure probability" to "Fine-tuned vs pretrained navigation error". It's measured on the realistic set and the stress suns, separately and labelled.

- **Recommendation: yes.** Comparing P(fail) between two navigators that both give 0 compares two constants.
- **The Jezero exclusion is unchanged** (`CLAUDE.md`). The stress suns are a new test condition, not training data, so they add no new route for leakage.
- **[Inference] The likely target.** The shift probably comes from edges moving with shading (`stress-sun-scan.md`, exploratory). Training pairs need sun differences between the render and the map, or fine-tuning won't see the effect it's meant to fix.

### 4. The profiling step and #20

Profiling is done (see above). The open question is #20 (splat scenes for 4–6 suns). Since ADR 0002 the splat is only a fast stand-in for the mesh renderer, and it's only 21% cheaper per descent (0.184 vs 0.234 s, L40S, `splat.md`). Training 4–6 splats spends GPU time for that 21%.

- **Recommendation: defer #20** until decision 2's sweep shows whether the estimators need the splat's saving. If they do, profile one mesh descent first (laptop-sized, about 30 minutes) to confirm how much of a descent the render takes.

### 5. #37–#39 still come first for the realistic rate

Unchanged from ADR 0003 decision 6. Laptop checks predict no misses from haze, focal-length error or attitude error (comments on #37, #38, #39). Those are predictions, not results.

- **Recommendation: run them next.** Decisions 1 and 2 both need the full realistic model, so this is on the critical path either way.

## Proposed order

#37 → #38 → #39 → the level re-selection under varying suns (decision 2) → the 10,000-descent bound (decision 1) → estimator trials against brute force (#21). Fine-tuning (decision 3) can run alongside once #15 and #16 are ready. Every GPU step asks first (compute guardrail).

## Consequences, if accepted as recommended

- **The headline becomes two numbers:** a certified upper bound on P(fail at 40 m) from plain MC, and the MFMC/ACV speedup on mean navigation error. Neither replaces the other. The speedup is never quoted as a speedup on P(fail).
- **The spec, README and EXPLAINER** say that multifidelity is demonstrated on navigation error because the navigator didn't fail under any sourced condition. That's a finding, not a retreat.
- **#18 is retitled; #20 is deferred;** #21's QoI is the mean map-frame error.
- **ADR 0002 §4** (re-check levels on the smoothed failure indicator once failures occur) stays open. The failures may never occur.
