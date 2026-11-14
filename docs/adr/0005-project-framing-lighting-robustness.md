# 5. Fidelis is a lighting-robustness study, with multifidelity as one bounded experiment

Date: 2026-11-13. Status: accepted (the user, in session, 2026-11-13).

## Context

Fidelis set out to show four skills on one aerospace problem: Gaussian splatting, deep learning, HPC, and applicability to aerospace. Multifidelity Monte Carlo on P(fail) was the glue between them. ADR 0004 took that glue away. The failure indicator is 0 in every scan, so multifidelity moves to mean navigation error, and its first test of cheap levels (one sun) said they don't help.

Checked against that goal, as of this ADR:

- **HPC and aerospace are solid.** CARC through Snakemake and SLURM, pinned GPU types, SU accounting. LVS's 40 m requirement, JPL's 100° sun envelope, and a dose-response that crosses it (`docs/benchmarks/stress-sun-scan.md`).
- **Splatting is built but sidelined.** The Jezero splat matches held-out mesh renders to at worst 32.6 dB PSNR (`docs/benchmarks/splat.md`). ADR 0002 made it only a renderer stand-in, and it is only 21% cheaper.
- **Deep learning is the real gap.** The navigator uses a pretrained matcher. Fine-tuning (M4) is the only place the project trains a network.
- **The stress scan gave fine-tuning a measurable target.** Error grows steadily with sun difference, mostly as a shift away from the sun (mean shift 4.67 → 9.93 m from 39° to 112.6°).

## Decision

1. **The headline question becomes: how does lighting mismatch degrade Mars terrain-relative navigation, and does a fine-tuned matcher fix it?** The parts:
   - a splat-rendered Jezero, shown in the viewer;
   - HPC-scale plain Monte Carlo that gives the certified upper bound on P(fail) (ADR 0004, decision 1);
   - the sun-difference dose-response as the benchmark;
   - fine-tuning judged on that benchmark;
   - a viewer that shows the shift happening.
2. **Multifidelity is one bounded experiment with a go/no-go**, not the headline. The cheap levels are re-selected with the sun varying, against plain MC *and* sun-stratified MC. If no level beats both, the honest result is that multifidelity doesn't pay here, and M5 shrinks to writing that up.
3. **Fine-tuning moves up** to M4, worked on alongside the go/no-go. It is the deep-learning pillar, not an optional extra.
4. **Sourced errors 2–4 (#37–#39) are built separately and scanned once, together.** This amends ADR 0003 decision 6, which asked for one pre-registered scan per error. Every laptop check predicts no failures from any of them (comments on #37–#39). So separate scans would mainly buy an error breakdown, at two extra cluster jobs. Each one is still smoke-tested on the laptop, and a scenario input set to zero must reproduce today's result. The combined scan is pre-registered as before.
5. **Before #38 is built, the user settles whether the 40 m line counts horizontal or 3D error.** #38's check found focal-length error goes mostly into height. JPL's figures suggest the requirement was horizontal (an inference). Fidelis currently counts 3D.
6. **One 10,000-descent realistic run gives both the P(fail) bound and the brute-force reference** for mean error. #21 keeps the repeated estimator trials against that reference.

## Consequences

- **Milestones are restructured:**
  - M3: realistic model and certified bound
  - M4: fine-tuning for lighting robustness
  - M5: multifidelity on navigation error (go/no-go)
  - M6: viewer and write-up
- **#28 (sigmoid steepness) closes.** There are no failures to smooth. **#20 (splats for more suns) closes.** The viewer uses the existing splat.
- **Fine-tuning's training data needs two things the old plan lacked.** Render-to-map pairs with a sun difference, on de-shaded textures (#19's method), or the matcher never sees the effect it's meant to fix. And possibly the real CTX mosaic for each training region, as Jezero now uses (#36), instead of a map averaged from the same photo, the gap ADR 0003 found. Both are recorded on #16.
- **ADR 0002 §4 stays open:** re-check levels on the failure indicator if failures ever occur.
- **No new number is claimed here.** Whether fine-tuning shrinks the shift, and whether any cheap level helps, are both unmeasured.
