# 6. Close Fidelis with the bound, the no-go and the error-vs-sun report

Date: 2026-11-15. Status: accepted (the user, in session, 2026-11-15).

## Context

ADR 0005 left two questions open: whether multifidelity pays (M5) and whether fine-tuning fixes lighting mismatch (M4). As of this ADR:

- **The bound is in.** 1 failure in 10,000 realistic descents, under 0.057% at 95% confidence (`docs/benchmarks/certified-bound.md`). The one failure came from an exposure brighter than flight, so the bound is pessimistic about exposure (`docs/benchmarks/exposure-brightness.md`).
- **Multifidelity is a no-go** (#48, `docs/benchmarks/sun-sweep.md`). So the repeated estimator trials and the validation plan (#11–#14, #21, #29) have nothing to validate.
- **Fine-tuning can't show up in the headline result.** From 1 failure in 10,000, a better matcher could at best reach 0, which no affordable run can tell apart. What it could shrink is the error's growth with sun difference, and that error stays well inside the 40 m line (`docs/benchmarks/stress-sun-scan.md`). It also wouldn't touch what the simulator can't produce (cast shadows, real cameras, suns past 113°). Fixing an accuracy cost that never threatens the requirement isn't worth four tickets and the GPU time.
- **The project spread too thin.** Four areas in one project meant most of the effort went into plumbing, not depth in any one area. The user's next project focuses on Gaussian splatting alone.

## Decision

1. **Fidelis closes with three results:** the certified bound (#47), the multifidelity no-go (#48), and the error-vs-sun-difference report (#22).
2. **Fine-tuning (M4) is dropped:** #15–#18 close as wontfix. The headline question becomes "how does lighting mismatch degrade terrain-relative navigation?", answered with the pretrained matcher. Whether fine-tuning would shrink the shift stays unmeasured, and the write-up says so.
3. **Multifidelity validation (M5) is dropped:** #11–#14, #21 and #29 close as wontfix. The free formula's control-variate result (about 0.2× plain MC's variance on mean error) is reported as-is in `sun-sweep.md`, without repeated trials.
4. **The showcase shrinks to the README and a short video (#27).** The viewer, replay, deploy, results format and showcase frames (#23–#26, #30) close as wontfix. #27 is no longer blocked by #26. Its video uses frames the pipeline can already render, and its novelty wording drops the multifidelity claim.
5. **#22 drops the fine-tuned curves** and is built from saved results only, with no new GPU runs.
6. **#31 (gsplat loader on non-ASCII paths) closes as wontfix.** It never affected the dev laptop.

## Consequences

- No more cluster jobs for Fidelis. The remaining work is laptop-only: #22, then #27.
- The EXPLAINER's pitch, results and limitations are updated to say fine-tuning and multifidelity validation were scoped out, and why.
- ADR 0005 decisions 1 and 3 are superseded by decision 2 here. Its decision 2 has run its course: the go/no-go said no, and M5 became the write-up.
- No new number is claimed here.
