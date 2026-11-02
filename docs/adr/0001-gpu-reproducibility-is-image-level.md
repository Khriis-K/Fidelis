# 1. On the GPU, a seed reproduces the camera image, not the exact navigation error

Date: 2026-11-02. Status: accepted (the user approved replacing the GPU seam test's error check).

## Context

Story 11 asks that "any individual result can be regenerated and inspected" from its seed, and Seam 1 asked that "the same scenario sample and seed produce the same navigation error". The first plain Monte Carlo run (issue #8) showed that on the GPU the same seed's error moves between runs, by up to 0.839 m so far, even within one process. `scripts/check_gpu_repeatability.py` traced it (`docs/benchmarks/plain-mc.md`):

- Mitsuba's CUDA film adds samples in no fixed order, so the same render differs by about 1e-7 to 1e-6 per pixel from run to run.
- The navigator (ALIKED, LightGlue, PnP + RANSAC) repeats exactly on the same image, within a process and across processes.
- But it amplifies: noise of SD 1e-7 on a fixed image moved seed 10's fix by up to 0.518 m. A change that small decides which keypoints and matches survive.
- Mitsuba's `samples_per_pass = 1` didn't make renders bit-exact, and it cost 2.5-2.7x the render time.

## Decision

Reproducibility on the GPU is promised at the camera image, not at the error:

- **CPU:** the same seed gives the same error. Tested within one process.
- **GPU:** the same seed gives the same camera image within 1e-5. Tested within one process; measured at up to 3.70e-6 across processes. The error regenerates only to within the navigator's jitter.

The GPU seam test checks camera images, not errors. The error check it replaced required 0.05 m and could fail on any run.

## Consequences

- **Story 11 is weakened on the GPU.** Rerunning a seed, for example for a match picture, shows the same image and very nearly the same matches, but its error can differ from the record by up to the jitter. The record, not the rerun, is the result. Records written by `fidelis-plain-mc` say this in its docs. Story 11 needs a matching amendment in the issue tracker.
- **The navigation error is a noisy function of the inputs.** That noise is a property of the system under test, not of the simulator. Treat it as part of the quantity being estimated, not as a bug to engineer away.
- **The pilot study (#13) should measure the jitter per fidelity level**, by rerunning some seeds, before it sets r. Samples within the jitter of r are partly random. The jitter also limits how closely two fidelity levels' errors can agree, and the multifidelity estimators' savings depend on that agreement.
- **Not ruled out:** a GPU nondeterminism in the navigator that a busier GPU would expose. The navigator repeated on the same image in every check so far, but only on this laptop.
