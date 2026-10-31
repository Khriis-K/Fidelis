# Fidelis — Explainer

A refresher for explaining Fidelis to recruiters. Plain English first, key technical terms kept.

> **Status:** early development. The Python environment and a smoke test of every core library exist (`docs/environment.md`). All five pass on the dev laptop; the cluster is untested. Everything else below describes the *planned* system from `docs/fidelis-spec.md`, and nothing has been measured yet.

## The two-minute pitch

**The problem.** Mars landers navigate the last stretch of descent by photographing the ground and matching it against an orbital map. This is called terrain-relative navigation (TRN), and Perseverance used it at Jezero Crater. Before a landing, the team needs to know how often that matching fails. You can't test-land on Mars, so the answer has to come from thousands of simulated descents. Realistic simulations are expensive, though, and cheap ones are biased.

**What Fidelis does.** It builds a photoreal 3D model of Jezero from public NASA/USGS data and flies thousands of randomized simulated descents over it. On each one it runs a learned image-matching navigator and counts how often the navigator ends up too far off. To keep that affordable, it mixes many cheap, low-quality renders with a few expensive, high-quality ones. That technique is multifidelity Monte Carlo, run with NASA Langley's MXMCPy library. Then it proves the cheaper estimate is right by checking it against a brute-force answer.

**Why it's interesting.** It ties together three skills on one problem NASA and JPL actually have: 3D reconstruction (Gaussian splatting), deep learning (learned feature matching), and uncertainty quantification (multifidelity estimation and reliability analysis).

## How it works

1. **Build the scene.** Download a USGS elevation map (DTM) and orthoimage of Jezero, then turn them into a textured 3D terrain mesh.
2. **Make it fast to render.** Train Gaussian splat scenes from mesh renders, one per sun angle. Derive three quality levels (fidelities) from each.
3. **Fly descents.** Randomize position, altitude, camera tilt, noise, blur, exposure, and sun angle, then render what the descent camera would see.
4. **Navigate.** ALIKED finds keypoints, LightGlue matches them to the orbital map, and PnP solves for position. Record how far off the estimate is.
5. **Estimate failure.** Use MXMCPy to combine all three fidelity levels and estimate how often the error exceeds a threshold *r*.
6. **Prove it.** Compare the result against a brute-force high-fidelity reference, repeated over many independent trials.
7. **Show it.** A web viewer replays descents over the terrain and shows landing-estimate scatter building up. The centerpiece is the lander camera's view, with the matcher's keypoints and map matches drawn live, so you can watch the network work out where it is.

## Key concepts

<!-- Format: **Term** — plain-English meaning. *In Fidelis:* why it matters here. -->

**Lander Vision System (LVS) and the CTX map** — the camera-based navigation Perseverance used at Jezero. It matched descent photos against an orbital map built from MRO's Context Camera (CTX), which is much coarser than the descent photos. *In Fidelis:* our matching map is deliberately downsampled to CTX-like resolution so the navigator faces the same sharp-photo-to-coarse-map gap as the real mission. LVS's 40 m position-knowledge target appears as a reference line on our results, but it doesn't set our threshold.

**Pilot study** — a small trial run that measures each fidelity level's cost and how closely the levels track each other (their covariance). MXMCPy treats those measurements as exact and uses them to split the sample budget and to predict its own accuracy. *In Fidelis:* a pilot that's too small makes MXMCPy overconfident. Its predicted variance was off by about 100% at 10 pilot samples, even on an easy test case (MXMCPy NASA report, Table 3), and small pilots tend to bias it low (Stanley et al. 2026). Rare failures make this worse, so our pilot grows in stages until its conclusions stop changing. See `docs/research/pilot-study-sizing.md`.

**Keypoints, matches, and RANSAC inliers** — a keypoint is a distinctive spot in an image (a crater rim, a boulder) that a neural network (ALIKED) picks out. A match pairs a keypoint in the descent photo with the same spot on the orbital map (LightGlue). Some matches are wrong, so RANSAC keeps only the group of matches that agree on one camera position. Those are the inliers, and the rest are rejected as outliers. *In Fidelis:* inlier count is the first clue to why a descent failed. Drawing matches, with inliers and outliers in different colors, is both the main debugging view and the project's headline visual.

**JIT (just-in-time) compilation** — building code into machine instructions the first time it's needed, on the user's own machine, instead of shipping it prebuilt. *In Fidelis:* gsplat compiles its GPU code on first import (about 6 minutes on the laptop), so its build has to work on every machine we run on. That's where most of the Windows setup trouble came from.

**Pinning and forking dependencies** — pinning means locking a library to one exact version or commit so every machine installs identical code. Forking means keeping your own copy of a library to carry fixes upstream hasn't released yet. *In Fidelis:* `uv.lock` pins everything; MXMCPy and gsplat are pinned to GitHub commits, and gsplat comes from our fork until its Windows fixes are released. A fork is a maintenance cost, so the docs say exactly when to drop it.

**Why failure flags correlate worse than errors** — turning a continuous error into a pass/fail flag throws information away, and a biased cheap model puts its pass/fail cutoff in a different place from the expensive one. *In Fidelis:* a synthetic prototype showed this clearly: cheap and expensive errors that tracked each other closely gave pass/fail flags that agreed much less. That is why Fidelis plans a smoothed (sigmoid) failure indicator, and why the pilot study measures correlation on the failure quantity itself. It still needs confirming on real renders.

## Challenges and how we solved them

<!-- Format: ### Title — then: the problem, how we noticed, what we did, what it taught us. -->

### Getting five GPU libraries to agree on one brand-new GPU

**The problem.** The dev laptop has an RTX 5070, a Blackwell-generation GPU new enough that much of the scientific Python stack hadn't caught up. Fidelis needs PyTorch, Mitsuba, gsplat, LightGlue, and NASA's MXMCPy all working in one environment, and two of them broke.

**How we noticed.** The first ticket was a smoke test: one command that makes each library do a small piece of real GPU work and checks the answer against something known (a CPU result, a known camera projection, a known image warp). An import-only check would have missed both failures below.

**What we did.** MXMCPy's PyPI release is from 2020 and calls a PyTorch function that no longer exists. NASA had fixed it on GitHub in 2022 but never published the fix, so we install that exact commit instead. gsplat was harder. It compiles its own CUDA code on first use, and it's developed mostly on Linux, so on Windows with the new CUDA 13 its build failed five different ways, one after another. Every fix already existed, but scattered: two in gsplat's unreleased main branch, and the rest in two separate pull requests from outside contributors that hadn't been merged. We forked gsplat, combined the pieces on one branch (after testing the last fix locally before publishing anything), and pinned Fidelis to that exact commit until an official release catches up. Two more surprises: gsplat throws away its whole compiled cache whenever its build settings change, and on Windows PyTorch needs the Microsoft compiler on *every* load, not just the first build. So Fidelis now loads gsplat through one small function that pins the settings and finds the compiler itself. We also switched off a gsplat module for fisheye-style cameras that we don't use, because it was taking over an hour per file to compile. We also picked the PyTorch build whose CUDA version exactly matches the laptop's CUDA toolkit, which heads off the most common way gsplat's build fails.

**What it taught us.** Test that a library *works*, not that it *imports*. Check upstream before patching: every fix we needed already existed, just unreleased. And when you must carry fixes, pin them to an exact commit and write down when to drop them.

## Headline results

None yet. Numbers go here only once they're committed in `docs/benchmarks/` or `docs/adr/`, each with its caveat.

## Honest limitations

<!-- What the results don't show, and why. -->
