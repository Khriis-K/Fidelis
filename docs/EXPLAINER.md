# Fidelis — Explainer

A refresher for explaining Fidelis to recruiters. Plain English first, key technical terms kept.

> **Status:** early development. The Python environment and a smoke test of every core library exist (`docs/environment.md`). All five pass on the dev laptop and on L40S and A100 GPUs on USC's CARC cluster, run through the same Snakemake workflow. The workflow also builds the Jezero test scene: it downloads the USGS data, cuts a 2 km tile, makes the CTX-like matching map, and turns the tile into a textured 3D terrain mesh (checks in `docs/benchmarks/terrain-mesh.md`). Everything else below describes the *planned* system from `docs/fidelis-spec.md`, and nothing has been measured yet.

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

**JIT (just-in-time) compilation** — building code into machine instructions the first time it's needed, on the user's own machine, instead of shipping it prebuilt. *In Fidelis:* gsplat compiles its GPU code the first time it renders (minutes, not seconds), so its build has to work on every machine we run on. That's where most of the Windows setup trouble came from.

**Pinning and forking dependencies** — pinning means locking a library to one exact version or commit so every machine installs identical code. Forking means keeping your own copy of a library to carry fixes upstream hasn't released yet. *In Fidelis:* `uv.lock` pins everything; MXMCPy and gsplat are pinned to GitHub commits, and gsplat comes from our fork until its Windows fixes are released. A fork is a maintenance cost, so the docs say exactly when to drop it.

**SLURM, Snakemake, and profiles**: SLURM is the job queue on a shared computing cluster. You ask it for resources (one GPU of a given model, some memory, a time limit, and which account to bill), and it runs your job when they're free. Snakemake is a workflow manager: you write each step as a rule saying what it needs, and Snakemake works out what to run and submits it. A profile is a settings file that says *where* the rules run. *In Fidelis:* the same rules run on the laptop GPU (`profiles/local`) or on USC's CARC cluster (`profiles/carc`) without changing the workflow. The cluster profile always names the billing account (otherwise the SLURM plugin guesses, and could bill another group) and pins the GPU model, because cost measurements are only comparable on one GPU type.

**Why failure flags correlate worse than errors** — turning a continuous error into a pass/fail flag throws information away, and a biased cheap model puts its pass/fail cutoff in a different place from the expensive one. *In Fidelis:* a synthetic prototype showed this clearly: cheap and expensive errors that tracked each other closely gave pass/fail flags that agreed much less. That is why Fidelis plans a smoothed (sigmoid) failure indicator, and why the pilot study measures correlation on the failure quantity itself. It still needs confirming on real renders.

**DTM, orthoimage, and the matching map**: a DTM (digital terrain model) is a height map, one elevation per pixel. An orthoimage is an overhead photo corrected so every pixel lands where it belongs on the map. *In Fidelis:* USGS's Jezero DTM (1 m pixels) and orthoimage (0.25 m pixels, from HiRISE) are the ground truth for the test scene. The navigator doesn't match against that sharp photo, though. Perseverance's Lander Vision System matched against a map from the coarser CTX camera, about 6 m per pixel, so Fidelis averages the orthoimage down 24× to match. A sharper map than the real one would make the navigator look better than it is.

**Checksums and pinning data**: a checksum (here SHA-256) is a short fingerprint of a file's exact bytes. Change one byte and the fingerprint changes. *In Fidelis:* USGS doesn't publish checksums for the Jezero mosaics, so the first download records its own, and every later download on any machine has to match it. That catches a truncated download or a file quietly replaced upstream. The catch is that it trusts the first download ("trust on first use").

**Textured mesh, UV coordinates, and float32 precision**: a terrain mesh is a surface made of triangles, one corner (vertex) per height sample. A texture is an image draped over it. Each vertex carries a UV coordinate, its position in the image from 0 to 1 across and up, which says which part of the image paints that spot. GPUs store positions as float32, which keeps about 7 significant digits. *In Fidelis:* the Jezero mesh has a vertex at every 1 m DTM pixel (4 million of them), so it matches the DTM exactly at its samples. It's built around the tile centre: in raw map coordinates (around 4.6 million metres east), float32 could only place vertices to the nearest half metre.

## Challenges and how we solved them

<!-- Format: ### Title — then: the problem, how we noticed, what we did, what it taught us. -->

### Getting five GPU libraries to agree on one brand-new GPU

**The problem.** The dev laptop has an RTX 5070, a Blackwell-generation GPU new enough that much of the scientific Python stack hadn't caught up. Fidelis needs PyTorch, Mitsuba, gsplat, LightGlue, and NASA's MXMCPy all working in one environment, and two of them broke.

**How we noticed.** The first ticket was a smoke test: one command that makes each library do a small piece of real GPU work and checks the answer against something known (a CPU result, a known camera projection, a known image warp). An import-only check would have missed both failures below.

**What we did.** MXMCPy's PyPI release is from 2020 and calls a PyTorch function that no longer exists. NASA had fixed it on GitHub in 2022 but never published the fix, so we install that exact commit instead. gsplat was harder. It compiles its own CUDA code on first use, and it's developed mostly on Linux, so on Windows with the new CUDA 13 its build failed five different ways, one after another. Every fix already existed, but scattered: two in gsplat's unreleased main branch, and the rest in two separate pull requests from outside contributors that hadn't been merged. We forked gsplat, combined the pieces on one branch (after testing the last fix locally before publishing anything), and pinned Fidelis to that exact commit until an official release catches up. Two more surprises: gsplat throws away its whole compiled cache whenever its build settings change, and on Windows PyTorch needs the Microsoft compiler on *every* load, not just the first build. So Fidelis now loads gsplat through one small function that pins the settings and finds the compiler itself. We also switched off a gsplat module for fisheye-style cameras that we don't use, because its files were taking hours to compile. We also picked the PyTorch build whose CUDA version exactly matches the laptop's CUDA toolkit, which heads off the most common way gsplat's build fails.

**What it taught us.** Test that a library *works*, not that it *imports*. Check upstream before patching: every fix we needed already existed, just unreleased. And when you must carry fixes, pin them to an exact commit and write down when to drop them.

### A cluster job that waited hours to fail in seconds

**The problem.** On the university cluster, a GPU job waits in a shared queue, sometimes for hours, before it runs. The first cluster smoke test waited its turn in the queue, ran for 40 seconds, and failed one of its five checks: gsplat found no CUDA compiler.

**How we noticed.** The job log had a one-line module-system error near the top: the CUDA module "exists but cannot be loaded as requested". It needed two other modules loaded first. Snakemake runs a job's command even when its module load fails, so the job carried on without the compiler instead of stopping.

**What we did.** `module spider` named the missing prerequisites, and we checked the corrected module line on the login node (no GPU, no queue) before resubmitting. Then we made that check automatic: a `preflight` step that Snakemake runs on the login node before submitting any GPU job, with the same modules. It confirms the CUDA compiler is there, matches PyTorch's CUDA version, and can compile a tiny kernel. If it fails, nothing is submitted. The next run got further and hit two new walls inside gsplat's build: a line in our own Windows fork that Linux's compiler rejects, and a system Python missing its C headers. We reproduced the first in a Linux environment on the laptop before touching the cluster again, and added a header check to the preflight for the second. Instead of spending another GPU slot to test the fix, we compiled gsplat on a CPU-only debug node, which gets through the queue in minutes: the build doesn't need a GPU as long as you tell it which chip to target. The next GPU job reused that build and passed all five checks in 38 seconds.

**What it taught us.** Anything checkable without the expensive resource should be checked before asking for it. Our docs had already warned about exactly this risk, and we submitted anyway, so a warning in a doc isn't a check. A step that runs every time is.

### Two programs that disagree about which way is up in an image

**The problem.** The terrain mesh needs to work in two places: Mitsuba, the renderer that makes all of Fidelis's training images, and ordinary 3D viewers, which we use to eyeball it. A texture coordinate of v = 0 means "bottom row of the image" to MeshLab, Blender and three.js, but "top row" to Mitsuba's texture lookup. Pick either convention and one of them shows the terrain with its texture mirrored north to south: craters in the picture land on flat ground in the geometry. Nothing crashes, and a quick look might not catch it, because Mars terrain looks plausible either way.

**How we noticed.** Before writing the mesh code, we gave Mitsuba a two-pixel image and a hand-made triangle and asked which pixel it sampled at a known coordinate. Its PLY loader keeps coordinates as written, and its textures count rows from the top.

**What we did.** The mesh file follows the standard convention (v = 0 at the bottom), so viewers display it correctly. The one function that hands the mesh to Mitsuba flips v. Tests check registration through Mitsuba's own texture lookup, on a fake image where each pixel's value encodes its own row and column, so the value Mitsuba returns says exactly where it looked. On the real tile, a straight-down render matches the orthoimage (correlation 0.972, no measurable shift), and flipping it drops the correlation to −0.11.

**What it taught us.** When two tools share a file format, check the convention each one reads, not the one the format spec implies. And test through the consumer that actually matters (here, the renderer), on data built so a mistake can't look plausible.

## Headline results

None yet. Numbers go here only once they're committed in `docs/benchmarks/` or `docs/adr/`, each with its caveat.

## Honest limitations

<!-- What the results don't show, and why. -->

- **The CTX-like map is an approximation.** It's the HiRISE orthoimage averaged in 24×24 blocks. That removes detail CTX can't see, but it doesn't model CTX's optics, noise, or different viewing geometry, and it isn't the real onboard LVS map.

- **The terrain is only as detailed as the 1 m DTM.** Rocks and ridges smaller than a metre exist in the photo texture but not in the geometry, so they don't cast their own shadows when relit. The texture also still holds the shadows from the moment HiRISE took the picture; removing those is a separate step (issue #19).
