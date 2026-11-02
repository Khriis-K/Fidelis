# Fidelis — Explainer

A refresher for explaining Fidelis to recruiters. Plain English first, key technical terms kept.

> **Status:** early development. The Python environment and a smoke test of every core library exist (`docs/environment.md`). All five pass on the dev laptop and on L40S and A100 GPUs on USC's CARC cluster, run through the same Snakemake workflow. The workflow also builds the Jezero test scene: it downloads the USGS data, cuts a 10 km tile around the landing site, makes the CTX-like matching map, and turns the tile into a textured 3D terrain mesh at 4 m detail, in true ground metres (checks in `docs/benchmarks/terrain-mesh.md`). A path-traced renderer draws that mesh as the descent camera would see it, from any pose and under any sun. From the 4.2 km and 2.2 km altitudes where Perseverance's vision system took its pictures, a 1024 × 1024 image at 64 samples per pixel takes a median 0.053–0.058 s on the laptop GPU and uses about 2.2 GB (2,102 MiB) of its memory, once its GPU code is compiled (the first render in a fresh process took 0.60 s; only nadir views on one GPU measured; `docs/benchmarks/mesh-render.md`). The navigator exists too: it matches a descent image against the 6 m CTX-like map and solves for the camera's position. In perfect conditions (straight down, north up, overhead sun, no noise) it locates within 0.6–1.9 m from both ends of that altitude range, on one test position (`docs/benchmarks/navigator.md`). Everything else below describes the *planned* system from `docs/fidelis-spec.md`, and nothing else has been measured yet.

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

**Textured mesh, UV coordinates, and float32 precision**: a terrain mesh is a surface made of triangles, one corner (vertex) per height sample. A texture is an image draped over it. Each vertex carries a UV coordinate, its position in the image from 0 to 1 across and up, which says which part of the image paints that spot. GPUs store positions as float32, which keeps about 7 significant digits. *In Fidelis:* the Jezero mesh has a vertex every 4 m (6.27 million of them over 10 km), each one the average of 16 of the DTM's 1 m pixels, and it matches that averaged DTM exactly at its samples. It's built around the tile centre: in raw map coordinates (around 4.6 million metres east), float32 could only place vertices to the nearest half metre.

**Path tracing and samples per pixel**: path tracing makes an image by following light rays backwards from the camera into the scene, bouncing off surfaces, and checking at each hit whether the sun is visible. Shadows come out right because they're computed, not painted on. Each pixel averages many random rays (samples per pixel, spp); fewer samples is faster but grainier. *In Fidelis:* Mitsuba 3 path-traces the terrain mesh. At 64 spp a descent image takes a median 0.053–0.058 s on the laptop once warmed up, and is within 0.9–2.2% (RMS) of a 1024-spp render on the 10 km mesh. With the same random seed the image repeats to within 1e-6.

**Map projections and the east–west stretch**: a map projection flattens a sphere onto a grid of metres, and no flat grid keeps every distance right. The USGS Jezero maps use an equirectangular projection, which keeps every line of latitude as long as the equator. Away from the equator that stretches east–west distances by 1/cos(latitude). *In Fidelis:* at Jezero (18.4° N) one map metre east–west is 0.949 ground metres, so the first terrain mesh was 5.4% too wide east–west, with east–west slopes too gentle and shadows cast east or west too long. The mesh now multiplies east–west distances by cos(latitude) and records the factor, so poses and distances are in true ground metres.

**Area averaging (downsampling)**: making an image coarser by replacing each block of pixels with their average, the way a camera pixel collects all the light falling on it. *In Fidelis:* the 10 km tile is four times coarser than the raw elevation data (4 m) and eight times coarser than the raw photo (2 m), because a full-detail 10 km mesh filled the laptop GPU. At the 4.2–2.2 km altitudes the descent camera images from, one of its pixels covers 4.3–8.2 m of ground, so the coarser tile is no coarser than what the camera sees (the 4 m mesh is about one height sample per camera pixel at 2.2 km).

**Camera pose and intrinsics**: a pose says where a camera is and which way it points (a position plus a rotation). Intrinsics say how it maps directions to pixels: focal length and the principal point, where the optical axis hits the image. *In Fidelis:* the renderer, splat training and the navigator all have to mean the same thing by a pose, or a perfect navigator would still report errors. Everything uses OpenCV's convention (x right, y down, z forward), written down in `docs/camera-conventions.md`. The descent camera copies Perseverance's LCAM: 1024 × 1024 pixels, 90° field of view (Maki et al. 2020).

**Seeded random sampling**: a computer's "random" numbers come from a formula started by a number called the seed; the same seed always produces the same numbers. *In Fidelis:* every simulated descent draws its conditions (position, altitude, tilt, heading, noise, blur, exposure, sun angle) from a recorded seed, so any single descent can be regenerated exactly to inspect why it failed. Each condition gets its own random stream, keyed by the seed and the condition's name, so changing how tilt is drawn doesn't silently reshuffle every other condition of an existing seed.

**PnP (Perspective-n-Point)**: given a handful of spots whose 3D positions you know and where each one appears in a photo, PnP works out where the camera must have been and which way it pointed. *In Fidelis:* each matched map keypoint becomes a 3D point (its map position plus the terrain height beneath it), and OpenCV's PnP inside RANSAC turns the matches into the navigator's position fix. Fewer than 10 agreeing matches counts as a failed solve, recorded rather than crashing the run.

**Scale and rotation invariance**: a feature detector is scale-invariant if it finds the same spot whether the image is zoomed in or out, and rotation-invariant if it does so when the image is turned. ALIKED and LightGlue are neither, much. *In Fidelis:* matching held up from a map about twice as sharp as the descent image to one almost 3× coarser, but broke when the map was 4× sharper (`docs/benchmarks/navigator.md`). Turning the camera 90° from north-up broke it too. Perseverance avoided that by rotating each photo using its known attitude before matching; Fidelis will need the same step before the Monte Carlo runs.

**Knowledge error vs true dispersion**: two different ways a lander can be "off". True dispersion is where the vehicle actually is, compared with where it was aimed (Perseverance: about 1.1 km standard deviation before flight). Knowledge error is how wrong the vehicle's *own belief* about its position is (designed for up to 3.2 km; 137 m in flight). *In Fidelis:* the scenario sampler draws the true position, because that decides what ground the camera sees. Knowledge error only matters if the navigator narrows its search around a prior guess, which it doesn't yet.

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

**What we did.** The mesh file follows the standard convention (v = 0 at the bottom), so viewers display it correctly. The one function that hands the mesh to Mitsuba flips v. Tests check registration through Mitsuba's own texture lookup, on a fake image where each pixel's value encodes its own row and column, so the value Mitsuba returns says exactly where it looked. On the real tile, a straight-down render matches the orthoimage: correlation 0.972, and no shift that phase correlation can measure at 2 m pixels (it resolves a fraction of a pixel, not centimetres). Mirroring the render north–south drops the correlation to −0.109. See `docs/benchmarks/terrain-mesh.md`.

**What it taught us.** When two tools share a file format, check the convention each one reads, not the one the format spec implies. And test through the consumer that actually matters (here, the renderer), on data built so a mistake can't look plausible.

### Making the renderer and the navigator agree on what a camera pose means

**The problem.** Fidelis renders descent images with Mitsuba and later asks the navigator (OpenCV's PnP) to recover the camera position from them. If the two disagree about a pose by even a flipped axis or half a pixel, the navigator gets "wrong" answers that are really a bookkeeping error, and the failure estimate inherits it. Mitsuba's camera frame isn't OpenCV's: it points +x left and +y up, and counts pixel centres from 0.5 instead of 0.

**How we noticed.** Before trusting the conversion, a test renders small glowing spheres from a tilted, rolled camera with its optical centre deliberately off the image centre, and requires each sphere's image to land within 0.1 pixel of where OpenCV's own projection says it should. The first run missed by up to 14.4 pixels in a telltale way: every sphere was mirrored about the image centre by exactly the principal-point offset. Mitsuba's `principal_point_offset` setting runs the opposite way to pixel coordinates.

**What we did.** Flipped the sign, and the test now holds to 0.1 pixel for all three spheres. The conventions are written down in `docs/camera-conventions.md` so splat training and the navigator reuse them rather than re-deriving them.

**What it taught us.** Check a convention against the other library's own code (here `cv2.projectPoints`), not against your reading of its docs. And make the test geometry asymmetric, with an off-centre principal point and a rolled camera, so that a mirrored or swapped convention can't pass by symmetry.

### A tile too small, a map too wide, and a render that failed without an error

**The problem.** The first Jezero tile was 2 km across. The descent camera's 90° view sees twice its altitude across, so above about 1 km it saw past the edge of the terrain. Perseverance's Lander Vision System took its pictures from 4.2 km down to 2.2 km, entirely above that. Separately, the mesh was 5.4% too wide east–west.

**How we noticed.** Building the renderer meant picking test altitudes, and the 2 km tile only fit views below 1 km. That sent us to NASA/JPL's landing press kit for the real altitudes. While working out how big a tile we'd need, converting the map's coordinates back to latitude and longitude showed that 1000 map metres east–west is only 948.6 m of ground at Jezero. Then, trying a 10 km tile at 2 m detail, the laptop GPU filled up and Mitsuba returned images with streaked rows and a black wedge, without raising any error.

**What we did.** The tile is now 10 km, the most valid data centred on the landing site allows, averaged down to 4 m geometry and a 2 m photo texture, no coarser than a descent-camera pixel at those heights. It renders in about 0.055 s per image using about 2.2 GB of GPU memory. The mesh multiplies east–west distances by cos(latitude), and its registration check now compares in ground metres: the render matches the photo with correlation 0.978, against 0.529 if the photo is left uncorrected. Every benchmark render now goes through a check that fails on any row or column of exact zeros. That catches a render that goes black along whole lines, and a view that runs off the edge of the terrain. We never ran it on the corrupt 2 m render itself, so whether it would have caught that one is untested.

**What it taught us.** Check the operating envelope (here, the real mission altitudes) before sizing the test scene. Never assume map units are ground units. And when a GPU tool can fail silently, add a cheap check on its output instead of trusting the absence of an error.

### A map too small for a tilting camera

**The problem.** Each simulated descent tilts the camera and shifts it sideways at random, as Perseverance's did on its parachute. But the terrain only exists inside our 10 km tile, so a view that reaches past the edge renders black ground, and the navigator would "fail" because the map ran out, not because navigation is hard. That would quietly inflate the failure probability.

**How we noticed.** The sampler refuses any configuration whose worst-case draw could see past the tile. Working that case through properly (the image corner reaches about 1.4 times farther than its edge, the camera can turn so a corner points outward, and lower ground widens the view) left only a few hundred metres of slack, not the 2.5 km a straight-down estimate suggested. Research into the real descent then showed Perseverance's camera swung about −12° to +5° off vertical, and pre-flight simulations averaged 7–18°. Our map fits 3°.

**What we did.** We shipped the sampler with tilt capped at 3° and offsets at ±400 m, labelled in the config as limits of the map, not of Mars, and kept the researched values next to them. We checked whether a bigger map exists: the sharp HiRISE data allows at most about 12.4 km even off-centre (measured on a 20 m grid, so an upper bound), but USGS's coarser CTX terrain covers a hole-free 27.7 km square on the same grid. Its photo is the map Perseverance's vision system actually carried. Its heights differ from HiRISE's by a median of 0.5 m, but individual spots differ by several metres (5th–95th percentile −5.8 to +7.1 m), so the boundary needs blending. Building a wider scene with a CTX outer ring is its own ticket (#33); when it lands, widening tilt is a config change.

**What it taught us.** Check the worst case, not the typical one, and check it with the real geometry: the corner of the image, not its edge. And keep distributions in config with their sources beside them, so a constraint imposed by our tooling is visible as such rather than mistaken for physics.

### A coarse map that wasn't the problem, and a library that quietly shrank it

**The problem.** The project's biggest early worry was that the 6 m CTX-like map would be too blurry to match against sharp descent images, the way it is for the real mission. We needed to know where matching breaks before building anything on top of it.

**How we noticed.** Two surprises. First, LightGlue's ALIKED wrapper shrinks every image to 1024 pixels by default, so our 1670-pixel map would have been matched at about 10 m per pixel without any warning; reading its source caught it before any number was taken. Second, extracting features from the sharpest test map ran the 8 GB laptop GPU out of memory: the network builds a 128-channel copy of the whole image.

**What we did.** Both images are now matched at their own resolution, and the map's features are extracted in 1024-pixel tiles, once per map. Then a sweep from a 2 m map to a 12 m one, at the top and bottom of the altitude range where Perseverance's vision system worked, showed the 6 m map works fine (within 0.9 m, about 2,000 agreeing matches). The one failure was the opposite of the worry: the *sharpest* map against the *highest*, blurriest descent image. What matters is that map and photo show the ground at a similar scale, not that the map is sharp. The same check found a bigger problem waiting: a camera turned 90° from north-up fails to match at all.

**What it taught us.** Read a library's defaults before trusting its output. Measure the risk you're worried about instead of guarding against it in advance (the spec's fallbacks for a too-coarse map turned out unnecessary). And test one step beyond the easy case: the heading check took minutes and found the issue that matters next.

## Headline results

None yet. Numbers go here only once they're committed in `docs/benchmarks/` or `docs/adr/`, each with its caveat.

## Honest limitations

<!-- What the results don't show, and why. -->

- **The CTX-like map is an approximation.** It's the HiRISE orthoimage averaged in 24×24 blocks (in two steps: 8×8 into the 2 m texture, then 3×3, rounding to whole grey levels in between). That removes detail CTX can't see, but it doesn't model CTX's optics, noise, or different viewing geometry, and it isn't the real onboard LVS map.

- **The terrain geometry is 4 m, not the DTM's 1 m.** The 10 km mesh averages the 1 m DTM into 4 m cells so it fits on the laptop GPU. Relief smaller than about 4 m exists in the photo texture but not in the geometry, so it doesn't cast its own shadows when relit. A descent-camera pixel at LVS altitudes covers 4.3–8.2 m, but low-sun shadows from small steep features are probably still understated (not measured). The texture also still holds the shadows from the moment HiRISE took the picture; removing those is a separate step (issue #19).

- **Rendered shadows are too dark, and the brightness isn't physical.** The renderer has a sun but no sky, so shadows only get light bounced off nearby terrain; on Mars, dusty sky light fills them in. And the orthoimage's pixel values are tone-matched brightness, not calibrated reflectance, so the renderer's output is relative brightness, not physical radiance.

- **The tile leaves little room at the top of the descent.** The 10 km tile is 9.5 km of ground east–west. A straight-down view from 4.2 km covers 8.4 km of it, so it fits only within about 3.5° of tilt or 550 m of the centre, pointing north-up; turning it more than about 8° swings a corner off the terrain. From 2.2 km there's about 2.5 km of room. The data runs out not far beyond this square, so a bigger tile centred on the landing site would have holes. Because of this, the scenario sampler caps camera tilt at 3° and sideways offset at ±400 m, well below Perseverance's real swing (about −12° to +5°), so descents are probably easier than flight and the failure probability biased low (not yet measured). A wider scene with a coarser CTX outer ring (#33) would lift the cap.

- **The navigator has only been checked in near-perfect conditions.** Straight down, north up, one test position, two suns, pretrained matcher. The descent renders and the map come from the same photo, so they differ only in resolution and shading, which flatters matching. Camera heading isn't handled yet: turned 90° from north-up, matching fails (`docs/benchmarks/navigator.md`). The map's terrain heights come from the same 4 m DTM as the rendered terrain, finer than the real onboard elevation map.

- **The terrain is flat; Mars is curved.** The tile is a flat map, while the real surface falls away from it by 3.7 m at 5 km from the centre. That's below a descent-camera pixel at those heights, and it isn't modelled.
