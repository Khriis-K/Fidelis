# Fidelis — Explainer

A refresher for explaining Fidelis to recruiters. Plain English first, key technical terms kept.

> **Status:** planning. Everything below describes the *planned* system from `docs/fidelis-spec.md`. Nothing has been built or measured yet.

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
7. **Show it.** A web viewer replays descents over the terrain and shows landing-estimate scatter building up.

## Key concepts

<!-- Format: **Term** — plain-English meaning. *In Fidelis:* why it matters here. -->

**Lander Vision System (LVS) and the CTX map** — the camera-based navigation Perseverance used at Jezero. It matched descent photos against an orbital map built from MRO's Context Camera (CTX), which is much coarser than the descent photos. *In Fidelis:* our matching map is deliberately downsampled to CTX-like resolution so the navigator faces the same sharp-photo-to-coarse-map gap as the real mission. LVS's 40 m position-knowledge target appears as a reference line on our results, but it doesn't set our threshold.

**Pilot study** — a small trial run that measures each fidelity level's cost and how closely the levels track each other (their covariance). MXMCPy treats those measurements as exact and uses them to split the sample budget and to predict its own accuracy. *In Fidelis:* a pilot that's too small makes MXMCPy overconfident. Its predicted variance was off by about 100% at 10 pilot samples, even on an easy test case (MXMCPy NASA report, Table 3), and small pilots tend to bias it low (Stanley et al. 2026). Rare failures make this worse, so our pilot grows in stages until its conclusions stop changing. See `docs/research/pilot-study-sizing.md`.

## Challenges and how we solved them

<!-- Format: ### Title — then: the problem, how we noticed, what we did, what it taught us. -->

## Headline results

None yet. Numbers go here only once they're committed in `docs/benchmarks/` or `docs/adr/`, each with its caveat.

## Honest limitations

<!-- What the results don't show, and why. -->
