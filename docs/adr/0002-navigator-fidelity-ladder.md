# 2. The fidelity ladder thins the navigator's map, not the splat

Date: 2026-11-11. Status: accepted (decisions 1–3 by the user; 4–7 delegated to Claude and open to review).

## Context

Issue #10 asked for three fidelity levels built from the splat: fewer Gaussians, lower render resolution, lower SH degree. Profiling one splat descent on an A40 (`docs/benchmarks/descent-profile.md`) found LightGlue matching at 75% of the median cost and the render at 2%. Splat-only levels would cost nearly the same, and that's the multifidelity null case: MXMCPy has nothing cheap to spend samples on. The spec's fallback (`docs/fidelis-spec.md`, "shrink the navigator") applies.

LightGlue's cost grows with both keypoint counts, and the two sides are lopsided: at most 4,096 descent keypoints against 70,726 on the wide map. The real navigator already runs LightGlue's adaptive pruning (`depth_confidence` 0.95, `width_confidence` 0.99 by default), so "turn pruning on" isn't a cheaper level.

## Decision

1. **The real navigator keeps searching the whole map.** It gets no window around a position prior. A real window needs a prior and a knowledge-error model, and Fidelis models neither. A window centred on the true position would hand the navigator the answer it's meant to find.
2. **The cheaper levels use map thinning.** Each one keeps the top-K map keypoints by detector score across the whole map. They get no position window either: a window centred on the truth removes the far-away look-alikes that cause wrong fixes, so a windowed level couldn't fail the way the real one does.
3. **No splat knobs in the ladder.** Every level renders the same full splat and differs only in the navigator. Degrading the render could save at most about 2%, and it can only lower the correlation with the real navigator.

## Consequences

4. **Levels are judged on per-seed position error, plus match and inlier counts as failure precursors.** The failure indicator can't be used yet: all 70 descents so far got a fix within about 6 m, so it's nearly constant and its correlation is noise. Agreeing on metre-level errors doesn't prove agreeing on kilometre-level wrong fixes. **The levels must be re-checked on the smoothed failure indicator once the pilot study (#13) sets r and failures occur.** Seeds where one level fails and another doesn't are the tail signal to watch.
5. **The two cheaper levels' K values come from a CARC sweep** of candidate map sizes on shared seeds. The selection rule is written down before the run, so the result can't be tuned to noise. ADR 0001's GPU jitter caps how closely any two levels can agree.
6. **The descent keypoint cap stays fixed for now.** It's the backup axis if map thinning alone saves too little.
7. **If global top-K bunches up** in textured ground and leaves bland ground empty (seen as low correlation, or cheap-level failures concentrated in bland terrain), switch to top-K per map tile.
8. **Honest limitation:** LVS searched near a prior; our real navigator searches the whole map, so it faces more look-alikes than the flight system did. Our P(fail) describes the global-search navigator, which errs toward overstating failure. Adding knowledge error and a prior window is future work and would change the system under test.
9. **Wording changes.** Fidelis is a navigator-fidelity ladder on a splat-rendered scene, not a ladder of splat render levels. The spec (including its novelty wording), the README, `EXPLAINER.md` and issue #10 need updating to match. The splat's role is the fast stand-in for the mesh renderer.
