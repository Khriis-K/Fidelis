# Fidelis

Estimates the probability that a vision-based Mars descent navigator fails at Jezero, using multifidelity Monte Carlo over simulated descents.

## Language

**Real navigator**:
The system under test: ALIKED keypoints, LightGlue matching against the whole orbital map, then PnP for position. The highest fidelity level is always the real navigator, unchanged.
_Avoid_: High-fidelity model (when meaning the system itself), baseline

**Fidelity level**:
One way of simulating a descent, defined by which navigator variant runs on the shared splat render. Levels differ in per-sample cost and in how closely their per-seed results track the real navigator's.
_Avoid_: Render level, splat level, LOD

**Global search**:
Matching a descent image against every map keypoint, with no window around a position prior.
_Avoid_: Full search, unwindowed

**Map thinning**:
Keeping only the K map keypoints with the highest detector scores across the whole map; the knob that makes a fidelity level cheaper.
_Avoid_: Map window, map crop, subsampling

**Knowledge error**:
How wrong the lander's own belief about its position is, as opposed to how far it actually landed from its aim point. Not modelled in Fidelis.
_Avoid_: Prior error, navigation error
