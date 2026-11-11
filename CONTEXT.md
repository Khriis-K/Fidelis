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

**Failure**:
A descent whose fix is more than r = 40 m from the truth in the map frame, or that gets no fix at all (counted separately). r is LVS's requirement and isn't tuned to a target rate (ADR 0003).
_Avoid_: Miss (unqualified), r chosen for a 2–10% rate

**Map frame**:
Positions measured relative to the matching map, after removing one whole-map offset between the map and the rendered scene. The offset is measured independently, by registering the two orbital images, and reported on its own.
_Avoid_: True frame (when meaning this), map-corrected error

**Sourced error**:
An error source added to the scenario model because real landings face it, with its size quoted from a source. Anything harsher is a **stress case** and is never reported as the realistic failure rate.
_Avoid_: Difficulty knob, lever (in write-ups)

**Knowledge error**:
How wrong the lander's own belief about its position is, as opposed to how far it actually landed from its aim point. Not modelled in Fidelis.
_Avoid_: Prior error, navigation error
