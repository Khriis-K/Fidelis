# 3. Failure is a 40 m miss in the map frame, and failures come only from sourced errors

Date: 2026-11-11. Status: accepted (all decisions by the user, in a grilling session on 2026-11-11).

## Context

The failure scan (`docs/benchmarks/failure-scan.md`) found 0 failures in 300 mesh descents at the provisional r = 12 m. The worst error was 6.70 m. Errors are a per-sun offset of 1.5–4.7 m plus about 1 m of scatter, so more descents of the same kind won't produce failures. Multifidelity estimation, and the pilot (#13), need failures to exist.

The obvious fix, turning up a difficulty dial until the navigator breaks, would make the failure rate something we chose rather than measured. The research note `docs/research/lvs-error-sources.md` shows a better reason the scan was clean: Fidelis gives its navigator a perfect world. The descent images and the matching map come from the same photo, the camera model is exact, and the attitude is the true one. The real LVS matched against a different orbital map (the CTX mosaic), through dust haze, with an imperfectly calibrated camera. Its only documented field-test misses came from map and image looking different (Johnson et al. 2020, via the research note).

Two other things were unsettled. The threshold r was a placeholder (12 m), and #28 planned to choose it so that 2–10% of descents fail. Measuring error against the true position would also count any whole-map registration offset as navigation failure, which LVS didn't do: its map could be off by 150–300 m because the hazard map shared the same frame (Johnson et al. 2017 p.10, via the research note).

## Decision

1. **Realism is the test for every error source.** An error goes into the scenario model only if real landings face it. Nothing is added only to make the navigator fail. Anything harsher than realistic is labelled a stress case and never reported as the realistic failure rate.
2. **Sizes come from sources.** Each error's size is quoted from a source. Where none exists, the error is left out, or a range of sizes is tested and labelled as a range.
3. **Failure is a miss beyond r = 40 m**, LVS's requirement (40 m within 10 s, 99% of the time; Mohan et al. 2021 p.4, via the research note). A descent with no fix also fails, and is counted separately.
4. **Error is measured in the map frame.** A single whole-map offset between the matching map and the rendered scene is measured once, independently, by registering the two orbital images to each other. It is never estimated from the descent results, since that would also hide navigator bias. The offset is removed before counting failures and reported as its own number. Local map distortion stays in the error, because that is real navigation error.
5. **r is not tuned to a target failure rate.** #28's plan to set r so that 2–10% of descents fail is dropped. The full error distribution is still stored and reported, so other lines (12 m included) can be read off, labelled as exploratory.
6. **Sourced errors go in one at a time, regardless of the failure count:**
   1. Match against the real LVS map (the CTX orthomosaic) instead of a block average of the rendered orthoimage.
   2. Dust haze, over the tested optical-depth range.
   3. Camera calibration error. Only a pre-calibration figure was found, so this is likely a labelled range.
   4. Attitude-knowledge error.

   Each step is a pre-registered failure scan with the same design as the first: 300 fresh seeds, the mesh renderer, one A40. The realistic failure rate is whatever the full set gives. Stopping early because a convenient count appeared would be the same bias in reverse.
7. **The stress-case question is deferred** until the full realistic set has run.

## Consequences

- **#28 shrinks** to choosing the smoothed indicator's sigmoid steepness. r is fixed here. Its line "the LVS 40 m target plays no part in choosing r" is reversed.
- **#13 (pilot) waits** for the realistic scenario model. It still needs working cheaper fidelity levels (the sweep found none) and #12.
- **The failure rate may get smaller, not bigger.** 40 m is more lenient than 12 m. If the full realistic model still almost never fails, that is the honest answer to "how often does it fail?", and decision 7 comes back.
- **Map frame vs truth frame** changes the meaning of every `error_m` reported so far, since those were all against truth. With the self-made map, the offset is zero by construction, so earlier numbers are unaffected.
- **The spec, README and EXPLAINER** must describe failure as a 40 m miss in the map frame.
- **Dust haze and calibration** need new rendering and camera-model work. Their sizes come from `docs/research/lvs-error-sources.md`, which lists what wasn't found.
