# Test verification: stress-sun plumbing (#43)

spec-verify, 2026-11-13. The spec is issue #43's acceptance criteria and `stress-sun-scan.md`. For each criterion, one targeted mutation reintroduced the bug its test guards against. The test had to pass on the real code and fail on the mutant. Each file was restored and checked byte for byte before the next mutation.

| Criterion | Test (`tests/test_stress.py` unless named) | Mutation | Result |
|---|---|---|---|
| Stress suns stay out of the realistic list | `test_no_stress_sun_is_in_the_realistic_list` | `config_suns` appends `stress_suns` | VERIFIED |
| Stated sun differences are the 3D angle | `test_each_stress_suns_stated_difference_is_its_computed_one` | `sun_difference_deg` returns the bearing difference only | VERIFIED |
| A stress run renders from the stress list | `test_a_stress_run_renders_from_the_stress_list_and_a_realistic_run_from_the_realistic_one` | `suns_for` always returns the realistic suns | VERIFIED |
| Pinning refuses a stress sun past the list | `test_a_stress_sun_past_the_list_is_refused` | range check removed | VERIFIED |
| The de-shaded texture alone makes a stress case | `test_a_realistic_sun_on_the_deshaded_texture_is_a_stress_case` | `is_stress_case` ignores the texture | VERIFIED |
| Estimates refuse to pool stress with realistic | `test_an_estimate_refuses_to_pool_stress_and_realistic_records` | label check in `estimate()` removed | VERIFIED |
| Estimates refuse to pool two stress suns | `test_an_estimate_refuses_to_pool_two_stress_suns` | `stress_label` drops the sun | VERIFIED |
| Realistic records still pool across suns | `test_realistic_records_from_before_and_after_the_labels_still_pool_across_suns` | `stress_label` keeps the sun for realistic records too | VERIFIED |
| A stress estimate is labelled in its summary | `test_a_stress_estimate_is_labelled_as_one_in_its_fields_and_its_summary` | summary label never added | VERIFIED |
| Every simulator record carries its labels | `test_a_stress_simulator_labels_even_a_no_fix_record` | `sun_list` left off the record | VERIFIED |
| The mesh renders the texture it's given | `tests/render/test_mesh_renderer_texture.py` (GPU) | `MeshRenderer` ignores its texture | VERIFIED |

The criterion "a stress run renders from the stress list" had no test until this pass. The choice sat inside `build_simulator`, which needs the GPU and the full scene. It was pulled out into `fidelis.stress.suns_for` and tested there.

## Verification results
11/11 criteria VERIFIED. None VACUOUS, BROKEN-TEST or UNVALIDATABLE.
