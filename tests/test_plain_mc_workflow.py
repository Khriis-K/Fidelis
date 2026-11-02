"""The plain Monte Carlo and match-picture rules, checked by dry runs (no GPU, no SLURM)."""

import pytest
from test_workflow import CARC_GPU_TYPES, dry_run, planned_jobs

TARGETS = ("plain_mc", "results/plain_mc/matches/seed_12.png")


@pytest.mark.parametrize("profile", ["local", "carc"])
def test_one_command_plans_the_monte_carlo_run_and_a_match_picture_on_a_gpu(profile):
    jobs = planned_jobs(dry_run(profile, *TARGETS))

    for rule in ("plain_mc", "match_viz"):
        assert jobs[rule]["gpu"] == "1", rule


def test_carc_pins_the_gpu_model_for_both_since_their_costs_are_recorded():
    jobs = planned_jobs(dry_run("carc", *TARGETS))

    for rule in ("plain_mc", "match_viz"):
        assert jobs[rule].get("gpu_model") in CARC_GPU_TYPES, rule
