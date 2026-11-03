"""The splat rules: views, training, and plain Monte Carlo on the splat, checked by dry runs
(no GPU, no SLURM) and by reading the Snakefile."""

import re
import subprocess

import pytest
from test_workflow import CARC_GPU_TYPES, REPO, dry_run, planned_jobs

SPLAT_RULES = ("splat_views", "splat_train", "plain_mc_splat")
# --forceall: once a run's outputs exist, a plain dry run plans nothing.
TARGETS = ("splat", "plain_mc_splat", "--forceall")


@pytest.mark.parametrize("profile", ["local", "carc"])
def test_one_command_plans_views_training_and_monte_carlo_on_a_gpu(profile):
    jobs = planned_jobs(dry_run(profile, *TARGETS))

    for rule in SPLAT_RULES:
        assert jobs[rule]["gpu"] == "1", rule


def test_carc_pins_the_gpu_model_for_every_splat_job():
    jobs = planned_jobs(dry_run("carc", *TARGETS))

    for rule in SPLAT_RULES:
        assert jobs[rule].get("gpu_model") in CARC_GPU_TYPES, rule


def test_monte_carlo_on_the_splat_pins_the_trained_sun():
    output = dry_run("local", *TARGETS)

    assert re.search(r"fidelis-plain-mc .* --fidelity splat --sun-index 0\b", output)


def rule_block(name: str) -> str:
    snakefile = (REPO / "workflow/Snakefile").read_text()
    block = re.search(rf"^rule {name}:\n((?:[ \t]+.*\n|\n)+?)(?=^\S)", snakefile + "\nEND\n", re.MULTILINE)
    assert block, f"no rule {name} in workflow/Snakefile"
    return block.group(1)


@pytest.mark.parametrize("rule", ["splat_train", "plain_mc_splat"])  # splat_views renders with Mitsuba only
def test_every_rule_that_runs_gsplat_loads_the_cuda_modules(rule):
    # gsplat JIT-builds (or at least checks) its CUDA extension, which needs nvcc on CARC.
    assert "*CUDA_MODULES" in rule_block(rule)


@pytest.mark.parametrize(
    "path",
    [
        "data/jezero/splat/sun0/views/train.npy",
        "data/jezero/splat/sun0/splat.pt",
        "data/jezero/splat/sun0/splat.ply",
    ],
)
def test_splat_checkpoints_and_renders_stay_out_of_git(path):
    # On CARC data/ sits in the clone on /project2, the project storage (docs/carc.md).
    result = subprocess.run(["git", "check-ignore", "--quiet", path], cwd=REPO)

    assert result.returncode == 0, f"{path} is not git-ignored"
