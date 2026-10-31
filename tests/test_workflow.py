"""The Snakemake workflow and its two profiles, checked by dry runs (no GPU, no SLURM)."""

import re
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CARC_ACCOUNT = "irahbari_1147"  # docs/carc.md; the user's default account belongs to another group


@cache  # each dry run starts Snakemake, which takes ~10 s
def dry_run(profile: str, *targets: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "snakemake", "--profile", f"profiles/{profile}", "--dry-run", *targets],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def planned_jobs(output: str) -> dict[str, dict[str, str]]:
    """Map each planned rule to the resources the dry run says it will request."""
    jobs = {}
    for block in re.split(r"\n(?=(?:local)?rule \w+:)", output):
        rule = re.match(r"(?:local)?rule (\w+):", block)
        resources = re.search(r"resources: (.*)", block)
        if rule and resources:
            jobs[rule.group(1)] = dict(item.split("=", 1) for item in resources.group(1).split(", "))
    return jobs


@pytest.mark.parametrize("profile", ["local", "carc"])
def test_the_smoke_rule_runs_under_both_profiles(profile):
    jobs = planned_jobs(dry_run(profile, "smoke"))

    assert jobs["smoke"]["gpu"] == "1"


def test_carc_charges_the_course_account_explicitly():
    # Without slurm_account the SLURM plugin guesses from recent jobs, which could
    # charge another group.
    jobs = planned_jobs(dry_run("carc", "smoke"))

    assert all(resources.get("slurm_account") == CARC_ACCOUNT for resources in jobs.values())


def test_every_carc_gpu_job_pins_its_gpu_model():
    jobs = planned_jobs(dry_run("carc", "smoke"))

    gpu_jobs = {rule: r for rule, r in jobs.items() if int(r.get("gpu", "0")) > 0}
    assert gpu_jobs, "expected at least one GPU job"
    assert {rule: r.get("gpu_model") for rule, r in gpu_jobs.items()} == {rule: "l40s" for rule in gpu_jobs}


def test_carc_stays_under_the_gpu_partition_queue_cap():
    profile = yaml.safe_load((REPO / "profiles/carc/config.yaml").read_text())

    assert 0 < profile["jobs"] < 100  # the gpu partition allows 100 queued jobs per user
