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
CARC_GPU_TYPES = {"l40s", "a100", "a40", "v100", "p100"}  # docs/carc.md, GPUs table


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


@pytest.mark.parametrize("profile", ["local", "carc"])
def test_smoke_plans_a_preflight_first(profile):
    # preflight is a localrule: it runs where Snakemake was launched (the CARC login
    # node), so a broken module line fails in seconds instead of after a queue wait.
    # A dry run doesn't show which jobs run locally; the CARC run log does.
    jobs = planned_jobs(dry_run(profile, "smoke"))

    assert "preflight" in jobs


def test_carc_charges_the_course_account_explicitly():
    # Without slurm_account the SLURM plugin guesses from recent jobs, which could
    # charge another group.
    jobs = planned_jobs(dry_run("carc", "smoke"))

    assert all(resources.get("slurm_account") == CARC_ACCOUNT for resources in jobs.values())


def test_every_carc_gpu_job_pins_its_gpu_model():
    jobs = planned_jobs(dry_run("carc", "smoke"))

    gpu_jobs = {rule: r for rule, r in jobs.items() if int(r.get("gpu", "0")) > 0}
    assert gpu_jobs, "expected at least one GPU job"
    unpinned = {rule: r.get("gpu_model") for rule, r in gpu_jobs.items() if r.get("gpu_model") not in CARC_GPU_TYPES}
    assert not unpinned, f"GPU jobs without a pinned CARC GPU type: {unpinned}"


JEZERO_RULES = {"download_mosaic", "jezero_tile", "jezero_scene", "jezero_matching_map", "jezero_mesh"}


@pytest.mark.parametrize("profile", ["local", "carc"])
def test_jezero_plans_both_downloads_the_tile_and_the_matching_map(profile):
    output = dry_run(profile, "jezero", "--forceall")

    assert JEZERO_RULES <= set(planned_jobs(output))
    # DTM and orthoimage from HiRISE, and from CTX for the ring (#33).
    assert len(re.findall(r"(?:local)?rule download_mosaic:", output)) == 4


def test_every_download_is_checked_against_a_pinned_sha256():
    # USGS publishes no checksums: an unpinned mosaic would accept whatever bytes arrive.
    output = dry_run("local", "jezero", "--forceall")

    pinned = re.findall(r"fidelis-fetch \S+ \S+ --sha256 [0-9a-f]{64} --record", output)
    assert len(pinned) == len(re.findall(r"(?:local)?rule download_mosaic:", output)) == 4


def localrules() -> set[str]:
    """Rules in the Snakefile's `localrules:` block. A dry run labels every job `rule`
    under the SLURM executor, so it can't show which ones run locally."""
    snakefile = (REPO / "workflow/Snakefile").read_text()
    block = re.search(r"^localrules:\n((?:[ \t]+.*\n)+)", snakefile, re.MULTILINE)
    assert block, "no localrules: block in workflow/Snakefile"
    return set(re.findall(r"^[ \t]+(\w+),", block.group(1), re.MULTILINE))


def test_downloads_run_locally_not_as_slurm_jobs():
    # CARC compute nodes have no internet (docs/carc.md); a SLURM download job would fail.
    assert "download_mosaic" in localrules()


def test_localrules_lists_only_real_rules():
    # Guards the parser above: a typo or a parse miss would make the download check meaningless.
    rules = set(planned_jobs(dry_run("local", "jezero", "--forceall")))

    assert localrules() == {"preflight", "jezero", "download_mosaic"}
    assert localrules() <= rules


def test_jezero_data_jobs_stay_off_the_gpu_partition():
    # CPU-only work on the gpu partition would wait in the GPU queue and count against its cap.
    jobs = planned_jobs(dry_run("carc", "jezero", "--forceall"))

    for rule in JEZERO_RULES - {"download_mosaic"}:  # downloads are localrules, never submitted
        assert jobs[rule].get("slurm_partition") != "gpu", rule
        assert int(jobs[rule].get("gpu", "0")) == 0, rule


@pytest.mark.parametrize(
    "path",
    [
        "data/jezero/raw/ortho.tif",
        "data/jezero/raw/dtm.verified",
        "data/jezero/tile/dtm.tif",
        "data/jezero/tile/tile.json",
        "data/jezero/matching_map/map.tif",
        "data/jezero/mesh/mesh.ply",
        "data/jezero/mesh/texture.png",
    ],
)
def test_jezero_data_stays_out_of_git(path):
    result = subprocess.run(["git", "check-ignore", "--quiet", path], cwd=REPO)

    assert result.returncode == 0, f"{path} is not git-ignored"


def test_carc_stays_under_the_gpu_partition_queue_cap():
    profile = yaml.safe_load((REPO / "profiles/carc/config.yaml").read_text())

    assert 0 < profile["jobs"] < 100  # the gpu partition allows 100 queued jobs per user
