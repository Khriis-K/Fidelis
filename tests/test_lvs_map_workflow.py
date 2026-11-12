"""The LVS-map rule (#36), checked by dry runs (no GPU, no SLURM)."""

import re
import subprocess

import pytest
from test_workflow import REPO, dry_run, planned_jobs


@pytest.mark.parametrize("profile", ["local", "carc"])
def test_jezero_builds_the_lvs_map_from_the_raw_ctx_mosaic(profile):
    output = dry_run(profile, "jezero", "--forceall")

    assert "jezero_lvs_map" in planned_jobs(output)
    command = re.search(r"fidelis-lvs-map .*", output).group(0)
    assert "data/jezero/raw/ctx_ortho.tif" in command
    assert command.rstrip().endswith("data/jezero/lvs_map/map.tif")


def test_the_old_map_is_still_built_so_earlier_results_reproduce():
    assert "jezero_matching_map" in planned_jobs(dry_run("local", "jezero", "--forceall"))


def test_the_monte_carlo_run_waits_for_the_lvs_map():
    jobs = planned_jobs(dry_run("local", "plain_mc", "--forceall"))

    assert {"jezero_lvs_map", "plain_mc"} <= set(jobs)


def test_the_lvs_map_job_stays_off_the_gpu_partition():
    # CPU-only work on the gpu partition would wait in the GPU queue and count against its cap.
    job = planned_jobs(dry_run("carc", "jezero", "--forceall"))["jezero_lvs_map"]

    assert job.get("slurm_partition") == "main"
    assert int(job.get("gpu", "0")) == 0


@pytest.mark.parametrize("path", ["data/jezero/lvs_map/map.tif", "data/jezero/lvs_map/map.json"])
def test_the_lvs_map_stays_out_of_git(path):
    result = subprocess.run(["git", "check-ignore", "--quiet", path], cwd=REPO)

    assert result.returncode == 0, f"{path} is not git-ignored"
