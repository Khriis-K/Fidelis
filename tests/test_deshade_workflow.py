"""The de-shaded texture rule (#19), checked by dry runs (no GPU, no SLURM)."""

import re
import subprocess

import pytest
from test_workflow import REPO, dry_run, planned_jobs

TEXTURE = "data/jezero/mesh/texture_deshaded.png"


@pytest.mark.parametrize("profile", ["local", "carc"])
def test_the_workflow_builds_the_deshaded_texture_from_the_scene(profile):
    output = dry_run(profile, TEXTURE, "--forceall")

    assert "jezero_deshaded_texture" in planned_jobs(output)
    command = re.search(r"fidelis-deshade .*", output).group(0)
    assert "config/jezero.yaml" in command
    assert "data/jezero/scene" in command
    assert command.rstrip().endswith(TEXTURE)


def test_the_deshade_job_stays_off_the_gpu_partition():
    # CPU-only work on the gpu partition would wait in the GPU queue and count against its cap.
    job = planned_jobs(dry_run("carc", TEXTURE, "--forceall"))["jezero_deshaded_texture"]

    assert job.get("slurm_partition") == "main"
    assert int(job.get("gpu", "0")) == 0


@pytest.mark.parametrize("target", ["jezero", "plain_mc"])
def test_earlier_targets_do_not_pick_up_the_new_texture(target):
    # Results so far were rendered from the photo texture. Adding an input to their rules
    # would send them back through the queue.
    assert "jezero_deshaded_texture" not in planned_jobs(dry_run("local", target, "--forceall"))


@pytest.mark.parametrize("path", [TEXTURE, "data/jezero/mesh/texture_deshaded.json"])
def test_the_deshaded_texture_stays_out_of_git(path):
    result = subprocess.run(["git", "check-ignore", "--quiet", path], cwd=REPO)

    assert result.returncode == 0, f"{path} is not git-ignored"
