"""Seam 1 with the splat renderer: the trained Jezero splat behind the simulation interface.

Needs a GPU and the trained scene: `snakemake splat` (on CARC), then these run where the
checkpoint is, e.g. `uv run pytest -m gpu tests/test_simulation_splat.py`.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from fidelis.scenario import Scenario
from fidelis.simulation import LCAM_PX, SimulationRecord, build_simulator, scenario_pose
from fidelis.splat.metrics import QualityGate, quality_report

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "jezero.yaml"
DATA = REPO / "data" / "jezero"
SCENE = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
SUN = SCENE["splat"]["sun_index"]
# As tests/test_simulation.py: GPU camera images of one seed repeat to within this.
CAMERA_IMAGE_REPEAT_TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def simulator():
    assert (DATA / "splat" / f"sun{SUN}" / "splat.pt").exists(), (
        "train the splat first: snakemake --profile profiles/carc splat"
    )
    return build_simulator(CONFIG, DATA, fidelities=("mesh", "splat"))


def scenario(**overrides):
    """Perfect conditions under the trained sun: nadir, north up, no noise or blur."""
    values = dict(
        seed=11,
        offset_east_m=150.0,
        offset_north_m=-230.0,
        altitude_m=2300.0,
        tilt_deg=0.0,
        tilt_azimuth_deg=0.0,
        heading_deg=0.0,
        noise_sd=0.0,
        blur_px=0.0,
        blur_angle_deg=0.0,
        exposure_gain=1.0,
        sun_index=SUN,
    )
    return Scenario(**(values | overrides))


def test_perfect_conditions_on_the_splat_locate_within_a_ctx_map_pixel(simulator):
    # The mesh under the landing-day sun located this camera within 3.4 m
    # (docs/benchmarks/navigator.md); a 6 m map pixel leaves the splat some room, not much.
    record = simulator.run(scenario(), "splat").record
    assert record.status == "ok"
    assert record.error_m < 6.0
    assert record.gpu != "cpu" and record.fidelity == "splat"
    assert SimulationRecord(**json.loads(json.dumps(asdict(record)))) == record


def test_a_tilted_turned_noisy_descent_still_locates(simulator):
    s = scenario(tilt_deg=10.0, tilt_azimuth_deg=120.0, heading_deg=200.0, noise_sd=0.005, blur_px=0.4)
    record = simulator.run(s, "splat").record
    assert record.status == "ok"
    assert record.error_m < 12.0  # the config's provisional r: this descent shouldn't count as a failure


def test_the_same_scenario_and_seed_give_the_same_camera_image(simulator):
    s = scenario(noise_sd=0.005)
    once, again = simulator.run(s, "splat"), simulator.run(s, "splat")
    assert np.abs(again.image - once.image).max() < CAMERA_IMAGE_REPEAT_TOLERANCE


def test_splat_renders_match_mesh_renders_at_descent_poses_it_never_trained_on(simulator):
    # Monte Carlo seeds' poses: never training views (fidelis.splat.views keeps them apart).
    from fidelis.scenario import draw, load_config, pin_sun

    config = pin_sun(load_config(CONFIG), SUN)
    gate = QualityGate(**SCENE["splat"]["quality"])
    mesh, splat = simulator.renderers["mesh"], simulator.renderers["splat"]
    mesh_images, splat_images = [], []
    for seed in range(5):
        s = draw(config, seed)
        ground = simulator.navigator.map.ground_height(s.offset_east_m, s.offset_north_m)
        pose = scenario_pose(s, ground)
        mesh_images.append(mesh(pose, simulator.intrinsics, SUN, seed))
        splat_images.append(splat(pose, simulator.intrinsics, SUN, seed))
    assert mesh_images[0].shape == (LCAM_PX, LCAM_PX)
    report = quality_report(splat_images, mesh_images, gate)
    assert report["passed"], report


def test_the_splat_costs_less_per_sample_than_the_mesh(simulator):
    # Seam 1: per-sample cost increases with fidelity level, and the splat is the cheaper
    # one. Medians over a few samples, after one warm-up each (the first render in a process
    # pays one-off GPU setup, docs/benchmarks/plain-mc.md).
    def median_cost(fidelity):
        simulator.run(scenario(seed=100), fidelity)
        return float(np.median([simulator.run(scenario(seed=101 + i), fidelity).record.cost_s for i in range(5)]))

    assert median_cost("splat") < median_cost("mesh")


def test_a_view_of_empty_sky_gives_a_valid_failure_record(simulator):
    # Tilted 120 deg the camera looks up, past every Gaussian: nothing to match.
    record = simulator.run(scenario(tilt_deg=120.0), "splat").record
    assert record.status != "ok"
    assert record.estimated_position is None and record.error_m is None
    assert record.inlier_count == 0
    assert SimulationRecord(**json.loads(json.dumps(asdict(record)))) == record


def test_a_sun_with_no_splat_is_a_caller_error_not_a_crash_mid_batch(simulator):
    missing = next(i for i in range(len(SCENE["suns"])) if i not in simulator.renderers["splat"].suns)
    with pytest.raises(ValueError, match="sun"):
        simulator.run(scenario(sun_index=missing), "splat")
