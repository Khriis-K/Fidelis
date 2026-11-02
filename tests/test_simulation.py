"""Seam 1, the simulation interface, on the CPU: a flat textured map stands in for Jezero,
and a ray-cast of it stands in for the mesh renderer. The GPU test at the end runs the
real mesh renderer on Jezero."""

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest
from rasterio.transform import from_origin

from fidelis.nav.navigator import Navigator
from fidelis.nav.orbital_map import OrbitalMap
from fidelis.render.camera import CameraPose, Intrinsics
from fidelis.scenario import Scenario
from fidelis.simulation import SimulationRecord, Simulator, scenario_pose
from fidelis.terrain.fetch import sha256_of

REPO = Path(__file__).resolve().parents[1]
LEFT, TOP = 4589000.0, 1094000.0
MAP_RES = 2.0
MAP_SIZE = 512
EAST_SCALE = 0.95
ORIGIN = (LEFT + MAP_RES * MAP_SIZE / 2, TOP - MAP_RES * MAP_SIZE / 2)  # world (0, 0) at the map's centre
GROUND_Z = -2500.0
K = Intrinsics.from_fov(256, 256, 90)
MAP_PIXEL_M = MAP_RES * EAST_SCALE  # the map's coarser ground pixel, east-west
# GPU camera images of the same seed: 19x the largest in-process jitter measured (5.36e-7),
# about 1/700 of a wrongly seeded noise draw's difference.
CAMERA_IMAGE_REPEAT_TOLERANCE = 1e-5


def texture(size, seed=0):
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), np.float32)
    for sigma in (1, 2, 4, 8):
        image += cv2.GaussianBlur(rng.random((size, size), np.float32), (0, 0), sigma)
    return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)


FLAT_MAP = OrbitalMap(
    image=texture(MAP_SIZE),
    transform=from_origin(LEFT, TOP, MAP_RES, MAP_RES),
    dtm=np.full((8, 8), GROUND_Z, np.float32),
    dtm_transform=from_origin(LEFT, TOP, MAP_RES * MAP_SIZE / 8, MAP_RES * MAP_SIZE / 8),
    origin=ORIGIN,
    east_scale=EAST_SCALE,
)


def render_flat(pose: CameraPose, intrinsics: Intrinsics, sun_index: int, seed: int) -> np.ndarray:
    """Ray-cast the flat map from any pose: each pixel's ray, met with the ground plane."""
    u, v = np.meshgrid(np.arange(intrinsics.width, dtype=np.float64), np.arange(intrinsics.height, dtype=np.float64))
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(intrinsics.matrix()).T @ pose.rotation.T
    s = (GROUND_Z - pose.position[2]) / rays[..., 2]
    east = pose.position[0] + s * rays[..., 0]
    north = pose.position[1] + s * rays[..., 1]
    col = ((east / EAST_SCALE + ORIGIN[0]) - LEFT) / MAP_RES - 0.5
    row = (TOP - (north + ORIGIN[1])) / MAP_RES - 0.5
    return cv2.remap(FLAT_MAP.image, col.astype(np.float32), row.astype(np.float32), cv2.INTER_CUBIC)


def render_black(pose, intrinsics, sun_index, seed):
    return np.zeros((intrinsics.height, intrinsics.width), np.float32)


@pytest.fixture(scope="module")
def simulator():
    navigator = Navigator(FLAT_MAP, device="cpu")
    return Simulator({"flat": render_flat, "black": render_black}, navigator, K)


def scenario(**overrides):
    """Perfect conditions unless overridden: nadir, north up, no noise or blur, nominal exposure."""
    values = dict(
        seed=11,
        offset_east_m=40.0,
        offset_north_m=-25.0,
        altitude_m=192.0,  # 1.5 m descent pixels: finer than the map, as in flight
        tilt_deg=0.0,
        tilt_azimuth_deg=0.0,
        heading_deg=0.0,
        noise_sd=0.0,
        blur_px=0.0,
        blur_angle_deg=0.0,
        exposure_gain=1.0,
        sun_index=0,
    )
    return Scenario(**(values | overrides))


# --- the true camera a scenario describes


def test_the_camera_sits_its_altitude_above_the_ground_beneath_it():
    pose = scenario_pose(scenario(offset_east_m=12.0, offset_north_m=-30.0, altitude_m=500.0), ground_z=-2400.0)
    np.testing.assert_allclose(pose.position, [12.0, -30.0, -1900.0])


@pytest.mark.parametrize("heading", [0.0, 90.0, 200.0])
def test_an_untilted_camera_looks_straight_down_with_its_image_top_towards_its_heading(heading):
    pose = scenario_pose(scenario(heading_deg=heading), ground_z=0.0)
    h = math.radians(heading)
    np.testing.assert_allclose(pose.rotation[:, 2], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(-pose.rotation[:, 1], [math.sin(h), math.cos(h), 0], atol=1e-12)


@pytest.mark.parametrize("azimuth", [0.0, 90.0, 225.0])
def test_tilt_leans_the_optical_axis_off_nadir_towards_its_azimuth(azimuth):
    pose = scenario_pose(scenario(tilt_deg=3.0, tilt_azimuth_deg=azimuth, heading_deg=140.0), ground_z=0.0)
    t, a = math.radians(3.0), math.radians(azimuth)
    expected = [math.sin(t) * math.sin(a), math.sin(t) * math.cos(a), -math.cos(t)]
    np.testing.assert_allclose(pose.rotation[:, 2], expected, atol=1e-12)


# --- the record


def test_perfect_conditions_locate_within_a_quarter_map_pixel(simulator):
    record = simulator.run(scenario(), "flat").record
    assert record.status == "ok"
    assert record.error_m < 0.25 * MAP_PIXEL_M


def test_a_tilted_turned_noisy_view_still_locates_within_a_map_pixel(simulator):
    # The navigator must get the attitude the render was made with: without it a view turned
    # 200 degrees doesn't match at all. (The tilt convention itself is pinned by the
    # scenario_pose tests above; here the renderer shares the pose, so it can't show.)
    s = scenario(
        tilt_deg=3.0,
        tilt_azimuth_deg=120.0,
        heading_deg=200.0,
        noise_sd=0.005,
        blur_px=0.4,
        blur_angle_deg=60.0,
        exposure_gain=1.3,
    )
    record = simulator.run(s, "flat").record
    assert record.status == "ok"
    assert record.error_m < MAP_PIXEL_M


def test_the_error_vector_is_estimate_minus_truth_and_its_magnitude_is_the_error(simulator):
    record = simulator.run(scenario(), "flat").record
    truth = np.array(record.true_position)
    np.testing.assert_allclose(truth, [40.0, -25.0, GROUND_Z + 192.0])
    np.testing.assert_allclose(record.error_vector, np.array(record.estimated_position) - truth)
    assert record.error_m == pytest.approx(np.linalg.norm(record.error_vector))


def test_the_record_carries_diagnostics_cost_gpu_and_the_scenario(simulator):
    s = scenario()
    record = simulator.run(s, "flat").record
    assert record.seed == s.seed and record.fidelity == "flat"
    assert record.scenario == {k: v for k, v in asdict(s).items() if k != "seed"}
    assert record.match_count >= record.inlier_count > 50
    assert record.cost_s > 0
    assert record.gpu == "cpu"


def test_the_same_scenario_and_seed_give_the_same_error(simulator):
    s = scenario(noise_sd=0.005, blur_px=0.3, heading_deg=45.0)
    first, second = (simulator.run(s, "flat").record for _ in range(2))
    assert first.status == second.status == "ok"
    assert first.error_vector == second.error_vector


def test_another_seed_with_the_same_conditions_draws_other_noise_and_another_error(simulator):
    a = simulator.run(scenario(noise_sd=0.005, seed=1), "flat").record
    b = simulator.run(scenario(noise_sd=0.005, seed=2), "flat").record
    assert a.error_vector != b.error_vector


def test_a_solve_failure_gives_a_valid_failure_record_not_an_exception(simulator):
    record = simulator.run(scenario(), "black").record
    assert record.status != "ok"
    assert record.estimated_position is None and record.error_vector is None and record.error_m is None
    assert record.inlier_count == 0
    np.testing.assert_allclose(record.true_position, [40.0, -25.0, GROUND_Z + 192.0])
    assert record.cost_s > 0


@pytest.mark.parametrize("renderer", ["flat", "black"])
def test_a_record_survives_a_json_round_trip(simulator, renderer):
    record = simulator.run(scenario(), renderer).record
    assert SimulationRecord(**json.loads(json.dumps(asdict(record)))) == record


def test_a_run_keeps_the_camera_image_and_the_matches_for_replay(simulator):
    run = simulator.run(scenario(), "flat")
    assert run.image.shape == (K.height, K.width)
    assert run.nav.match_count == run.record.match_count
    assert run.nav.inlier_count == run.record.inlier_count


def test_a_renderers_scene_loading_is_kept_out_of_the_samples_cost(simulator):
    # Cost feeds MXMCPy's sample allocation: it's the per-sample price, not the price of
    # loading a sun's scene, which a batch grouped by sun pays once.
    class SlowToLoad:
        loaded = None

        def use_sun(self, sun_index):
            time.sleep(2.0)
            self.loaded = sun_index

        def __call__(self, pose, intrinsics, sun_index, seed):
            assert self.loaded == sun_index, "the scene wasn't loaded before the render"
            return render_black(pose, intrinsics, sun_index, seed)

    record = Simulator({"slow": SlowToLoad()}, simulator.navigator, K).run(scenario(sun_index=2), "slow").record
    assert record.cost_s < 2.0


def test_every_record_names_the_config_it_ran_under(simulator, tmp_path):
    # A seed reruns the same descent only under the same config: the record says which.
    config = tmp_path / "scene.yaml"
    config.write_text("scenario: {}\n")
    digest = sha256_of(config)
    assert digest == hashlib.sha256(config.read_bytes()).hexdigest()
    sim = Simulator(simulator.renderers, simulator.navigator, K, config_sha256=digest)
    assert sim.run(scenario(), "black").record.config_sha256 == digest


def test_an_unknown_fidelity_is_a_caller_error(simulator):
    with pytest.raises(ValueError, match="fidelity"):
        simulator.run(scenario(), "ultra")


def test_the_renderer_gets_the_scenarios_sun_seed_and_true_camera(simulator):
    seen = []

    def spy(pose, intrinsics, sun_index, seed):
        seen.append((pose, sun_index, seed))
        return render_black(pose, intrinsics, sun_index, seed)

    s = scenario(sun_index=3, seed=5, tilt_deg=2.0, tilt_azimuth_deg=70.0, heading_deg=33.0)
    Simulator({"spy": spy}, simulator.navigator, K).run(s, "spy")
    [(pose, sun_index, seed)] = seen
    assert (sun_index, seed) == (3, 5)
    expected = scenario_pose(s, ground_z=GROUND_Z)
    np.testing.assert_allclose(pose.rotation, expected.rotation)
    np.testing.assert_allclose(pose.position, expected.position)


def test_the_scene_config_has_a_sun_for_every_sun_index_the_sampler_can_draw():
    import yaml

    scene = yaml.safe_load((REPO / "config" / "jezero.yaml").read_text(encoding="utf-8"))
    sun_index = scene["scenario"]["sun_index"]
    assert (sun_index["low"], sun_index["high"]) == (0, len(scene["suns"]) - 1)


@pytest.mark.gpu
@pytest.mark.parametrize("altitude", [2300.0, 4100.0])
def test_perfect_conditions_on_the_jezero_mesh_locate_within_half_a_ctx_map_pixel(altitude):
    # As the navigator's GPU test, but through the seam: the scenario's camera, the mesh
    # renderer, exposure from the camera model. Zenith sun so the shading adds nothing
    # the map's orthoimage doesn't carry; half the 6 m map pixel is the tolerance.
    from fidelis.render.mesh_render import Sun
    from fidelis.simulation import build_mesh_simulator

    data = REPO / "data" / "jezero"
    assert (data / "mesh" / "mesh.json").exists(), (
        "build the Jezero scene first: snakemake --profile profiles/local jezero"
    )
    sim = build_mesh_simulator(REPO / "config" / "jezero.yaml", data, suns=[Sun(0.0, 90.0)])
    s = scenario(offset_east_m=150.0, offset_north_m=-230.0, altitude_m=altitude)
    first = sim.run(s, "mesh").record
    assert first.status == "ok"
    assert first.error_m < 3.0
    assert first.gpu != "cpu"
    # Same sample and seed, same camera image, up to the renderer's GPU jitter: within one
    # process camera images have differed by at most 5.36e-7 (scripts/check_gpu_repeatability.py).
    # Not the same error: the navigator has turned that jitter into up to 0.839 m, so errors
    # aren't compared (docs/adr/0001-gpu-reproducibility-is-image-level.md). A seed that didn't
    # fix the camera noise would differ by about 7e-3 RMS at this noise SD, far past the tolerance.
    # Not covered here: the scenario's own draw from a seed (tests/test_scenario.py does that).
    noisy = scenario(offset_east_m=150.0, offset_north_m=-230.0, altitude_m=altitude, noise_sd=0.005)
    once, again = sim.run(noisy, "mesh"), sim.run(noisy, "mesh")
    assert np.abs(again.image - once.image).max() < CAMERA_IMAGE_REPEAT_TOLERANCE
    # Records stay JSON-safe on the GPU path too.
    assert SimulationRecord(**json.loads(json.dumps(asdict(first)))) == first
