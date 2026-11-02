"""Seam 1: one simulated descent, from a scenario and a fidelity level to a record of how navigation went.

`Simulator.run(scenario, fidelity)` puts the true camera where the scenario says, renders
it at that fidelity under the scenario's sun, turns the render into a camera image
(fidelis.render.degrade), and has the navigator locate it. The record holds the
navigation error, the navigator's diagnostics, the wall-clock cost and the GPU it ran on;
everything else about the descent comes back from the seed.

Fidelity levels are renderers, by name: `"mesh"` (path-traced mesh) for now, with splat
levels to come behind the same interface. A renderer is any callable
`(pose, intrinsics, sun_index, seed) -> radiance image`. One that loads a scene per sun
may also have `use_sun(sun_index)`, which `run` calls before the clock starts, so the
recorded cost is the per-sample price and not a scene load.

The navigator gets the true attitude: attitude-knowledge (IMU) error isn't modelled
(issue #34, docs/benchmarks/navigator.md).
"""

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from fidelis.nav.navigator import Navigator, NavResult
from fidelis.nav.orbital_map import load_orbital_map
from fidelis.render.camera import CameraPose, Intrinsics, look_at
from fidelis.render.degrade import camera_image
from fidelis.scenario import Scenario
from fidelis.terrain.fetch import sha256_of

Renderer = Callable[[CameraPose, Intrinsics, int, int], np.ndarray]

# LCAM: 1024 x 1024 pixels (Maki et al. 2020; docs/camera-conventions.md).
LCAM_PX = 1024


def scenario_pose(scenario: Scenario, ground_z: float) -> CameraPose:
    """The true camera: its altitude above `ground_z`, image top towards the heading, then tilted.

    The tilt turns the optical axis off nadir towards the tilt azimuth's compass bearing,
    about the horizontal axis square to it.
    """
    position = np.array([scenario.offset_east_m, scenario.offset_north_m, ground_z + scenario.altitude_m])
    h = math.radians(scenario.heading_deg)
    nadir = look_at(position, position - [0, 0, 1], [math.sin(h), math.cos(h), 0]).rotation
    a = math.radians(scenario.tilt_azimuth_deg)
    # Turning nadir (0, 0, -1) towards bearing a: the axis is (0, 0, -1) x (sin a, cos a, 0).
    axis = np.array([math.cos(a), -math.sin(a), 0.0])
    tilt = cv2.Rodrigues(axis * math.radians(scenario.tilt_deg))[0]
    return CameraPose(rotation=tilt @ nadir, position=position)


@dataclass(frozen=True)
class SimulationRecord:
    """One descent's outcome, in plain JSON types. Failed solves leave the estimate and error as None."""

    seed: int
    fidelity: str
    status: str  # the navigator's: "ok", "too_few_matches" or "no_solution"
    true_position: list[float]  # world metres (docs/camera-conventions.md)
    estimated_position: list[float] | None
    error_vector: list[float] | None  # estimate minus truth, metres
    error_m: float | None  # its length
    match_count: int
    inlier_count: int
    cost_s: float  # wall clock for render, camera model and navigation; not scene or model loading
    gpu: str  # the device that did the work, "cpu" without one
    scenario: dict  # the scenario's inputs, without the seed
    # SHA-256 of the scene config: the seed reruns this descent only under the same config.
    # None for a simulator built without one (tests); optional so older records still load.
    config_sha256: str | None = None


@dataclass(frozen=True)
class Run:
    record: SimulationRecord
    image: np.ndarray  # the camera image the navigator saw
    nav: NavResult  # with the matched pixel pairs and inlier mask, for drawing


class Simulator:
    def __init__(
        self,
        renderers: dict[str, Renderer],
        navigator: Navigator,
        intrinsics: Intrinsics,
        config_sha256: str | None = None,
    ):
        self.renderers = renderers
        self.navigator = navigator
        self.intrinsics = intrinsics
        self.config_sha256 = config_sha256
        self.gpu = torch.cuda.get_device_name() if navigator.device.startswith("cuda") else "cpu"

    def run(self, scenario: Scenario, fidelity: str) -> Run:
        if fidelity not in self.renderers:
            raise ValueError(f"unknown fidelity {fidelity!r}; this simulator has {sorted(self.renderers)}")
        ground_z = self.navigator.map.ground_height(scenario.offset_east_m, scenario.offset_north_m)
        pose = scenario_pose(scenario, ground_z)
        renderer = self.renderers[fidelity]
        if hasattr(renderer, "use_sun"):
            renderer.use_sun(scenario.sun_index)
        start = time.perf_counter()
        radiance = renderer(pose, self.intrinsics, scenario.sun_index, scenario.seed)
        image = camera_image(radiance, scenario)
        nav = self.navigator.locate(image, self.intrinsics, pose.rotation)
        cost = time.perf_counter() - start

        estimate = error = None
        if nav.position is not None:
            estimate = nav.position.tolist()
            error = (nav.position - pose.position).tolist()
        inputs = asdict(scenario)
        del inputs["seed"]
        record = SimulationRecord(
            seed=scenario.seed,
            fidelity=fidelity,
            status=nav.status,
            true_position=pose.position.tolist(),
            estimated_position=estimate,
            error_vector=error,
            error_m=None if error is None else float(np.linalg.norm(error)),
            match_count=nav.match_count,
            inlier_count=nav.inlier_count,
            cost_s=cost,
            gpu=self.gpu,
            scenario=inputs,
            config_sha256=self.config_sha256,
        )
        return Run(record, image, nav)


def build_mesh_simulator(config_path: Path, data_dir: Path, suns=None, device: str = "cuda") -> Simulator:
    """The Jezero mesh at fidelity "mesh", the CTX-like map, and LCAM, from the scene config and workflow data.

    `suns` overrides the config's sun angles (a test's zenith sun, say).
    """
    from fidelis.render.mesh_render import MeshRenderer, Sun

    scene = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_dir = Path(data_dir)
    if suns is None:
        suns = [Sun(s["azimuth_deg"], s["elevation_deg"]) for s in scene["suns"]]
    orbital_map = load_orbital_map(
        data_dir / "matching_map" / "map.tif", data_dir / "tile" / "dtm.tif", data_dir / "mesh" / "mesh.json"
    )
    renderer = MeshRenderer(data_dir / "mesh", suns, spp=scene["simulation"]["mesh_spp"])
    intrinsics = Intrinsics.from_fov(LCAM_PX, LCAM_PX, scene["scenario"]["camera_fov_deg"])
    navigator = Navigator(orbital_map, device=device)
    return Simulator({"mesh": renderer}, navigator, intrinsics, config_sha256=sha256_of(config_path))
