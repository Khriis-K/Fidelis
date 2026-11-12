"""Seam 1: one simulated descent, from a scenario and a fidelity level to a record of how navigation went.

`Simulator.run(scenario, fidelity)` puts the true camera where the scenario says, renders
it at that fidelity under the scenario's sun, turns the render into a camera image
(fidelis.render.degrade), and has the navigator locate it. The record holds the
navigation error, the navigator's diagnostics, the wall-clock cost and the GPU it ran on;
everything else about the descent comes back from the seed.

A fidelity level names a renderer, `"mesh"` (path-traced mesh) or `"splat"` (a Gaussian
splat scene trained from mesh renders, one per sun), optionally thinned: `"splat-k8000"`
renders the same splat but matches against only the map's 8,000 highest-scoring keypoints
(map thinning, docs/adr/0002-navigator-fidelity-ladder.md). A bare name is the real
navigator; the cheaper levels differ only in the navigator. A renderer is any callable
`(pose, intrinsics, sun_index, seed) -> radiance image`. One that loads a scene per sun
may also have `use_sun(sun_index)`, which `run` calls before the clock starts, so the
recorded cost is the per-sample price and not a scene load.

Each record also carries the map-frame error: the error less the whole-map offset between
the matching map and the scene, read from the scene config (fidelis.map_frame, ADR 0003).
The true-frame error stays alongside it.

The navigator matches against the map the scene config names (`matching_map.source`): the
real LVS map, CTX imagery with CTX elevation (#36), or the scene's own map, kept so earlier
results reproduce. The true camera is placed above the scene's terrain either way: the
truth can't depend on what the navigator's map believes the ground height is.

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

from fidelis.map_frame import MapOffset, load_map_offset, map_frame_error, matching_map_source
from fidelis.nav.navigator import Navigator, NavResult
from fidelis.nav.orbital_map import OrbitalMap, load_lvs_map, load_orbital_map, load_terrain
from fidelis.render.camera import CameraPose, Intrinsics, look_at
from fidelis.render.degrade import camera_image
from fidelis.scenario import Scenario
from fidelis.terrain.fetch import sha256_of

Renderer = Callable[[CameraPose, Intrinsics, int, int], np.ndarray]

# LCAM: 1024 x 1024 pixels (Maki et al. 2020; docs/camera-conventions.md).
LCAM_PX = 1024

# For a simulator built without a scene config (tests): the map is the scene's own.
NO_OFFSET = MapOffset(0.0, 0.0, "none given: simulator built without a scene config")


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
    # How many map keypoints the level's navigator matched against: the whole map for a bare
    # renderer name, and K past the map's count is the whole map too. None in older records.
    map_keypoints: int | None = None
    # The whole-map offset [east, north] removed for the map-frame error, and where it came
    # from (fidelis.map_frame, ADR 0003). Failure is judged on the map-frame error; None in
    # records from before ADR 0003, whose map was the scene's own, so the offset was zero.
    map_offset_m: list[float] | None = None
    map_offset_source: str | None = None
    map_frame_error_vector: list[float] | None = None  # error_vector less the offset
    map_frame_error_m: float | None = None  # its length
    # The map the navigator matched against: "ctx" (the real LVS map) or "scene" (the scene's
    # own). None in records from before #36, which all used the scene's own, and from a
    # simulator built without a scene config (tests).
    matching_map: str | None = None


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
        map_offset: MapOffset = NO_OFFSET,
        ground_height: Callable[[float, float], float] | None = None,
        matching_map: str | None = None,
    ):
        self.renderers = renderers
        self.navigator = navigator
        self.intrinsics = intrinsics
        self.config_sha256 = config_sha256
        self.map_offset = map_offset
        # The scene's true ground height beneath (east, north). Without one, the map's own
        # DTM is the scene's (tests, and the scene's own map).
        self.ground_height = ground_height or navigator.map.ground_height
        self.matching_map = matching_map
        self._thinned: dict[int, Navigator] = {}
        self.gpu = torch.cuda.get_device_name() if navigator.device.startswith("cuda") else "cpu"

    def run(self, scenario: Scenario, fidelity: str) -> Run:
        renderer_name, map_keypoints = parse_fidelity(fidelity)
        if renderer_name not in self.renderers:
            raise ValueError(f"unknown fidelity {fidelity!r}; this simulator has {sorted(self.renderers)}")
        navigator = self._navigator(map_keypoints)
        ground_z = self.ground_height(scenario.offset_east_m, scenario.offset_north_m)
        pose = scenario_pose(scenario, ground_z)
        renderer = self.renderers[renderer_name]
        if hasattr(renderer, "use_sun"):
            renderer.use_sun(scenario.sun_index)
        start = time.perf_counter()
        radiance = renderer(pose, self.intrinsics, scenario.sun_index, scenario.seed)
        image = camera_image(radiance, scenario)
        nav = navigator.locate(image, self.intrinsics, pose.rotation)
        cost = time.perf_counter() - start

        estimate = error = in_map = None
        if nav.position is not None:
            estimate = nav.position.tolist()
            error = (nav.position - pose.position).tolist()
            in_map = map_frame_error(error, self.map_offset)
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
            map_keypoints=len(navigator.map_features["keypoints"][0]),
            map_offset_m=[self.map_offset.east_m, self.map_offset.north_m],
            map_offset_source=self.map_offset.source,
            map_frame_error_vector=in_map,
            map_frame_error_m=None if in_map is None else float(np.linalg.norm(in_map)),
            matching_map=self.matching_map,
        )
        return Run(record, image, nav)

    def _navigator(self, map_keypoints: int | None) -> Navigator:
        """The real navigator, or one thinned to `map_keypoints`, built once per K so it isn't charged to a sample."""
        if map_keypoints is None:
            return self.navigator
        if map_keypoints not in self._thinned:
            self._thinned[map_keypoints] = self.navigator.thinned(map_keypoints)
        return self._thinned[map_keypoints]


def parse_fidelity(fidelity: str) -> tuple[str, int | None]:
    """A level's renderer and map keypoint count: "splat-k8000" is ("splat", 8000), "splat" is ("splat", None)."""
    renderer, sep, k = fidelity.rpartition("-k")
    if not sep:
        return fidelity, None
    if not k.isdigit() or int(k) < 1:
        raise ValueError(f"fidelity {fidelity!r}: after '-k' comes a whole number of map keypoints, at least 1")
    return renderer, int(k)


def load_matching_map(data_dir: Path, source: str) -> OrbitalMap:
    """The workflow's matching map by name: "ctx", the LVS map with CTX elevation, or "scene", the scene's own."""
    data_dir = Path(data_dir)
    mesh_json = data_dir / "mesh" / "mesh.json"
    if source == "ctx":
        return load_lvs_map(data_dir / "lvs_map" / "map.tif", data_dir / "scene" / "ring_dtm.tif", mesh_json)
    if source == "scene":
        return load_orbital_map(data_dir / "matching_map" / "map.tif", data_dir / "scene" / "dtm.tif", mesh_json)
    raise ValueError(f"unknown matching map {source!r}; expected 'ctx' or 'scene'")


def build_simulator(
    config_path: Path,
    data_dir: Path,
    fidelities=("mesh",),
    suns=None,
    device: str = "cuda",
    matching_map: str | None = None,
) -> Simulator:
    """Jezero at the asked fidelities, the matching map, its offset, and LCAM, from the scene config and workflow data.

    "mesh" path-traces the mesh under any of the config's suns; `suns` overrides them (a
    test's zenith sun, say). "splat" renders the splat scene of each sun that has one in
    data_dir/splat/sun<i>/ (`snakemake splat`). `matching_map` overrides the config's
    `matching_map.source`; "scene" reruns earlier results on the map they used.
    """
    scene = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_dir = Path(data_dir)
    renderers = {}
    # A thinned level ("splat-k8000") renders with its renderer; the simulator thins the map itself.
    renderer_names = {parse_fidelity(f)[0]: f for f in fidelities}  # each renderer once, with a level that asked for it
    for renderer_name, fidelity in renderer_names.items():
        if renderer_name == "mesh":
            from fidelis.render.mesh_render import MeshRenderer, config_suns

            if suns is None:
                suns = config_suns(scene)
            renderers["mesh"] = MeshRenderer(data_dir / "mesh", suns, spp=scene["simulation"]["mesh_spp"])
        elif renderer_name == "splat":
            from fidelis.render.splat_render import SplatRenderer

            checkpoints = {i: data_dir / "splat" / f"sun{i}" / "splat.pt" for i in range(len(scene["suns"]))}
            renderers["splat"] = SplatRenderer({i: p for i, p in checkpoints.items() if p.exists()}, device=device)
        else:
            raise ValueError(f"unknown fidelity {fidelity!r}; expected 'mesh' or 'splat', maybe with '-k<K>'")
    source = matching_map or matching_map_source(scene)
    intrinsics = Intrinsics.from_fov(LCAM_PX, LCAM_PX, scene["scenario"]["camera_fov_deg"])
    navigator = Navigator(load_matching_map(data_dir, source), device=device)
    terrain = load_terrain(data_dir / "scene" / "dtm.tif", data_dir / "mesh" / "mesh.json")
    return Simulator(
        renderers,
        navigator,
        intrinsics,
        config_sha256=sha256_of(config_path),
        map_offset=load_map_offset(scene, source),
        ground_height=terrain.ground_height,
        matching_map=source,
    )


def build_mesh_simulator(config_path: Path, data_dir: Path, suns=None, device: str = "cuda") -> Simulator:
    """`build_simulator` at fidelity "mesh" alone."""
    return build_simulator(config_path, data_dir, ("mesh",), suns=suns, device=device)
