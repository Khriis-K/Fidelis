"""Multi-view mesh renders to train a splat scene on, plus held-out views to check it.

The views are the descent camera itself: each is the true camera of a scenario drawn
from the scene config (position, altitude, tilt, heading), so the splat learns the scene
from exactly the range of views the simulator will ask it for, oblique glimpses of the
CTX ring included. Training and held-out views use seed ranges of their own, at or
above MC_SEED_LIMIT, so no Monte Carlo descent ever renders a pose the splat trained on.

    uv run fidelis-splat-views config/jezero.yaml data/jezero data/jezero/splat/sun0/views --sun-index 0

writes train.npy and heldout.npy (float16 image stacks, views x height x width: one file
each, not thousands of small ones, for CARC's storage) and views.json (seeds, camera
poses, intrinsics, sun).
"""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from fidelis.render.camera import CameraPose, Intrinsics
from fidelis.scenario import ScenarioConfig, draw, load_config
from fidelis.terrain.fetch import sha256_of

# Monte Carlo seeds stay below this; view seeds start at it.
MC_SEED_LIMIT = 1_000_000_000
SPLIT_SEEDS = {"train": MC_SEED_LIMIT, "heldout": MC_SEED_LIMIT + 100_000_000}


def view_seeds(split: str, count: int) -> range:
    if split not in SPLIT_SEEDS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_SEEDS)}")
    return range(SPLIT_SEEDS[split], SPLIT_SEEDS[split] + count)


def view_poses(config: ScenarioConfig, seeds: Sequence[int], ground_height: Callable[[float, float], float]):
    """The true camera of each seed's scenario, `ground_height(east, north)` giving the ground beneath it."""
    from fidelis.simulation import scenario_pose

    poses = []
    for seed in seeds:
        s = draw(config, seed)
        poses.append(scenario_pose(s, ground_height(s.offset_east_m, s.offset_north_m)))
    return poses


def render_views(renderer, poses: Sequence[CameraPose], intrinsics: Intrinsics, sun_index: int, seeds) -> np.ndarray:
    """Each pose rendered under the one sun, its render noise seeded by its view seed."""
    images = np.empty((len(poses), intrinsics.height, intrinsics.width), np.float16)
    for i, (pose, seed) in enumerate(zip(poses, seeds, strict=True)):
        images[i] = renderer(pose, intrinsics, sun_index, seed)
    return images


@dataclass(frozen=True)
class Views:
    seeds: list[int]
    poses: list[CameraPose]
    images: np.ndarray  # views x height x width, float16
    intrinsics: Intrinsics
    sun_index: int


def write_views(out_dir: Path, splits: dict, intrinsics: Intrinsics, sun_index: int, meta: dict) -> None:
    """`splits` maps a split name to (seeds, poses, images). Each file is written to a temp name, then renamed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = dict(meta, sun_index=sun_index, intrinsics=asdict(intrinsics))
    for split, (seeds, poses, images) in splits.items():
        tmp = out_dir / f"{split}.tmp.npy"
        np.save(tmp, np.asarray(images, np.float16))
        tmp.replace(out_dir / f"{split}.npy")
        record[split] = {
            "seeds": list(seeds),
            "rotation": [p.rotation.tolist() for p in poses],
            "position": [p.position.tolist() for p in poses],
        }
    tmp = out_dir / "views.tmp.json"
    tmp.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out_dir / "views.json")


def load_views(views_dir: Path, split: str) -> Views:
    views_dir = Path(views_dir)
    record = json.loads((views_dir / "views.json").read_text(encoding="utf-8"))
    part = record[split]
    poses = [CameraPose(np.array(r), np.array(p)) for r, p in zip(part["rotation"], part["position"], strict=True)]
    return Views(
        seeds=part["seeds"],
        poses=poses,
        images=np.load(views_dir / f"{split}.npy"),
        intrinsics=Intrinsics(**record["intrinsics"]),
        sun_index=record["sun_index"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("data_dir", type=Path, help="the workflow's scene data, e.g. data/jezero")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--sun-index", type=int, required=True)
    args = parser.parse_args()

    from fidelis.nav.orbital_map import load_orbital_map
    from fidelis.render.mesh_render import MeshRenderer, Sun
    from fidelis.simulation import LCAM_PX

    scene = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = scene["splat"]["views"]
    data = args.data_dir
    ground = load_orbital_map(
        data / "matching_map" / "map.tif", data / "scene" / "dtm.tif", data / "mesh" / "mesh.json"
    )
    suns = [Sun(s["azimuth_deg"], s["elevation_deg"]) for s in scene["suns"]]
    renderer = MeshRenderer(data / "mesh", suns, spp=settings["spp"])
    intrinsics = Intrinsics.from_fov(LCAM_PX, LCAM_PX, scene["scenario"]["camera_fov_deg"])
    config = load_config(args.config)

    splits = {}
    for split in ("train", "heldout"):
        seeds = view_seeds(split, settings[split])
        poses = view_poses(config, seeds, ground.ground_height)
        splits[split] = (seeds, poses, render_views(renderer, poses, intrinsics, args.sun_index, seeds))
        print(f"{split}: {len(seeds)} views rendered")
    meta = {"spp": settings["spp"], "sun": scene["suns"][args.sun_index], "config_sha256": sha256_of(args.config)}
    write_views(args.out_dir, splits, intrinsics, args.sun_index, meta)
    return 0
