"""Render the wide Jezero scene (issue #33): the numbers in docs/benchmarks/wide-scene.md.

Loads the HiRISE tile and CTX ring under the landing-day sun (config sun 0), then:

- scene load time and GPU memory (the whole GPU's use from nvidia-smi, less its use
  before loading, so other processes on the GPU count too);
- render time per 1024 x 1024 image at 64 spp and at the simulator's mesh_spp, over the
  sampler's worst cases: highest altitude, largest tilt, leaning outward from the largest
  offset, in four directions. Every render must pass the blank-line check, which also
  fails a view that runs off the scene along a whole image edge;
- seam visibility: a nadir view at the bottom of the LVS band, straight above the tile's
  east edge, north up, so the seam is the image's middle column. The mean brightness step
  and mean horizontal gradient across that column are compared with the same statistics
  at 40 control columns away from it (the null case: render noise and real terrain).

Writes a PNG per view and wide_scene.json into OUT_DIR.

    uv run python scripts/check_wide_scene.py config/jezero.yaml data/jezero OUT_DIR
"""

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import mitsuba as mi
import numpy as np
import yaml

from fidelis.nav.orbital_map import load_orbital_map
from fidelis.render.camera import Intrinsics, look_at
from fidelis.render.mesh_render import Sun, render, require_no_blank_lines, terrain_scene
from fidelis.scenario import draw, load_config
from fidelis.simulation import LCAM_PX, scenario_pose

config_path, data, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
scene_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
mi.set_variant("cuda_ad_rgb")


def gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


ground = load_orbital_map(data / "matching_map" / "map.tif", data / "scene" / "dtm.tif", data / "mesh" / "mesh.json")
k = Intrinsics.from_fov(LCAM_PX, LCAM_PX, scene_cfg["scenario"]["camera_fov_deg"])
spp_sim = scene_cfg["simulation"]["mesh_spp"]
results = {}

before = gpu_used_mib()
t0 = time.perf_counter()
scene = terrain_scene(data / "mesh", Sun(**scene_cfg["suns"][0]))
results["load_s"] = time.perf_counter() - t0
render(scene, look_at([0, 0, 0], [0, 0, -1], [0, 1, 0]), k, spp=1)  # warm-up: compiles kernels
results["gpu_mib"] = gpu_used_mib() - before

# The sampler's worst cases, built from a drawn scenario with its extremes swapped in.
cfg = load_config(config_path)
base = draw(cfg, 0)
alt, tilt = cfg.altitude_m.bounds()[1], cfg.tilt_deg.bounds()[1]
east, north = cfg.offset_east_m.bounds()[1], cfg.offset_north_m.bounds()[1]
views = {}
for name, (dx, dy, az) in {
    "east": (east, 0, 90),
    "north": (0, north, 0),
    "west": (-east, 0, 270),
    "south": (0, -north, 180),
}.items():
    s = replace(base, offset_east_m=dx, offset_north_m=dy, altitude_m=alt, tilt_deg=tilt, tilt_azimuth_deg=az)
    # Heading 45 deg off the lean puts an image corner, the farthest-reaching ray, outward.
    s = replace(s, heading_deg=(az + 45) % 360)
    views[f"worst_{name}"] = scenario_pose(s, ground.ground_height(dx, dy))

# Straight above the tile mesh's east edge (its outermost DTM pixel centres), north up.
mesh = json.loads((data / "mesh" / "mesh.json").read_text())
edge_east = (mesh["bounds"][2] - 2.0 - mesh["origin"][0]) * mesh["east_scale"]
bottom = scene_cfg["lvs_altitude_m"]["bottom"]
seam_pose = look_at(
    [edge_east, 0.0, ground.ground_height(edge_east, 0.0) + bottom], [edge_east, 0.0, -1e4], [0.0, 1.0, 0.0]
)
views["seam_nadir"] = seam_pose

times = {64: [], spp_sim: []}
images = {}
for name, pose in views.items():
    for spp in times:
        t0 = time.perf_counter()
        image = render(scene, pose, k, spp=spp, seed=1)
        times[spp].append(time.perf_counter() - t0)
    require_no_blank_lines(image)
    images[name] = image
    cv2.imwrite(
        str(out_dir / f"{name}.png"), np.clip(image / np.percentile(image, 99.5) * 255, 0, 255).astype(np.uint8)
    )
results["render_s"] = {str(spp): {"median": float(np.median(t)), "max": float(np.max(t))} for spp, t in times.items()}
results["worst_case"] = {"altitude_m": alt, "tilt_deg": tilt, "offset_east_m": east, "offset_north_m": north}


def column_stats(image: np.ndarray, col: float, width: int = 8) -> tuple[float, float]:
    """Brightness step (right minus left mean, over `width` columns each side) and mean |gradient| at col."""
    c = int(round(col))
    step = image[:, c : c + width].mean() - image[:, c - width : c].mean()
    gradient = np.abs(np.diff(image[:, c - 1 : c + 1], axis=1)).mean()
    return float(step), float(gradient)


seam = images["seam_nadir"] / images["seam_nadir"].mean()
seam_col = (LCAM_PX - 1) / 2 + 0.5  # the seam runs between the two middle columns
step, gradient = column_stats(seam, seam_col)
controls = [column_stats(seam, c) for c in np.r_[np.linspace(60, 400, 20), np.linspace(624, 964, 20)]]
control_steps = np.abs([c[0] for c in controls])
control_gradients = np.array([c[1] for c in controls])
results["seam"] = {
    "step": step,
    "gradient": gradient,
    "control_step_abs": {"median": float(np.median(control_steps)), "max": float(control_steps.max())},
    "control_gradient": {"median": float(np.median(control_gradients)), "max": float(control_gradients.max())},
    "note": "statistics of the image divided by its mean brightness",
}

(out_dir / "wide_scene.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
