"""Render the Jezero mesh as the descent camera sees it: the numbers in docs/benchmarks/mesh-render.md.

A nadir LCAM-like camera (1024 x 1024 px, 90 deg field of view) 900 m above the tile's
mean ground, under five suns (east and west at 10 and 30 deg, south at 60 deg). For each
sun: scene load time, render time per image, same-seed and other-seed differences, and
noise against a 1024 spp reference. Writes one PNG per sun into OUT_DIR, plus two panels
of the central 256 x 256 px: shadows_low_sun.png (east and west at 10 deg) and
shading.png (the other three).

    uv run python scripts/check_mesh_render.py data/jezero/mesh data/jezero/render
"""

import json
import sys
import time
from pathlib import Path

import cv2
import mitsuba as mi
import numpy as np

from fidelis.render.camera import Intrinsics, look_at
from fidelis.render.mesh_render import Sun, render, terrain_scene

mesh_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
mi.set_variant("cuda_ad_rgb")

rec = json.loads((mesh_dir / "mesh.json").read_text())
raw = (mesh_dir / "mesh.ply").read_bytes()
body = raw.index(b"end_header\n") + len(b"end_header\n")
z = np.frombuffer(raw, dtype="<f4", count=rec["vertex_count"] * 5, offset=body).reshape(-1, 5)[:, 2]
ground = float(z.mean())

ALTITUDE = 900.0  # 90 deg FOV sees 2 x altitude across: 1.8 km of the 2.0 km mesh
SPP = 64
pose = look_at(eye=[0.0, 0.0, ground + ALTITUDE], target=[0.0, 0.0, ground], up=[0.0, 1.0, 0.0])
k = Intrinsics.from_fov(width=1024, height=1024, fov_x_deg=90.0)
# Few slopes here are steeper than 30 deg, so only the 10 deg suns cast many shadows.
suns = {
    "east-10": Sun(90.0, 10.0),
    "west-10": Sun(270.0, 10.0),
    "east-30": Sun(90.0, 30.0),
    "west-30": Sun(270.0, 30.0),
    "south-60": Sun(180.0, 60.0),
}
print(f"camera {ALTITUDE:.0f} m above mean ground {ground:.1f} m, {k.width} x {k.height} px, 90 deg, {SPP} spp")

images = {}
for name, sun in suns.items():
    t0 = time.perf_counter()
    scene = terrain_scene(mesh_dir, sun)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    first = render(scene, pose, k, spp=SPP, seed=0)  # includes Dr.Jit kernel compilation
    first_s = time.perf_counter() - t0
    times = []
    for seed in range(1, 6):
        t0 = time.perf_counter()
        render(scene, pose, k, spp=SPP, seed=seed)
        times.append(time.perf_counter() - t0)

    again = render(scene, pose, k, spp=SPP, seed=0)
    other = render(scene, pose, k, spp=SPP, seed=1)
    t0 = time.perf_counter()
    ref = render(scene, pose, k, spp=1024, seed=100)
    ref_s = time.perf_counter() - t0

    lit = np.median(first)
    dark = (first < 0.1 * lit).mean()
    print(
        f"{name}: load {load_s:.1f} s, first render {first_s:.2f} s, then median {np.median(times):.3f} s",
        f"(min {min(times):.3f}, max {max(times):.3f}); 1024 spp {ref_s:.2f} s;",
        f"same seed max |diff| {np.abs(first - again).max():.2e};",
        f"other seed RMS diff {np.sqrt(((first - other) ** 2).mean()) / lit:.3%} of median;",
        f"RMS vs 1024 spp {np.sqrt(((first - ref) ** 2).mean()) / lit:.3%} of median;",
        f"pixels below 10% of median (shadow): {dark:.1%}",
    )
    images[name] = ref

# One exposure for all suns, so brightness differences between them are real.
scale = 255 / np.percentile(np.concatenate([im.ravel() for im in images.values()]), 99.5)
for name, im in images.items():
    cv2.imwrite(str(out_dir / f"nadir_{name}.png"), np.clip(im * scale, 0, 255).astype(np.uint8))
crop = (slice(384, 640), slice(384, 640))  # the central 256 x 256 px, about 450 m across
panels = {"shadows_low_sun.png": ["east-10", "west-10"], "shading.png": ["east-30", "west-30", "south-60"]}
for file, names in panels.items():
    # Each panel gets its own exposure: the 10 deg suns are about 3x dimmer than 30 deg.
    s = 255 / np.percentile(np.concatenate([images[n][crop].ravel() for n in names]), 99.5)
    panel = np.concatenate([np.clip(images[n][crop] * s, 0, 255) for n in names], axis=1).astype(np.uint8)
    cv2.imwrite(str(out_dir / file), cv2.resize(panel, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
print(f"wrote {out_dir}")
