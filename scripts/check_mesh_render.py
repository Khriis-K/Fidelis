"""Render the Jezero mesh as the descent camera sees it: the numbers in docs/benchmarks/mesh-render.md.

A nadir LCAM-like camera (1024 x 1024 px, 90 deg field of view) at the top and bottom
of the LVS imaging band, 4.2 and 2.2 km above the tile's mean ground, under five suns
(east and west at 10 and 30 deg, south at 60 deg). For each sun: scene load time, GPU
memory, and per altitude render time per image, same-seed and other-seed differences,
and noise against a 1024 spp reference. Every render must pass the blank-row check.
Writes one PNG per sun and altitude into OUT_DIR, plus two panels of the central
256 x 256 px at 2.2 km: shadows_low_sun.png (east and west at 10 deg) and shading.png
(the other three).

GPU memory is the whole GPU's use from nvidia-smi, minus its use before the first scene
loaded, so other processes on the GPU count too.

    uv run python scripts/check_mesh_render.py data/jezero/mesh data/jezero/render
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import mitsuba as mi
import numpy as np

from fidelis.render.camera import Intrinsics, look_at
from fidelis.render.mesh_render import Sun, render, require_no_blank_rows, terrain_scene

mesh_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
mi.set_variant("cuda_ad_rgb")


def gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


rec = json.loads((mesh_dir / "mesh.json").read_text())
raw = (mesh_dir / "mesh.ply").read_bytes()
body = raw.index(b"end_header\n") + len(b"end_header\n")
z = np.frombuffer(raw, dtype="<f4", count=rec["vertex_count"] * 5, offset=body).reshape(-1, 5)[:, 2]
ground = float(z.mean())
del raw

# LVS imaged from 4.2 km down to 2.2 km (NASA/JPL landing press kit, pp. 21-22).
# A 90 deg view sees 2 x altitude across: 8.4 and 4.4 km of the 10 km tile.
ALTITUDES = {"4.2km": 4200.0, "2.2km": 2200.0}
SPP = 64
poses = {
    name: look_at(eye=[0.0, 0.0, ground + alt], target=[0.0, 0.0, ground], up=[0.0, 1.0, 0.0])
    for name, alt in ALTITUDES.items()
}
k = Intrinsics.from_fov(width=1024, height=1024, fov_x_deg=90.0)
# Few slopes here are steeper than 30 deg, so only the 10 deg suns cast many shadows.
suns = {
    "east-10": Sun(90.0, 10.0),
    "west-10": Sun(270.0, 10.0),
    "east-30": Sun(90.0, 30.0),
    "west-30": Sun(270.0, 30.0),
    "south-60": Sun(180.0, 60.0),
}
print(
    f"mesh {rec['vertex_count']} vertices, {rec['face_count']} triangles; mean ground {ground:.1f} m;",
    f"{k.width} x {k.height} px, 90 deg, {SPP} spp",
)


def checked(image):
    require_no_blank_rows(image)
    return image


baseline = gpu_used_mib()
images = {}
for name, sun in suns.items():
    t0 = time.perf_counter()
    scene = terrain_scene(mesh_dir, sun)
    load_s = time.perf_counter() - t0
    loaded = gpu_used_mib()
    print(f"{name}: load {load_s:.1f} s, GPU +{loaded - baseline} MiB after load")

    for alt_name, pose in poses.items():
        t0 = time.perf_counter()
        first = checked(render(scene, pose, k, spp=SPP, seed=0))  # includes Dr.Jit kernel compilation
        first_s = time.perf_counter() - t0
        times = []
        for seed in range(1, 6):
            t0 = time.perf_counter()
            checked(render(scene, pose, k, spp=SPP, seed=seed))
            times.append(time.perf_counter() - t0)

        again = checked(render(scene, pose, k, spp=SPP, seed=0))
        other = checked(render(scene, pose, k, spp=SPP, seed=1))
        t0 = time.perf_counter()
        ref = checked(render(scene, pose, k, spp=1024, seed=100))
        ref_s = time.perf_counter() - t0

        lit = np.median(first)
        dark = (first < 0.1 * lit).mean()
        print(
            f"  {alt_name}: first render {first_s:.2f} s, then median {np.median(times):.3f} s",
            f"(min {min(times):.3f}, max {max(times):.3f}); 1024 spp {ref_s:.2f} s;",
            f"GPU +{gpu_used_mib() - baseline} MiB;",
            f"same seed max |diff| {np.abs(first - again).max():.2e};",
            f"other seed RMS diff {np.sqrt(((first - other) ** 2).mean()) / lit:.3%} of median;",
            f"RMS vs 1024 spp {np.sqrt(((first - ref) ** 2).mean()) / lit:.3%} of median;",
            f"pixels below 10% of median (shadow): {dark:.1%}",
        )
        images[name, alt_name] = ref
    del scene

# One exposure for all suns and altitudes, so brightness differences between them are real.
scale = 255 / np.percentile(np.concatenate([im.ravel() for im in images.values()]), 99.5)
for (name, alt_name), im in images.items():
    cv2.imwrite(str(out_dir / f"nadir_{alt_name}_{name}.png"), np.clip(im * scale, 0, 255).astype(np.uint8))
crop = (slice(384, 640), slice(384, 640))  # the central 256 x 256 px, about 1.1 km across at 2.2 km
panels = {"shadows_low_sun.png": ["east-10", "west-10"], "shading.png": ["east-30", "west-30", "south-60"]}
for file, names in panels.items():
    # Each panel gets its own exposure: the 10 deg suns are about 3x dimmer than 30 deg.
    views = [images[n, "2.2km"][crop] for n in names]
    s = 255 / np.percentile(np.concatenate([v.ravel() for v in views]), 99.5)
    panel = np.concatenate([np.clip(v * s, 0, 255) for v in views], axis=1).astype(np.uint8)
    cv2.imwrite(str(out_dir / file), cv2.resize(panel, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
print(f"wrote {out_dir}")
