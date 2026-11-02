"""Where matching breaks as the map coarsens: the numbers in docs/benchmarks/navigator.md.

Perfect-conditions descent renders of the Jezero mesh (nadir, north up, zenith sun, 1024
spp, no noise or blur) at the top and bottom of the LVS band, 4.2 and 2.2 km above the
tile's mean ground, located against the orthoimage averaged to 2, 4, 6, 10 and 12 m
pixels. 2 m is the finest map there is: it's the mesh texture's own resolution, so a
finer map would hold no detail the render has. 6 m is the CTX-like map.

The sweep's maps cover the same 6 km square around the tile centre at every
resolution, because the whole tile at 2 m (5010 px, about 100,000 keypoints) doesn't
fit LightGlue on an 8 GB GPU. Then the full-tile 6 m map from the workflow, as the
navigator uses it, under the zenith sun, under the landing-day sun, and at headings of
45, 90 and 180 degrees, which perfect conditions leave out.

    uv run python scripts/check_navigator.py data/jezero
"""

import sys
import time
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import rasterio
import torch
from rasterio.transform import Affine

from fidelis.nav.navigator import Navigator
from fidelis.nav.orbital_map import OrbitalMap, load_orbital_map
from fidelis.render.camera import Intrinsics, look_at
from fidelis.render.mesh_render import Sun, render, require_no_blank_lines, terrain_scene
from fidelis.terrain.matching_map import block_mean

data = Path(sys.argv[1])
mi.set_variant("cuda_ad_rgb")

full = load_orbital_map(data / "matching_map" / "map.tif", data / "tile" / "dtm.tif", data / "mesh" / "mesh.json")
with rasterio.open(data / "tile" / "ortho.tif") as src:
    ortho, ortho_transform = src.read(1), src.transform
ground = float(full.dtm.mean())
k = Intrinsics.from_fov(1024, 1024, 90)
OFFSET = np.array([150.0, -230.0])  # as the GPU test: off the tile centre, on no pixel grid
ZENITH = Sun(0.0, 90.0)
LANDING = Sun(257.0, 40.0)  # the landing-day sun suggested for the splat scenes (config/jezero.yaml)


def renders(sun, views):
    """{name: (truth, image)}; views maps a name to (altitude, heading in degrees)."""
    scene = terrain_scene(data / "mesh", sun)
    out = {}
    for name, (altitude, heading) in views.items():
        truth = np.array([*OFFSET, ground + altitude])
        h = np.radians(heading)
        image = render(scene, look_at(truth, truth - [0, 0, 1], [np.sin(h), np.cos(h), 0]), k, spp=1024)
        require_no_blank_lines(image)
        # The 99.9th percentile to full scale; the zenith sun's albedo image already sits in [0, 1].
        out[name] = truth, np.clip(image / max(1.0, np.percentile(image, 99.9)), 0, 1)
    del scene
    dr.flush_malloc_cache()
    return out


def window_map(factor, half_px=1500):
    """The 6 km square around the tile centre, averaged over factor x factor blocks of 2 m pixels."""
    c = ortho.shape[0] // 2
    window = ortho[c - half_px : c + half_px, c - half_px : c + half_px]
    transform = ortho_transform @ Affine.translation(c - half_px, c - half_px) @ Affine.scale(factor)
    return OrbitalMap(
        image=block_mean(window, factor).astype(np.float32) / 255,
        transform=transform,
        dtm=full.dtm,
        dtm_transform=full.dtm_transform,
        origin=full.origin,
        east_scale=full.east_scale,
    )


def report(label, navigator, truth, image):
    t0 = time.perf_counter()
    result = navigator.locate(image, k)
    took = time.perf_counter() - t0
    err = "-" if result.position is None else f"{np.linalg.norm(result.position - truth):.2f}"
    print(
        f"| {label} | {len(navigator.map_features['keypoints'][0])} | {result.match_count} |"
        f" {result.inlier_count} | {result.status} | {err} | {took:.2f} |",
        flush=True,
    )


header = (
    "| Map | Map keypoints | Matches | Inliers | Status | Position error (m) | Locate (s) |\n"
    "|---|---|---|---|---|---|---|"
)
band = {"4.2 km": (4200.0, 0.0), "2.2 km": (2200.0, 0.0)}
zenith = renders(ZENITH, band | {f"2.2 km, heading {h}": (2200.0, float(h)) for h in (45, 90, 180)})
landing = renders(LANDING, {"2.2 km": (2200.0, 0.0)})

print(f"{torch.cuda.get_device_name()}; mean ground {ground:.1f} m; truth offset {OFFSET.tolist()} m")
for alt in band:
    print(f"\nSweep, {alt}, zenith sun, 6 km map window\n{header}")
    for factor in (1, 2, 3, 5, 6):
        report(f"{2 * factor} m", Navigator(window_map(factor)), *zenith[alt])
        torch.cuda.empty_cache()

print(f"\nFull-tile 6 m map\n{header}")
navigator = Navigator(full)
for alt in band:
    report(f"{alt}, zenith sun", navigator, *zenith[alt])
report("2.2 km, landing-day sun (257, 40)", navigator, *landing["2.2 km"])
for h in (45, 90, 180):
    report(f"2.2 km, zenith sun, heading {h}", navigator, *zenith[f"2.2 km, heading {h}"])
print(f"\npeak torch GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
