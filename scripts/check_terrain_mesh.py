"""Check the built Jezero mesh against its tile: the numbers in docs/benchmarks/terrain-mesh.md.

1. Elevation: 10,000 random vertices, read straight from the PLY bytes (not through
   Mitsuba), against the DTM sampled at their map coordinates.
2. Registration, in ground metres: a top-down orthographic Mitsuba render (uniform sky,
   9 km square at 4 m/px) against the orthoimage area-averaged onto the same ground grid.
   A misplaced or flipped texture drops the correlation and shows up as a shift, and so
   does a wrong east-west scale: the control treats map metres as ground metres. Uses
   the GPU for a few seconds.

    uv run python scripts/check_terrain_mesh.py data/jezero/mesh data/jezero/tile data/jezero/mesh/render_vs_ortho.png
"""

import json
import sys
from pathlib import Path

import cv2
import mitsuba as mi
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

from fidelis.render.mesh_render import require_no_blank_lines
from fidelis.terrain.mesh import mitsuba_shape

mesh_dir, tile_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rec = json.loads((mesh_dir / "mesh.json").read_text())
ox, oy = rec["origin"]
scale = rec["east_scale"]

# Elevation: parse the PLY ourselves (independent of Mitsuba) and sample 10,000 vertices.
raw = (mesh_dir / "mesh.ply").read_bytes()
body = raw.index(b"end_header\n") + len(b"end_header\n")
v = np.frombuffer(raw, dtype="<f4", count=rec["vertex_count"] * 5, offset=body).reshape(-1, 5)
with rasterio.open(tile_dir / "dtm.tif") as dtm:
    rng = np.random.default_rng(1)
    k = rng.choice(len(v), 10_000, replace=False)
    pts = [(x / scale + ox, y + oy) for x, y in v[k, :2]]
    sampled = np.array([s[0] for s in dtm.sample(pts)])
err = np.abs(v[k, 2] - sampled)
print(
    f"elevation: 10,000 sampled vertices, max |mesh - DTM| = {err.max():.2e} m,",
    f"heights {v[:, 2].min():.1f}..{v[:, 2].max():.1f} m",
)
print(f"extent: x {v[:, 0].min():.1f}..{v[:, 0].max():.1f} m, y {v[:, 1].min():.1f}..{v[:, 1].max():.1f} m (ground)")

# Registration: top-down orthographic render under a uniform sky, 9 km of ground at 4 m/px,
# inside the mesh's 9.50 km of ground east-west.
PX_M = 4.0
HALF_M = 4500.0
n = round(2 * HALF_M / PX_M)
mi.set_variant("cuda_ad_rgb")
scene = mi.load_dict(
    {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 2},
        "sky": {"type": "constant", "radiance": 1.0},
        "terrain": mitsuba_shape(mesh_dir),
        "cam": {
            "type": "orthographic",
            "to_world": mi.ScalarTransform4f()
            .look_at(origin=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0])
            .scale([HALF_M, HALF_M, 1]),
            "near_clip": 1,
            "far_clip": 10000,
            "film": {"type": "hdrfilm", "width": n, "height": n, "pixel_format": "luminance"},
            "sampler": {"type": "independent", "sample_count": 16},
        },
    }
)
img = np.array(mi.render(scene))[..., 0]
require_no_blank_lines(img)  # a silent out-of-memory render would fail here, not as a low correlation


def ortho_on_grid(east_scale: float) -> np.ndarray:
    """The orthoimage area-averaged onto the render's ground grid, if map x = ground x / east_scale."""
    with rasterio.open(tile_dir / "ortho.tif") as src:
        grid = np.zeros((n, n), dtype=np.float32)
        reproject(
            src.read(1).astype(np.float32) / 255,
            grid,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=Affine(PX_M / east_scale, 0, ox - HALF_M / east_scale, 0, -PX_M, oy + HALF_M),
            dst_crs=src.crs,
            resampling=Resampling.average,
        )
    return grid


def corr(a, b):
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


ref = ortho_on_grid(scale)
(dx, dy), _ = cv2.phaseCorrelate(ref.astype(np.float64), img.astype(np.float64))
# Controls: a mirrored texture, or a mesh left in map metres, would look like these.
print(
    f"registration: render vs ortho correlation {corr(img, ref):.3f}",
    f"(mirrored north-south: {corr(img[::-1], ref):.3f}, east-west: {corr(img[:, ::-1], ref):.3f};",
    f"ortho at map metres, no east-west scale: {corr(img, ortho_on_grid(1.0)):.3f});",
    f"shift {dx:+.2f}, {dy:+.2f} px at {PX_M:.0f} m/px",
)
side = np.concatenate([ref, img / max(img.max(), 1e-6) * ref.max()], axis=1)
cv2.imwrite(str(out), np.clip(side * 255, 0, 255).astype(np.uint8))
