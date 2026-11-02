"""Check the built Jezero mesh against its tile: the numbers in docs/benchmarks/terrain-mesh.md.

1. Elevation: 10,000 random vertices, read straight from the PLY bytes (not through
   Mitsuba), against the DTM sampled at their map coordinates.
2. Registration: a top-down orthographic Mitsuba render (uniform sky, 2 m/px) against the
   orthoimage averaged to 2 m. A misplaced or flipped texture drops the correlation and
   shows up as a shift. Uses the GPU for a few seconds.

    uv run python scripts/check_terrain_mesh.py data/jezero/mesh data/jezero/tile data/jezero/mesh/render_vs_ortho.png
"""

import json
import sys
from pathlib import Path

import cv2
import mitsuba as mi
import numpy as np
import rasterio

from fidelis.terrain.matching_map import block_mean
from fidelis.terrain.mesh import mitsuba_shape

mesh_dir, tile_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rec = json.loads((mesh_dir / "mesh.json").read_text())
ox, oy = rec["origin"]

# Elevation: parse the PLY ourselves (independent of Mitsuba) and sample 10,000 vertices.
raw = (mesh_dir / "mesh.ply").read_bytes()
body = raw.index(b"end_header\n") + len(b"end_header\n")
v = np.frombuffer(raw, dtype="<f4", count=rec["vertex_count"] * 5, offset=body).reshape(-1, 5)
with rasterio.open(tile_dir / "dtm.tif") as dtm:
    rng = np.random.default_rng(1)
    k = rng.choice(len(v), 10_000, replace=False)
    pts = [(x + ox, y + oy) for x, y in v[k, :2]]
    sampled = np.array([s[0] for s in dtm.sample(pts)])
err = np.abs(v[k, 2] - sampled)
print(
    f"elevation: 10,000 sampled vertices, max |mesh - DTM| = {err.max():.2e} m,",
    f"heights {v[:, 2].min():.1f}..{v[:, 2].max():.1f} m",
)

# Registration: top-down orthographic render under a uniform sky, 1000 x 1000 px at 2 m/px.
mi.set_variant("cuda_ad_rgb")
half = 1000.0
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
            .scale([half, half, 1]),
            "near_clip": 1,
            "far_clip": 10000,
            "film": {"type": "hdrfilm", "width": 1000, "height": 1000, "pixel_format": "luminance"},
            "sampler": {"type": "independent", "sample_count": 16},
        },
    }
)
img = np.array(mi.render(scene))[..., 0]
with rasterio.open(tile_dir / "ortho.tif") as ortho_src:
    ortho = ortho_src.read(1).astype(np.float32) / 255
    px_per_m = 1 / ortho_src.res[0]
left, _, right, _ = rec["bounds"]
margin = round(((right - left) / 2 - half) * px_per_m)  # crop the ortho to the rendered square
ref = block_mean(ortho[margin:-margin, margin:-margin], round(2 * half / img.shape[0] * px_per_m))
corr = np.corrcoef(img.ravel(), ref.ravel())[0, 1]
(dx, dy), _ = cv2.phaseCorrelate(ref.astype(np.float64), img.astype(np.float64))
# Controls: a mirrored texture would look like these.
flip_ns = np.corrcoef(img[::-1].ravel(), ref.ravel())[0, 1]
flip_ew = np.corrcoef(img[:, ::-1].ravel(), ref.ravel())[0, 1]
print(
    f"registration: render vs ortho correlation {corr:.3f}",
    f"(mirrored north-south: {flip_ns:.3f}, east-west: {flip_ew:.3f});",
    f"shift {dx:+.2f}, {dy:+.2f} px at 2 m/px",
)
side = np.concatenate([ref, img / max(img.max(), 1e-6) * ref.max()], axis=1)
cv2.imwrite(str(out), np.clip(side * 255, 0, 255).astype(np.uint8))
