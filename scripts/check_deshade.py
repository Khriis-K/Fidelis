"""Relit renders with and without the baked shading: the picture and number in docs/benchmarks/deshade.md.

A nadir LCAM-like camera (1024 x 1024 px, 90 deg) 2.2 km above the tile's mean ground,
under two suns on opposite sides of the sky: the landing-day sun (257 deg, 40 deg), close
to the sun the orthoimage was taken under, and a morning sun (87 deg, 25 deg), the stress
sun #43 names. Each is rendered from the photo texture (texture.png) and from the
de-shaded one (texture_deshaded.png). Writes relit.png into OUT_DIR: the central
256 x 256 px, photo texture on the top row, de-shaded on the bottom, landing sun on the
left, morning sun on the right.

Also prints, and writes to relit.json, per texture, how alike the two suns' renders are (correlation of their
high-passed log brightness over the whole image). Baked shading doesn't move with the
sun, so it holds that number up.

    uv run python scripts/check_deshade.py data/jezero/mesh data/jezero/render/deshade
"""

import sys
from pathlib import Path

import cv2
import mitsuba as mi
import numpy as np

from fidelis.render.camera import Intrinsics, look_at
from fidelis.render.mesh_render import Sun, render, require_no_blank_lines, terrain_scene
from fidelis.terrain.mesh import read_ply_vertices
from fidelis.terrain.scene import write_json

mesh_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
mi.set_variant("cuda_ad_rgb")

ground = float(read_ply_vertices(mesh_dir / "mesh.ply")[0][:, 2].mean())

ALTITUDE_M, SPP = 2200.0, 256
HIGHPASS_PX = 23  # brightness is compared on structure finer than about 100 m: 23 px at 4.3 m per pixel
pose = look_at(eye=[0.0, 0.0, ground + ALTITUDE_M], target=[0.0, 0.0, ground], up=[0.0, 1.0, 0.0])
k = Intrinsics.from_fov(width=1024, height=1024, fov_x_deg=90.0)
suns = {"landing sun (257, 40)": Sun(257.0, 40.0), "morning sun (87, 25)": Sun(87.0, 25.0)}
textures = {"photo": "texture.png", "de-shaded": "texture_deshaded.png"}


def highpassed(image):
    log = np.log(np.maximum(image, 1e-6))
    return log - cv2.GaussianBlur(log, (0, 0), HIGHPASS_PX)


images, alike = {}, {}
for texture_name, texture in textures.items():
    for sun_name, sun in suns.items():
        scene = terrain_scene(mesh_dir, sun, texture)
        image = render(scene, pose, k, spp=SPP, seed=0)
        require_no_blank_lines(image)
        images[texture_name, sun_name] = image
        del scene
    a, b = (highpassed(images[texture_name, s]) for s in suns)
    alike[texture_name] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    print(f"{texture_name} texture: the two suns' renders correlate {alike[texture_name]:.3f}")

crop = (slice(384, 640), slice(384, 640))  # the central 256 x 256 px, about 1.1 km across
rows = []
for texture_name in textures:
    # Each view gets its own exposure: the 25 deg sun is dimmer than the 40 deg one.
    views = [images[texture_name, s][crop] for s in suns]
    rows.append(np.concatenate([np.clip(v * 255 / np.percentile(v, 99.5), 0, 255) for v in views], axis=1))
panel = np.concatenate(rows, axis=0).astype(np.uint8)
cv2.imwrite(str(out_dir / "relit.png"), cv2.resize(panel, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
write_json(
    out_dir / "relit.json",
    {"altitude_m": ALTITUDE_M, "spp": SPP, "highpass_px": HIGHPASS_PX, "suns": list(suns), "render_correlation": alike},
)
print(f"wrote {out_dir / 'relit.png'} and relit.json: rows {list(textures)}, columns {list(suns)}")
