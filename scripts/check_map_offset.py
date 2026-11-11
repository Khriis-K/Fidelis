"""Check the whole-map offset measurement on the real pair (#40): docs/benchmarks/map-offset.md.

- Window size: the offset at 64, 128 and 256 CTX-pixel windows (384 m to 1.5 km).
- Injection: the CTX geotransform is moved by a known amount (exact, no resampling) and
  the measured offset must move by the same amount. Fractions of a 6 m pixel show any
  pull towards whole pixels; this is the method's own accuracy on real texture.
- Halves: the offset over the west, east, south and north halves of the windows.
- Residual correlation between neighbouring windows: if high, the windows aren't
  independent and their standard error understates the uncertainty.

Prints a table; writes nothing but a temporary CTX copy in OUT_DIR.

    uv run python scripts/check_map_offset.py data/jezero/tile/ortho.tif data/jezero/raw/ctx_ortho.tif OUT_DIR
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from fidelis.terrain.map_offset import measure_offset

tile, ctx, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)


def line(tag, r):
    t = r["trend_m_per_km"]
    trends = " ".join(f"{k} {v['value']:+.3f}+-{v['se']:.3f}" for k, v in t.items())
    print(
        f"{tag}: east {r['east_m']:+.3f} north {r['north_m']:+.3f} m; sd {r['east_sd_m']:.2f}/{r['north_sd_m']:.2f};"
        f" windows {r['windows_used']}/{r['windows_total']}; residual rms {r['residual_rms_m']:.2f} p99"
        f" {r['residual_p99_m']:.2f} m; trends (m/km) {trends}"
    )


for window in (64, 128, 256):
    line(f"window {window}", measure_offset(tile, ctx, window))

base = measure_offset(tile, ctx, 128)
with rasterio.open(ctx) as src:
    profile, data = src.profile, src.read(1)
errors = []
for de, dn in ((1.5, 0), (3, 0), (4.5, 0), (6, 0), (0, 1.5), (0, 3), (0, 4.5), (0, -12), (20, -20), (-18, 20)):
    moved = out_dir / "ctx_injected.tif"
    moved_transform = Affine.translation(de, dn) * profile["transform"]
    with rasterio.open(moved, "w", **(profile | {"transform": moved_transform})) as dst:
        dst.write(data, 1)
    r = measure_offset(tile, moved, 128)
    error = (r["east_m"] - base["east_m"] - de, r["north_m"] - base["north_m"] - dn)
    errors += error
    print(f"inject {de:+6.1f} {dn:+6.1f} m: error {error[0]:+.2f} {error[1]:+.2f} m; windows {r['windows_used']}")
print(f"injection: max |error| {np.max(np.abs(errors)):.2f} m, rms {np.sqrt(np.mean(np.square(errors))):.2f} m")

w = base["windows"]
for name, keep in (
    ("west", lambda x: x["east_km"] < 0),
    ("east", lambda x: x["east_km"] > 0),
    ("south", lambda x: x["north_km"] < 0),
    ("north", lambda x: x["north_km"] > 0),
):
    half = [x for x in w if keep(x)]
    print(
        f"{name} half ({len(half)} windows): east {np.median([x['de_m'] for x in half]):+.2f}"
        f" north {np.median([x['dn_m'] for x in half]):+.2f} m"
    )

side = int(round(np.sqrt(len(w))))
for key, offset in (("de_m", base["east_m"]), ("dn_m", base["north_m"])):
    grid = (np.array([x[key] for x in w]) - offset).reshape(side, side)
    across = np.corrcoef(grid[:, :-1].ravel(), grid[:, 1:].ravel())[0, 1]
    down = np.corrcoef(grid[:-1, :].ravel(), grid[1:, :].ravel())[0, 1]
    print(f"{key} residual, neighbour correlation: east-west {across:.2f}, north-south {down:.2f}")
    with np.printoptions(precision=1, suppress=True, linewidth=200):
        print(f"{key} residual grid, rows north to south (m):\n{grid}")
