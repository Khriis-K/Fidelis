"""Cut one working tile (DTM + matching orthoimage) out of the Jezero mosaics.

The USGS mosaics use an equirectangular projection on a sphere (Eqc, latTs0, lon0),
so lon/lat turn into map metres as arc lengths. North-south, a map metre is a ground
metre; east-west, it's cos(latitude) of one (`east_west_ground_scale`).

Each tile is area-averaged down to its configured resolution: the DTM to the mesh's
vertex spacing, the orthoimage to the texture's pixels. Tile edges are snapped to a
grid, measured from the DTM's origin, that's a whole number of pixels at the mesh,
texture and matching-map resolutions alike, so every product shares exact edges.

    uv run fidelis-tile CONFIG DTM ORTHO OUT_DIR
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds

from fidelis.terrain.matching_map import block_mean, downsample_factor

Bounds = tuple[float, float, float, float]  # left, bottom, right, top (map metres)

# Snap tolerance: float noise in metres, far below any pixel size.
EPS = 1e-6
# Source pixels read at once. A 10 km window is 40,000 x 40,000 ortho pixels (1.6 GB),
# and averaging works in float64, so the window is read in strips of about this size.
STRIP_PIXELS = 2**25


def lonlat_to_eqc(lon_deg: float, lat_deg: float, radius_m: float) -> tuple[float, float]:
    """Map metres in an equirectangular projection with lat_ts=0, lon_0=0."""
    return radius_m * math.radians(lon_deg), radius_m * math.radians(lat_deg)


def east_west_ground_scale(y_m: float, radius_m: float) -> float:
    """Ground metres per map metre east-west, at map northing y_m: cos(latitude).

    With lat_ts=0 the projection keeps every parallel as long as the equator, so it
    stretches east-west distances by 1/cos(latitude). North-south is exact.
    """
    return math.cos(y_m / radius_m)


def grid_step(resolutions_m) -> float:
    """The shortest length that's a whole number of pixels at every resolution (to 1 mm)."""
    mm = [round(r * 1000) for r in resolutions_m]
    if any(m <= 0 or abs(m - r * 1000) > EPS for m, r in zip(mm, resolutions_m, strict=True)):
        raise ValueError(f"resolutions {resolutions_m} must be positive whole millimetres")
    return math.lcm(*mm) / 1000


def tile_bounds(center_x: float, center_y: float, size_m: float, origin: tuple[float, float], step_m: float) -> Bounds:
    """A square of about size_m around the centre, grown outward to the nearest grid lines."""

    def down(v: float, o: float) -> float:
        return o + math.floor((v - o) / step_m + EPS) * step_m

    def up(v: float, o: float) -> float:
        return o + math.ceil((v - o) / step_m - EPS) * step_m

    half = size_m / 2
    ox, oy = origin
    return (
        down(center_x - half, ox),
        down(center_y - half, oy),
        up(center_x + half, ox),
        up(center_y + half, oy),
    )


def cut_tile(src_path: Path, bounds: Bounds, dst_path: Path, resolution_m: float | None = None) -> dict:
    """Copy the window under bounds into its own GeoTIFF, area-averaged to resolution_m
    (default: the source's). Nodata pixels are left out of each average, and counted.
    Return the tile's bounds, resolutions, shape and nodata count."""
    with rasterio.open(src_path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        whole = window.round_offsets().round_lengths()
        if any(abs(a - b) > EPS for a, b in zip(window.flatten(), whole.flatten(), strict=True)):
            raise ValueError(f"tile bounds {bounds} are off {src_path}'s pixel grid")
        if not whole.intersection(Window(0, 0, src.width, src.height)) == whole:
            raise ValueError(f"tile bounds {bounds} fall outside {src_path} ({tuple(src.bounds)})")
        source_res = src.res[0]
        factor = 1 if resolution_m is None else downsample_factor(source_res, resolution_m)
        if whole.width % factor or whole.height % factor:
            raise ValueError(f"tile bounds {bounds} aren't a whole number of {resolution_m} m pixels")

        strip_rows = max(1, STRIP_PIXELS // (whole.width * factor)) * factor
        strips, holes = [], 0
        for r0 in range(0, whole.height, strip_rows):
            rows = min(strip_rows, whole.height - r0)
            data = src.read(1, window=Window(whole.col_off, whole.row_off + r0, whole.width, rows))
            if src.nodata is not None:
                holes += int((data == src.nodata).sum())
            strips.append(block_mean(data, factor, nodata=src.nodata))
        data = np.concatenate(strips)
        profile = src.profile | {
            "width": data.shape[1],
            "height": data.shape[0],
            "transform": src.window_transform(whole) @ Affine.scale(factor),
        }
        for key in ("blockxsize", "blockysize", "tiled"):
            profile.pop(key, None)

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_name(dst_path.stem + ".tmp" + dst_path.suffix)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data, 1)
    tmp.replace(dst_path)
    return {
        "source": str(src_path),
        "bounds": list(bounds),
        "resolution_m": source_res * factor,
        "source_resolution_m": source_res,
        "shape": list(data.shape),
        "nodata_pixels": holes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("dtm", type=Path)
    parser.add_argument("ortho", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    tile, radius = config["tile"], config["mars_radius_m"]
    with rasterio.open(args.dtm) as dtm:
        origin = (dtm.transform.c, dtm.transform.f)
        crs = dtm.crs.to_wkt()
    mesh_m, texture_m = tile["mesh_resolution_m"], tile["texture_resolution_m"]
    step = grid_step([mesh_m, texture_m, config["matching_map"]["lvs_map_resolution"]["value_m"]])
    bounds = tile_bounds(
        *lonlat_to_eqc(tile["center_lon"], tile["center_lat"], radius), tile["size_m"], origin, step_m=step
    )

    dtm = cut_tile(args.dtm, bounds, args.out_dir / "dtm.tif", resolution_m=mesh_m)
    if dtm["nodata_pixels"]:
        raise ValueError(f"the DTM has {dtm['nodata_pixels']} nodata pixels under {bounds}; the mesh would have holes")
    record = {
        "center_lon": tile["center_lon"],
        "center_lat": tile["center_lat"],
        "bounds": list(bounds),
        "crs": crs,
        # At the tile's centre; across a 10 km tile cos(latitude) varies by under 0.05%.
        "east_west_ground_scale": east_west_ground_scale((bounds[1] + bounds[3]) / 2, radius),
        "dtm": dtm,
        "ortho": cut_tile(args.ortho, bounds, args.out_dir / "ortho.tif", resolution_m=texture_m),
    }
    out = args.out_dir / "tile.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(out)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    print(f"tile {width:.0f} x {height:.0f} map metres at {bounds}")
    print(f"east-west ground scale {record['east_west_ground_scale']:.5f}")
    print(f"DTM {dtm['shape']} at {mesh_m} m, ortho {record['ortho']['shape']} at {texture_m} m")
    print(f"ortho nodata pixels left out of the averages: {record['ortho']['nodata_pixels']}")
    return 0
