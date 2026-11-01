"""Cut one working tile (DTM + matching orthoimage) out of the Jezero mosaics.

The USGS mosaics use an equirectangular projection on a sphere (Eqc, latTs0, lon0),
so lon/lat turn into map metres as arc lengths. Tile edges are snapped to the matching
map's 6 m grid, measured from the DTM's origin: 6 m is a whole number of DTM (1 m) and
orthoimage (0.25 m) pixels, so both tiles and the downsampled map share exact edges.

    uv run fidelis-tile CONFIG DTM ORTHO OUT_DIR
"""

import argparse
import json
import math
from pathlib import Path

import rasterio
import yaml
from rasterio.windows import Window, from_bounds

Bounds = tuple[float, float, float, float]  # left, bottom, right, top (map metres)

# Snap tolerance: float noise in metres, far below any pixel size.
EPS = 1e-6


def lonlat_to_eqc(lon_deg: float, lat_deg: float, radius_m: float) -> tuple[float, float]:
    """Map metres in an equirectangular projection with lat_ts=0, lon_0=0."""
    return radius_m * math.radians(lon_deg), radius_m * math.radians(lat_deg)


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


def cut_tile(src_path: Path, bounds: Bounds, dst_path: Path) -> dict:
    """Copy the window under bounds into its own GeoTIFF; return its bounds, resolution and shape."""
    with rasterio.open(src_path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        whole = window.round_offsets().round_lengths()
        if any(abs(a - b) > EPS for a, b in zip(window.flatten(), whole.flatten(), strict=True)):
            raise ValueError(f"tile bounds {bounds} are off {src_path}'s pixel grid")
        if not whole.intersection(Window(0, 0, src.width, src.height)) == whole:
            raise ValueError(f"tile bounds {bounds} fall outside {src_path} ({tuple(src.bounds)})")
        data = src.read(1, window=whole)
        profile = src.profile | {
            "width": whole.width,
            "height": whole.height,
            "transform": src.window_transform(whole),
        }
        for key in ("blockxsize", "blockysize", "tiled"):
            profile.pop(key, None)
        resolution = src.res[0]

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_name(dst_path.stem + ".tmp" + dst_path.suffix)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data, 1)
    tmp.replace(dst_path)
    return {
        "source": str(src_path),
        "bounds": list(bounds),
        "resolution_m": resolution,
        "shape": list(data.shape),
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
    bounds = tile_bounds(
        *lonlat_to_eqc(tile["center_lon"], tile["center_lat"], radius),
        tile["size_m"],
        origin,
        step_m=config["matching_map"]["lvs_map_resolution"]["value_m"],
    )

    record = {
        "center_lon": tile["center_lon"],
        "center_lat": tile["center_lat"],
        "bounds": list(bounds),
        "crs": crs,
        "dtm": cut_tile(args.dtm, bounds, args.out_dir / "dtm.tif"),
        "ortho": cut_tile(args.ortho, bounds, args.out_dir / "ortho.tif"),
    }
    out = args.out_dir / "tile.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(out)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    print(f"tile {width:.0f} x {height:.0f} m at {bounds}")
    print(f"DTM {record['dtm']['shape']}, ortho {record['ortho']['shape']}")
    return 0
