"""The orbital map the navigator matches against: an orthoimage coarsened to CTX-like pixels.

The real Mars 2020 Lander Vision System matched descent images against a map built
from MRO CTX imagery at about 6 m/pixel, much coarser than HiRISE's 0.25 m. Matching
against a sharp HiRISE map would flatter the navigator, so the map is the HiRISE
orthoimage averaged down to the LVS map's resolution. Training-region maps (#16) go
through the same function, so test and training maps get identical treatment.

Averaging whole blocks is a simple stand-in for CTX's optics: it removes detail finer
than a CTX pixel but doesn't model CTX's point-spread function or noise.

Map pixels are square in the mosaics' map metres, so on the ground they're
cos(latitude) narrower east-west; map.json records both ground sizes.

For the wide scene (#33) the map covers the CTX ring too: the inner texture averaged as
above, and the ring's (brightness-matched) CTX texture around it, bilinear-resampled onto
the inner map's pixel grid because CTX's own grid is a fraction of a pixel off it. So the
ring part of the map is CTX itself, slightly smoothed, not CTX-like.

    uv run fidelis-matching-map CONFIG SCENE_DIR OUT_TIF
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import Affine, from_origin

from fidelis.terrain.raster import bilinear, centres

# Snap tolerance: float noise in metres, far below any pixel size.
EPS = 1e-6


def require_same_resolution(lvs_m: float, ctx_m: float) -> None:
    """The LVS map was made at CTX's acquisition resolution; the two facts must agree."""
    if abs(lvs_m - ctx_m) > 1e-6:
        raise ValueError(f"LVS map resolution {lvs_m} m disagrees with CTX ground resolution {ctx_m} m")


def downsample_factor(source_res_m: float, target_res_m: float) -> int:
    """How many source pixels fit across one target pixel; must be a whole number."""
    ratio = target_res_m / source_res_m
    factor = round(ratio)
    if factor < 1 or abs(ratio - factor) > 1e-6:
        raise ValueError(f"{target_res_m} m is not a whole multiple of {source_res_m} m pixels")
    return factor


def block_mean(array: np.ndarray, factor: int, nodata: float | None = None) -> np.ndarray:
    """Average each factor x factor block, leaving nodata pixels out of the average.

    A block with no valid pixels comes out as nodata. Integer images are rounded back
    to their own dtype.
    """
    rows, cols = array.shape
    if rows % factor or cols % factor:
        raise ValueError(f"shape {array.shape} doesn't divide into {factor} x {factor} blocks")
    blocks = array.reshape(rows // factor, factor, cols // factor, factor)
    # Compare in the image's own dtype: a float32 nodata like -3.4e38 isn't equal to itself in float64.
    valid = np.ones(blocks.shape, dtype=bool) if nodata is None else blocks != nodata
    blocks = blocks.astype(np.float64)
    count = valid.sum(axis=(1, 3))
    total = np.where(valid, blocks, 0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    if nodata is not None:
        mean[count == 0] = nodata
    if np.issubdtype(array.dtype, np.integer):
        mean = np.rint(mean)
    return mean.astype(array.dtype)


def build_matching_map(ortho_path: Path, target_res_m: float, dst_path: Path) -> dict:
    """Write the coarsened map with the same footprint and CRS; return what was built."""
    with rasterio.open(ortho_path) as src:
        res_x, res_y = src.res
        if res_x != res_y:
            raise ValueError(f"{ortho_path} has non-square pixels {src.res}")
        factor = downsample_factor(res_x, target_res_m)
        image = block_mean(src.read(1), factor, nodata=src.nodata)
        profile = src.profile | {
            "width": image.shape[1],
            "height": image.shape[0],
            "transform": src.transform @ Affine.scale(factor),
        }
        # Drop any internal tiling block sizes from the source; they may not fit the small map.
        for key in ("blockxsize", "blockysize", "tiled"):
            profile.pop(key, None)

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_name(dst_path.stem + ".tmp" + dst_path.suffix)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(image, 1)
    tmp.replace(dst_path)
    return {
        "source": str(ortho_path),
        "source_resolution_m": res_x,
        "factor": factor,
        "resolution_m": res_x * factor,
        "shape": list(image.shape),
    }


def build_scene_map(inner_ortho: Path, ring_ortho: Path, target_res_m: float, dst_path: Path) -> dict:
    """One map over the ring: on the inner map's grid, shrunk to whole pixels inside the ring's data.

    Inside the inner tile, its texture block-averaged as `build_matching_map` does; outside,
    the ring's texture sampled bilinearly at each map pixel's centre.
    """
    with rasterio.open(inner_ortho) as src:
        factor = downsample_factor(src.res[0], target_res_m)
        inner = block_mean(src.read(1), factor, nodata=src.nodata)
        inner_t, profile, source_res = src.transform @ Affine.scale(factor), src.profile, src.res[0]
    with rasterio.open(ring_ortho) as src:
        ring, ring_t, b = src.read(1), src.transform, src.bounds

    res, (ox, oy) = target_res_m, (inner_t.c, inner_t.f)
    left = ox + math.ceil((b.left - ox) / res - EPS) * res
    right = ox + math.floor((b.right - ox) / res + EPS) * res
    top = oy + math.floor((b.top - oy) / res + EPS) * res
    bottom = oy + math.ceil((b.bottom - oy) / res - EPS) * res
    shape = (round((top - bottom) / res), round((right - left) / res))
    transform = from_origin(left, top, res, res)
    image = np.clip(np.rint(bilinear(ring, ring_t, *centres(transform, shape))), 0, 255).astype(np.uint8)
    col0, row0 = round((ox - left) / res), round((top - oy) / res)
    if col0 < 0 or row0 < 0 or col0 + inner.shape[1] > shape[1] or row0 + inner.shape[0] > shape[0]:
        raise ValueError(f"{inner_ortho} reaches past the ring's data in {ring_ortho}")
    image[row0 : row0 + inner.shape[0], col0 : col0 + inner.shape[1]] = inner

    profile = profile | {"width": shape[1], "height": shape[0], "transform": transform}
    for key in ("blockxsize", "blockysize", "tiled"):
        profile.pop(key, None)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_name(dst_path.stem + ".tmp" + dst_path.suffix)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(image, 1)
    tmp.replace(dst_path)
    return {
        "source": str(inner_ortho),
        "ring_source": str(ring_ortho),
        "source_resolution_m": source_res,
        "factor": factor,
        "resolution_m": res,
        "shape": list(shape),
        "bounds": [left, bottom, right, top],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("tile_dir", type=Path, help="ortho.tif and tile.json; also ring_ortho.tif if a ring")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    resolutions = yaml.safe_load(args.config.read_text())["matching_map"]
    lvs_m = resolutions["lvs_map_resolution"]["value_m"]
    require_same_resolution(lvs_m, resolutions["ctx_ground_resolution"]["value_m"])
    if (args.tile_dir / "ring.json").exists():
        info = build_scene_map(args.tile_dir / "ortho.tif", args.tile_dir / "ring_ortho.tif", lvs_m, args.out)
    else:
        info = build_matching_map(args.tile_dir / "ortho.tif", lvs_m, args.out)
    # The factor's provenance travels with the map.
    info["lvs_map_resolution"] = resolutions["lvs_map_resolution"]
    info["ctx_ground_resolution"] = resolutions["ctx_ground_resolution"]
    # Pixels are square in map metres, not on the ground: the projection stretches
    # east-west by 1/cos(latitude), so a matcher working in ground metres needs this.
    scale = json.loads((args.tile_dir / "tile.json").read_text())["east_west_ground_scale"]
    info["east_west_ground_scale"] = scale
    info["ground_resolution_m"] = {"east_west": info["resolution_m"] * scale, "north_south": info["resolution_m"]}
    record = args.out.with_suffix(".json")
    tmp = record.with_name(record.name + ".tmp")
    tmp.write_text(json.dumps(info, indent=2) + "\n")
    tmp.replace(record)
    print(f"{args.out}: {info['shape']} at {info['resolution_m']} m/pixel ({info['factor']}x)")
    return 0
