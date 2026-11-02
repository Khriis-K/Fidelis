"""The wide Jezero scene: the HiRISE tile in the middle, a CTX ring around it, blended at the seam.

On the 10 km HiRISE tile alone, the scenario sampler had to clip camera tilt to 3 deg and
position offset to +-400 m, or tilted views would see past the terrain (issue #33). The
USGS Mars 2020 TRN CTX mosaics (20 m DTM, 6 m orthoimage) cover a ~27.7 km square around
the landing site on the same projection, so a CTX ring fills out a scene about 26 km
across while the camera's own neighbourhood stays HiRISE.

The two sources disagree a little, so the HiRISE tile is eased into CTX across a band
inside its edge (`blend_band_m`): heights and texture both go linearly from CTX's at the
mesh's outermost vertices to HiRISE's own a band inside. Before that, the CTX texture is
brightness-matched to HiRISE (same mean and spread over the tile), since the two were
taken at different times under different suns. Otherwise the seam would be a feature
the matcher could latch onto.

The CTX mosaics are on their own pixel grid, a fraction of a pixel off HiRISE's (CTX's
origin sits 0.749 m east and 3.140 m north of a 6 m HiRISE pixel corner), so CTX values
under HiRISE pixels are bilinear samples.

Mars falls away from a flat tile by d^2 / 2R (3.7 m at 5 km, about 25 m at 13 km), so
the scene's local frame drops each point by that much (`to_local`); tile.json carries the
radius for the mesh.

    uv run fidelis-scene CONFIG TILE_DIR CTX_DTM CTX_ORTHO OUT_DIR

writes the blended tile (dtm.tif, ortho.tif, tile.json) and the ring (ring_dtm.tif,
ring_ortho.tif, ring.json) into OUT_DIR.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import Affine

from fidelis.terrain.matching_map import block_mean, downsample_factor
from fidelis.terrain.raster import bilinear, centres, read_raster, write_raster
from fidelis.terrain.tile import Bounds, cut_tile, grid_step, tile_bounds

# After blending, the tile's outermost heights are CTX's own, stored as float32: only
# rounding separates them (float32 spacing near -2500 m is 0.24 mm).
HEIGHT_STEP_TOLERANCE_M = 1e-3


def curvature_drop(east, north, radius_m: float):
    """How far a sphere of radius_m falls below the plane touching it at (0, 0): d^2 / 2R.

    Within a millimetre of the exact R - sqrt(R^2 - d^2) out to 13 km on Mars.
    """
    return (np.square(east) + np.square(north)) / (2 * radius_m)


def to_local(mx, my, h, origin: tuple[float, float], east_scale: float, radius_m: float | None) -> np.ndarray:
    """Float32 (..., 3) local ground metres: x east, y north of origin, z elevation less curvature.

    With no radius the frame is flat, as the 10 km tile's was before #33.
    """
    x = (mx - origin[0]) * east_scale
    y = my - origin[1]
    z = h if radius_m is None else h - curvature_drop(x, y, radius_m)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def band_weight(mx, my, edge: Bounds, band_m: float) -> np.ndarray:
    """0 on (and outside) the edge rectangle, rising linearly to 1 a band inside its nearest side."""
    left, bottom, right, top = edge
    inset = np.minimum.reduce([mx - left, right - mx, my - bottom, top - my])
    return np.clip(inset / band_m, 0.0, 1.0)


def brightness_match(reference: np.ndarray, source: np.ndarray) -> tuple[float, float]:
    """Gain and offset that give source the reference's mean and standard deviation."""
    gain = float(np.std(reference) / np.std(source))
    return gain, float(np.mean(reference) - gain * np.mean(source))


def ring_bounds(
    centre: tuple[float, float], size_m: float, ring_res_m: float, ortho_res_m: float, origin: tuple[float, float]
) -> Bounds:
    """A CTX window whose pixel centres reach size_m / 2 from centre every way.

    The window is whole DTM and orthoimage pixels (they share an origin), grown outward;
    one extra DTM pixel makes up for the mesh's outer vertices sitting half a pixel in.
    """
    step = grid_step([ring_res_m, ortho_res_m])
    return tile_bounds(*centre, size_m + ring_res_m, origin, step_m=step)


def write_json(path: Path, record: dict) -> None:
    """Indented JSON, written to a temp file and renamed into place."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(path)


def blend_scene(
    tile_dir: Path, ctx_dtm: Path, ctx_ortho: Path, out_dir: Path, size_m: float, band_m: float, radius_m: float
) -> dict:
    """Cut the CTX ring, brightness-match it, and blend the HiRISE tile into it. Return what was built."""
    tile_dir, out_dir = Path(tile_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tile = json.loads((tile_dir / "tile.json").read_text())
    heights, dtm_t, dtm_profile = read_raster(tile_dir / "dtm.tif")
    texture, tex_t, tex_profile = read_raster(tile_dir / "ortho.tif")
    rows, cols = heights.shape
    left, top = dtm_t.c, dtm_t.f
    right, bottom = left + cols * dtm_t.a, top + rows * dtm_t.e
    # The mesh's outermost vertices are the DTM's outermost pixel centres.
    half = dtm_t.a / 2
    mesh_edge = (left + half, bottom + half, right - half, top - half)

    with rasterio.open(ctx_dtm) as src:
        origin, ring_res = (src.transform.c, src.transform.f), src.res[0]
    with rasterio.open(ctx_ortho) as src:
        ortho_res = src.res[0]
    bounds = ring_bounds(((left + right) / 2, (bottom + top) / 2), size_m, ring_res, ortho_res, origin)
    ring_dtm = cut_tile(ctx_dtm, bounds, out_dir / "ring_dtm.tif")
    if ring_dtm["nodata_pixels"]:
        raise ValueError(f"the CTX DTM has {ring_dtm['nodata_pixels']} nodata pixels under {bounds}")
    ring_ortho = cut_tile(ctx_ortho, bounds, out_dir / "ring_ortho.tif")
    ring_heights, ring_t, _ = read_raster(out_dir / "ring_dtm.tif")
    ring_image, ring_image_t, ring_profile = read_raster(out_dir / "ring_ortho.tif")

    # Heights: CTX at the edge, HiRISE a band inside.
    mx, my = centres(dtm_t, heights.shape)
    w = band_weight(mx, my, mesh_edge, band_m)
    ctx_heights = bilinear(ring_heights, ring_t, mx, my)
    offset = ctx_heights - heights
    blended = (w * heights + (1 - w) * ctx_heights).astype(np.float32)

    # Brightness: compared at CTX's pixel size, so HiRISE's finer detail doesn't count as spread.
    factor = downsample_factor(tex_t.a, ortho_res)
    nodata = tex_profile.get("nodata")
    reference = block_mean(texture, factor, nodata=nodata)
    ref_t = tex_t @ Affine.scale(factor)
    source = bilinear(ring_image, ring_image_t, *centres(ref_t, reference.shape))
    valid = reference != nodata if nodata is not None else np.ones(reference.shape, bool)
    gain, shift = brightness_match(reference[valid].astype(np.float64), source[valid])
    # 0 is the orthoimages' nodata: matched values stay at 1 or above.
    matched = np.where(ring_image == 0, 0, np.clip(np.rint(gain * ring_image + shift), 1, 255)).astype(np.uint8)

    # Texture: the same easing, onto the matched CTX.
    tx, ty = centres(tex_t, texture.shape)
    w_tex = band_weight(tx, ty, mesh_edge, band_m)
    ctx_texture = gain * bilinear(ring_image, ring_image_t, tx, ty) + shift
    hirise = np.where(texture == 0, ctx_texture, texture) if nodata == 0 else texture
    eased = np.clip(np.rint(w_tex * hirise + (1 - w_tex) * ctx_texture), 1, 255).astype(np.uint8)

    write_raster(out_dir / "dtm.tif", blended, dtm_profile)
    write_raster(out_dir / "ortho.tif", eased, tex_profile)
    write_raster(out_dir / "ring_ortho.tif", matched, ring_profile)

    # For the sampler's check that views stay on the terrain: with the camera anywhere over
    # the tile, the ground at a view's edge is at most this much lower than beneath it. As
    # the mesh has it, curvature included. Higher ground (the crater rim) only shortens views.
    centre = ((left + right) / 2, (bottom + top) / 2)
    scale = tile["east_west_ground_scale"]
    rx, ry = centres(ring_t, ring_heights.shape)
    inner_z = to_local(mx, my, blended, centre, scale, radius_m)[..., 2]
    lowest = min(inner_z.min(), to_local(rx, ry, ring_heights, centre, scale, radius_m)[..., 2].min())
    ring = {
        "ctx_dtm": str(ctx_dtm),
        "ctx_ortho": str(ctx_ortho),
        "bounds": list(bounds),
        "dtm": ring_dtm,
        "ortho": ring_ortho,
        "blend_band_m": band_m,
        # CTX minus HiRISE over the tile, before blending.
        "height_offset_m": {"median": float(np.median(offset)), "std": float(np.std(offset))},
        "brightness_gain": gain,
        "brightness_offset": shift,
        "drop_below_tile_m": float(inner_z.max() - lowest),
    }
    tile = tile | {"curvature_radius_m": radius_m, "blended_from": str(tile_dir)}
    write_json(out_dir / "tile.json", tile)
    write_json(out_dir / "ring.json", ring)
    return {"tile": tile, "ring": ring}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("tile_dir", type=Path, help="holds dtm.tif, ortho.tif and tile.json from fidelis-tile")
    parser.add_argument("ctx_dtm", type=Path)
    parser.add_argument("ctx_ortho", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    scene = config["scene"]
    record = blend_scene(
        args.tile_dir,
        args.ctx_dtm,
        args.ctx_ortho,
        args.out_dir,
        scene["size_m"],
        scene["blend_band_m"],
        config["mars_radius_m"],
    )
    ring = record["ring"]
    left, bottom, right, top = ring["bounds"]
    print(f"ring {right - left:.0f} x {top - bottom:.0f} map metres at {ring['bounds']}")
    print(f"CTX - HiRISE heights over the tile: median {ring['height_offset_m']['median']:+.2f} m,")
    print(f"  sd {ring['height_offset_m']['std']:.2f} m; blended across {ring['blend_band_m']} m")
    print(
        f"CTX brightness matched to HiRISE: gain {ring['brightness_gain']:.3f}, offset {ring['brightness_offset']:+.1f}"
    )
    print(f"tile's highest ground to the scene's lowest, curvature included: {ring['drop_below_tile_m']:.1f} m")
    return 0
