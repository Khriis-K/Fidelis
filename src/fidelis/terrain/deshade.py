"""Shadow reduction (#19): divide the terrain's baked shading out of the orthoimage.

The orthoimage is a photo, so the sun it was taken under is painted into it: slopes
facing that sun are bright, slopes facing away are dark. The mesh renderer then shades
the terrain again under a simulated sun, on top of the old shading, and the old shading
never moves. This step estimates the baked shading and divides it out, leaving a texture
closer to the ground's own brightness (its albedo).

How. On a gently sloping diffuse surface, brightness is close to

    albedo * (1 + p zx + q zy)

with zx, zy the slopes east and north, and p, q fixed by where the sun was. p and q are
fitted from the photo itself, by least squares of its local brightness against the DTM's
slopes, separately in each block of the tile. The Jezero orthoimage is a patchwork of
HiRISE images taken under different suns, with unpublished seams
(docs/research/ctx-and-hirise-sun-angles.md), so one sourced sun for the whole tile would
be wrong on part of it; the fit also takes whatever strength the shading really has. Both
the photo and the slopes are high-passed before the fit, so broad albedo patterns don't
count as shading. Ground whose brightness has nothing to do with its slopes fits p = q = 0
and comes back unchanged.

Not removed: cast shadows, shading finer than the DTM's pixels, the part of steep slopes'
shading that isn't linear in the slope, and anything on the CTX ring. See docs/benchmarks/deshade.md.

    uv run fidelis-deshade CONFIG SCENE_DIR OUT_PNG
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from fidelis.terrain.raster import read_raster
from fidelis.terrain.scene import write_json

# Slopes steeper than this face the sun or face away from it (slope_brightness_ratio);
# slopes gentler than FLAT_SLOPE_DEG are flat ground (flat_detail_kept).
FACING_SLOPE_DEG = 5.0
FLAT_SLOPE_DEG = 2.0


@dataclass(frozen=True)
class ShadingFit:
    """The fitted shading, one value per block (row 0 is the northmost): brightness goes as
    1 + p zx + q zy, and r2 is the share of the photo's local brightness the fit explains
    (about zero, not about a block mean: both sides of the fit are high-passed)."""

    p: np.ndarray
    q: np.ndarray
    r2: np.ndarray


def slopes(heights: np.ndarray, dx_m: float, dy_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Rise per ground metre east and north, for a north-up height grid."""
    heights = np.asarray(heights, dtype=np.float64)
    return np.gradient(heights, axis=1) / dx_m, -np.gradient(heights, axis=0) / dy_m


def fitted_sun(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The sun a fit implies: compass bearing and elevation, degrees.

    The elevation is only as good as the surface is diffuse: shading stronger than a
    diffuse surface's reads as a lower sun.
    """
    bearing = np.degrees(np.arctan2(-p, -q)) % 360
    return bearing, np.degrees(np.arctan2(1.0, np.hypot(p, q)))


def _highpass(a: np.ndarray, sigma_px: float) -> np.ndarray:
    return a - cv2.GaussianBlur(a, (0, 0), sigma_px)


def _against_local_mean(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Brightness against the local average, less one."""
    return image / np.maximum(cv2.GaussianBlur(image, (0, 0), sigma_px), 1e-9) - 1


def _block_mean(image: np.ndarray, k: int) -> np.ndarray:
    rows, cols = image.shape[0] // k, image.shape[1] // k
    return image.reshape(rows, k, cols, k).mean(axis=(1, 3))


def _pixels_per_dtm_pixel(image_shape: tuple[int, int], dtm_shape: tuple[int, int]) -> int:
    k = image_shape[0] // dtm_shape[0]
    if k < 1 or image_shape != (dtm_shape[0] * k, dtm_shape[1] * k):
        raise ValueError(f"the image {image_shape} must be a whole number of pixels per DTM pixel {dtm_shape}")
    return k


def deshade(
    image: np.ndarray,
    heights: np.ndarray,
    dx_m: float,
    dy_m: float,
    block_m: float,
    highpass_m: float,
    shade_floor: float,
) -> tuple[np.ndarray, ShadingFit]:
    """The image with its fitted terrain shading divided out, and the fit.

    `image` covers the same ground as `heights`, at a whole number of pixels per height.
    dx_m, dy_m are a height pixel's ground size east and north. The shading is fitted in
    blocks of about block_m, on structure finer than about highpass_m. Where the fitted
    shade falls below shade_floor (a steep slope facing away), the floor is used, so
    nothing is brightened by more than 1 / shade_floor.
    """
    image = np.asarray(image, dtype=np.float64)
    k = _pixels_per_dtm_pixel(image.shape, heights.shape)
    zx, zy = slopes(heights, dx_m, dy_m)
    sigma = highpass_m / dy_m

    # Local brightness against local slope: both high-passed, as the module docstring says.
    target = _against_local_mean(_block_mean(image, k), sigma)
    hx, hy = _highpass(zx, sigma), _highpass(zy, sigma)

    rows, cols = heights.shape
    row_edges = np.linspace(0, rows, max(1, round(rows * dy_m / block_m)) + 1).astype(int)
    col_edges = np.linspace(0, cols, max(1, round(cols * dx_m / block_m)) + 1).astype(int)
    shape = (len(row_edges) - 1, len(col_edges) - 1)
    p, q, r2 = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            block = (slice(row_edges[i], row_edges[i + 1]), slice(col_edges[j], col_edges[j + 1]))
            predictors = np.stack([hx[block].ravel(), hy[block].ravel()], axis=1)
            y = target[block].ravel()
            (p[i, j], q[i, j]), *_ = np.linalg.lstsq(predictors, y, rcond=None)
            total = float(y @ y)
            if total > 0:
                residual = y - predictors @ np.array([p[i, j], q[i, j]])
                r2[i, j] = 1 - float(residual @ residual) / total

    # Each block's values hold at its centre and are blended between centres.
    p_field = cv2.resize(p, (cols, rows), interpolation=cv2.INTER_LINEAR)
    q_field = cv2.resize(q, (cols, rows), interpolation=cv2.INTER_LINEAR)
    shade = 1 + p_field * zx + q_field * zy
    if k > 1:
        shade = cv2.resize(shade, (cols * k, rows * k), interpolation=cv2.INTER_LINEAR)
    return image / np.maximum(shade, shade_floor), ShadingFit(p, q, r2)


def slope_brightness_ratio(
    image: np.ndarray, zx: np.ndarray, zy: np.ndarray, sun_bearing_deg: float, highpass_px: float
) -> float:
    """How much brighter slopes facing the sun are than slopes facing away: 1 means no shading.

    `image` is on the slopes' grid. Brightness is taken against the local average, so
    broad albedo patterns don't count. The bearing must come from somewhere other than
    the fit being judged.
    """
    az = math.radians(sun_bearing_deg)
    toward = -zx * math.sin(az) - zy * math.cos(az)  # the slope's rise away from the sun
    steep = math.tan(math.radians(FACING_SLOPE_DEG))
    relative = _against_local_mean(np.asarray(image, dtype=np.float64), highpass_px)
    facing, away = relative[toward > steep], relative[toward < -steep]
    if facing.size == 0 or away.size == 0:
        return float("nan")
    return float((1 + facing.mean()) / (1 + away.mean()))


def flat_detail_kept(before: np.ndarray, after: np.ndarray, zx: np.ndarray, zy: np.ndarray, detail_px: float) -> float:
    """The share of the image's fine contrast on flat ground that survived: 1 means all of it.

    Flat ground has no terrain shading to remove, so its detail should be untouched. A
    step that only blurs the image also lowers shading contrast; this is what catches it.
    The images are a whole number of pixels per slope pixel; detail_px is in their pixels.
    """
    k = _pixels_per_dtm_pixel(before.shape, zx.shape)
    flat = np.hypot(zx, zy) < math.tan(math.radians(FLAT_SLOPE_DEG))
    flat = np.repeat(np.repeat(flat, k, axis=0), k, axis=1)
    if not flat.any():
        return float("nan")

    def detail(image: np.ndarray) -> float:
        log = np.log(np.maximum(np.asarray(image, dtype=np.float64), 1e-9))
        return float(_highpass(log, detail_px)[flat].std())

    return detail(after) / detail(before)


def _block_values(values: np.ndarray) -> list:
    """A block grid as nested lists for JSON, with None where there is no value."""
    return [[None if not math.isfinite(v) else round(float(v), 3) for v in row] for row in values]


def build_deshaded_texture(dtm_path: Path, ortho_path: Path, out_png: Path, east_scale: float, params: dict) -> dict:
    """Write the de-shaded orthoimage as a texture on the orthoimage's own pixels, and its record beside it.

    `params` is the scene config's `deshade` section. Its `check_sun` is a sourced sun
    bearing that holds within half_width_m (map metres) east-west of the tile's centre;
    the record reports the slope brightness ratio there, before and after.
    """
    heights, transform, profile = read_raster(dtm_path)
    image, _, _ = read_raster(ortho_path)
    holes = ~np.isfinite(heights)
    if profile.get("nodata") is not None:
        holes |= heights == np.float32(profile["nodata"])
    if holes.any():
        raise ValueError(f"{dtm_path} has {holes.sum()} nodata pixels; their slopes would wreck the fit")
    k = _pixels_per_dtm_pixel(image.shape, heights.shape)
    dx_m, dy_m = transform.a * east_scale, -transform.e
    albedo, fit = deshade(image, heights, dx_m, dy_m, params["block_m"], params["highpass_m"], params["shade_floor"])

    full_scale = np.iinfo(image.dtype).max
    texture = np.clip(np.round(albedo), 0, full_scale).astype(image.dtype)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_png.with_name(out_png.stem + ".tmp" + out_png.suffix)
    if not cv2.imwrite(str(tmp), texture):
        raise OSError(f"could not write {tmp}")
    tmp.replace(out_png)

    zx, zy = slopes(heights, dx_m, dy_m)
    check_sun = params["check_sun"]
    half = round(check_sun["half_width_m"] / transform.a)
    centre = heights.shape[1] // 2
    region = np.s_[:, max(0, centre - half) : centre + half]
    highpass_px = params["highpass_m"] / dy_m

    def ratio(picture: np.ndarray) -> float:
        coarse = _block_mean(np.asarray(picture, dtype=np.float64), k)
        return slope_brightness_ratio(coarse[region], zx[region], zy[region], check_sun["azimuth_deg"], highpass_px)

    bearing, elevation = fitted_sun(fit.p, fit.q)
    record = {
        "dtm": str(dtm_path),
        "ortho": str(ortho_path),
        "shape": list(texture.shape),
        "dtype": str(texture.dtype),
        "dtm_pixel_ground_m": {"east": dx_m, "north": dy_m},
        **{key: params[key] for key in ("block_m", "highpass_m", "shade_floor", "detail_m")},
        "slope_brightness_ratio": {"before": ratio(image), "after": ratio(texture), "sun": check_sun},
        "flat_detail_kept": flat_detail_kept(image, texture, zx, zy, params["detail_m"] / (dy_m / k)),
        "clipped_fraction": float(np.mean(np.round(albedo) > full_scale)),
        "fitted_sun": {
            "note": "one value per block, northmost row first; see fitted_sun on what the elevation means",
            "bearing_deg": _block_values(bearing),
            "elevation_deg": _block_values(elevation),
            "r2": _block_values(fit.r2),
        },
    }
    write_json(out_png.with_suffix(".json"), record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("scene_dir", type=Path, help="holds dtm.tif, ortho.tif and tile.json")
    parser.add_argument("out", type=Path, help="the texture to write, beside the mesh's texture.png")
    args = parser.parse_args()

    params = yaml.safe_load(args.config.read_text(encoding="utf-8"))["deshade"]
    scale = json.loads((args.scene_dir / "tile.json").read_text())["east_west_ground_scale"]
    record = build_deshaded_texture(args.scene_dir / "dtm.tif", args.scene_dir / "ortho.tif", args.out, scale, params)
    ratio = record["slope_brightness_ratio"]
    print(
        f"{args.out}: sun-facing against sun-away slopes {ratio['before']:.3f} -> {ratio['after']:.3f},"
        f" flat-ground detail kept {record['flat_detail_kept']:.3f}"
    )
    return 0
