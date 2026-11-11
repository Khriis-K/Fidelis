"""The whole-map offset between the LVS map (the CTX ortho) and the scene Fidelis renders (HiRISE), #40.

ADR 0003 measures navigation error in the map frame: a fix is judged against where the
*map* puts the ground, not where the ground truly is. If the CTX map shows a feature
`d` metres from where the HiRISE scene has it, a navigator matching HiRISE-rendered views
against CTX lands `d` off truth while doing nothing wrong. That one shift is measured
here, from the two orthoimages alone, never from descent results (which would also
absorb any navigator bias).

Method: CTX's own 6 m pixels over the HiRISE tile are compared with the tile averaged
over each pixel's footprint (CTX's grid sits a fraction of a pixel off the tile's, and
only the finer tile is resampled), and the area is cut into square windows. Iterated
phase correlation gives each window's sub-pixel shift. The whole-map offset is the
median over windows that correlate well; what's left in each window afterwards is local
distortion, which stays in the error, and a plane fit across the tile says whether it
varies systematically. Results and checks: docs/benchmarks/map-offset.md.

Offsets are CTX minus HiRISE, in map metres east and north: where CTX shows the ground
that HiRISE shows at (x, y) is (x + east_m, y + north_m).

    uv run fidelis-map-offset TILE_DIR CTX_ORTHO OUT_JSON
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from fidelis.terrain.matching_map import EPS, downsample_factor
from fidelis.terrain.raster import bilinear, centres

# Phase-correlation peak height below which a window is too weak to trust. A featureless
# window scores under 0.05 (tests); a Fidelis choice, reported with the counts it drops.
MIN_RESPONSE = 0.1
# A window this far from the median is a wrong peak, not distortion: 5 robust sd, and never
# under 30 m, above the LVS map's 18 m (99%) local-distortion requirement (Cheng et al. 2021
# Table 1), so real distortion of the allowed size is kept.
OUTLIER_SD, OUTLIER_FLOOR_M = 5.0, 30.0


def _peak(ref_spectrum: np.ndarray, mov: np.ndarray, taper: np.ndarray) -> tuple[float, float, float]:
    """One phase correlation: the surface's peak, refined by a parabola through it on each axis."""
    cross = np.conj(ref_spectrum) * np.fft.fft2(mov * taper)
    surface = np.fft.ifft2(cross / (np.abs(cross) + 1e-12)).real
    r, c = np.unravel_index(surface.argmax(), surface.shape)
    rows, cols = surface.shape

    def vertex(before, at, after):
        curve = before - 2 * at + after
        return 0.0 if curve == 0 else 0.5 * (before - after) / curve

    dy = r + vertex(surface[(r - 1) % rows, c], surface[r, c], surface[(r + 1) % rows, c])
    dx = c + vertex(surface[r, (c - 1) % cols], surface[r, c], surface[r, (c + 1) % cols])
    # Shifts past half the window wrap round to negative.
    return (dx + cols / 2) % cols - cols / 2, (dy + rows / 2) % rows - rows / 2, float(surface[r, c])


def _fourier_shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """image moved dx columns right and dy rows down, by the Fourier shift theorem."""
    fy, fx = np.fft.fftfreq(image.shape[0])[:, None], np.fft.fftfreq(image.shape[1])[None, :]
    return np.fft.ifft2(np.fft.fft2(image) * np.exp(-2j * np.pi * (fx * dx + fy * dy))).real


def shift_between(ref: np.ndarray, mov: np.ndarray, iterations: int = 20) -> tuple[float, float, float]:
    """Sub-pixel (dx, dy, response): mov shows ref's content dx columns right and dy rows down.

    Both windows are normalised (so brightness gain and offset don't matter) and Hann-tapered
    (so their edges don't). Because both share the same taper, a single phase correlation is
    pulled towards zero shift, by 10-20% on broadband test texture. So mov is shifted back by
    the estimate and correlated again until the correction vanishes. response is the final
    peak's height, 1 for a perfect match; a window with no texture gives 0.
    """
    ref, mov = np.asarray(ref, dtype=np.float64), np.asarray(mov, dtype=np.float64)
    if ref.std() == 0 or mov.std() == 0:
        return 0.0, 0.0, 0.0
    ref, mov = (ref - ref.mean()) / ref.std(), (mov - mov.mean()) / mov.std()
    taper = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    ref_spectrum = np.fft.fft2(ref * taper)
    dx = dy = response = 0.0
    for _ in range(iterations):
        step_x, step_y, response = _peak(ref_spectrum, _fourier_shift(mov, -dx, -dy), taper)
        dx, dy = dx + step_x, dy + step_y
        if abs(step_x) < 1e-4 and abs(step_y) < 1e-4:
            break
    return float(dx), float(dy), response


def _plane_fit(e: np.ndarray, n: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares a + b*e + c*n; returns (b, c) and their standard errors."""
    design = np.column_stack([np.ones_like(e), e, n])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    dof = len(values) - 3
    if dof <= 0:
        return coef[1:], np.full(2, np.nan)
    sigma2 = np.sum((values - design @ coef) ** 2) / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    return coef[1:], np.sqrt(np.diag(cov))[1:]


def summarise(table: dict, min_response: float = MIN_RESPONSE) -> dict:
    """The whole-map offset from per-window shifts, and what one rigid shift leaves behind.

    table: arrays east_km, north_km (window centres from the tile centre), de_m, dn_m, response.
    """
    e, n = np.asarray(table["east_km"], float), np.asarray(table["north_km"], float)
    de, dn = np.asarray(table["de_m"], float), np.asarray(table["dn_m"], float)
    strong = np.asarray(table["response"], float) >= min_response
    if not strong.any():
        raise ValueError(f"no window correlated above response {min_response}")

    def far(values):
        median = np.median(values[strong])
        robust_sd = 1.4826 * np.median(np.abs(values[strong] - median))
        return np.abs(values - median) > max(OUTLIER_SD * robust_sd, OUTLIER_FLOOR_M)

    used = strong & ~far(de) & ~far(dn)
    if not used.any():
        raise ValueError("no window left after removing outliers")
    east, north = float(np.median(de[used])), float(np.median(dn[used]))
    re, rn = de[used] - east, dn[used] - north
    count = int(used.sum())

    sd_east, sd_north = (float(np.std(v, ddof=1)) if count > 1 else float("nan") for v in (de[used], dn[used]))

    trend = {}
    for name, residual in (("de", re), ("dn", rn)):
        slopes, errors = _plane_fit(e[used], n[used], residual)
        for axis, slope, error in zip(("east", "north"), slopes, errors, strict=True):
            trend[f"{name}_per_{axis}"] = {"value": float(slope), "se": float(error)}

    return {
        "east_m": east,
        "north_m": north,
        "east_sd_m": sd_east,
        "north_sd_m": sd_north,
        # Valid only if windows are independent; on the real pair they aren't (map-offset.md).
        "east_se_m": sd_east / np.sqrt(count),
        "north_se_m": sd_north / np.sqrt(count),
        "residual_rms_m": float(np.sqrt(np.mean(re**2 + rn**2))),
        "residual_p99_m": float(np.percentile(np.hypot(re, rn), 99)),
        "trend_m_per_km": trend,
        "min_response": min_response,
        "windows_total": len(de),
        "windows_weak": int((~strong).sum()),
        "windows_outlier": int((strong & ~used).sum()),
        "windows_used": count,
    }


def _on_ctx_grid(tile_path: Path, ctx_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Affine]:
    """CTX's own pixels inside the tile, and the tile averaged over each one's footprint.

    CTX is never resampled: interpolating the coarse image a fraction of a pixel locks
    shifts towards whole pixels (up to 0.5 m on the real pair, injection check in
    docs/benchmarks/map-offset.md). The finer tile is sampled bilinearly at the centres of
    the tile-sized cells that tile each CTX pixel, then averaged, so any interpolation
    error is on the tile's scale. Returns tile, CTX, a mask of pixels valid in both, and
    the CTX window's transform.
    """
    with rasterio.open(tile_path) as src:
        tile, tile_t, tb, tile_nodata = src.read(1), src.transform, src.bounds, src.nodata
    with rasterio.open(ctx_path) as src:
        res = src.res[0]
        # CTX pixels lying wholly inside the tile.
        c0 = math.ceil((tb.left - src.transform.c) / res - EPS)
        c1 = math.floor((tb.right - src.transform.c) / res + EPS)
        r0 = math.ceil((src.transform.f - tb.top) / res - EPS)
        r1 = math.floor((src.transform.f - tb.bottom) / res + EPS)
        if c0 < 0 or r0 < 0 or c1 > src.width or r1 > src.height or c1 <= c0 or r1 <= r0:
            raise ValueError(f"{ctx_path} doesn't cover {tile_path}")
        ctx = src.read(1, window=((r0, r1), (c0, c1))).astype(np.float64)
        ctx_t, ctx_nodata = src.transform @ Affine.translation(c0, r0), src.nodata
    factor = downsample_factor(tile_t.a, res)
    mx, my = centres(ctx_t, ctx.shape)
    valid_tile = np.ones(tile.shape) if tile_nodata is None else (tile != tile_nodata).astype(np.float64)
    averaged, coverage = np.zeros(ctx.shape), np.zeros(ctx.shape)
    for k in range(factor):
        for j in range(factor):
            x = mx + ((k + 0.5) / factor - 0.5) * res
            y = my - ((j + 0.5) / factor - 0.5) * res
            averaged += bilinear(tile, tile_t, x, y) / factor**2
            coverage += bilinear(valid_tile, tile_t, x, y) / factor**2
    valid = coverage > 1 - EPS
    if ctx_nodata is not None:
        valid &= ctx != ctx_nodata
    return averaged, ctx, valid, ctx_t


def measure_offset(tile_path: Path, ctx_path: Path, window_px: int) -> dict:
    """Register CTX against the HiRISE tile in non-overlapping windows; see the module docstring."""
    tile, ctx, valid, transform = _on_ctx_grid(tile_path, ctx_path)
    res = transform.a
    rows, cols = tile.shape
    # Whole windows, centred in the area, so the trend fit's origin is the area's centre.
    row_start, col_start = rows % window_px // 2, cols % window_px // 2
    n_rows, n_cols = rows // window_px, cols // window_px
    left, top = transform.c + col_start * res, transform.f - row_start * res
    right, bottom = left + n_cols * window_px * res, top - n_rows * window_px * res
    centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
    windows = []
    for r0 in range(row_start, row_start + n_rows * window_px, window_px):
        for c0 in range(col_start, col_start + n_cols * window_px, window_px):
            cut = np.s_[r0 : r0 + window_px, c0 : c0 + window_px]
            if not valid[cut].all():
                continue
            dx, dy, response = shift_between(tile[cut], ctx[cut])
            x = transform.c + (c0 + window_px / 2) * res
            y = transform.f - (r0 + window_px / 2) * res
            windows.append(
                {
                    "east_km": (x - centre_x) / 1000,
                    "north_km": (y - centre_y) / 1000,
                    "de_m": dx * res,
                    "dn_m": -dy * res,  # rows grow southward
                    "response": response,
                }
            )
    if not windows:
        raise ValueError(f"no whole {window_px}-pixel window of valid data in {ctx_path} over {tile_path}")
    table = {key: np.array([w[key] for w in windows]) for key in ("east_km", "north_km", "de_m", "dn_m", "response")}
    return summarise(table) | {
        "bounds": [left, bottom, right, top],  # the windows' extent
        "resolution_m": res,
        "window_m": window_px * res,
        "windows": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tile_dir", type=Path, help="ortho.tif and tile.json of the HiRISE tile")
    parser.add_argument("ctx_ortho", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--window-px", type=int, default=128)
    args = parser.parse_args()

    result = measure_offset(args.tile_dir / "ortho.tif", args.ctx_ortho, args.window_px)
    scale = json.loads((args.tile_dir / "tile.json").read_text())["east_west_ground_scale"]
    result["east_west_ground_scale"] = scale
    result["ground_m"] = {"east": result["east_m"] * scale, "north": result["north_m"]}
    result["sources"] = {"tile": str(args.tile_dir / "ortho.tif"), "ctx_ortho": str(args.ctx_ortho)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(args.out)
    print(
        f"CTX - HiRISE: {result['east_m']:+.2f} m east, {result['north_m']:+.2f} m north (map metres), "
        f"window sd {result['east_sd_m']:.2f} / {result['north_sd_m']:.2f} m, {result['windows_used']} windows; "
        f"residual rms {result['residual_rms_m']:.2f} m"
    )
    return 0
