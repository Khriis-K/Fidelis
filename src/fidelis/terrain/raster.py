"""North-up rasters: reading and writing them, where their pixels are, and values between them."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

# Internal tiling a source may carry; a new raster of a different shape may not fit it.
_TILING_KEYS = ("blockxsize", "blockysize", "tiled")


def read_raster(path: Path) -> tuple[np.ndarray, Affine, dict]:
    """Band 1, its geotransform, and the profile to write a raster like it."""
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.profile


def write_raster(path: Path, data: np.ndarray, profile: dict) -> None:
    """Write band 1 to a temp file, then rename it into place; the profile's shape follows data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {k: v for k, v in profile.items() if k not in _TILING_KEYS}
    profile |= {"width": data.shape[1], "height": data.shape[0]}
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data, 1)
    tmp.replace(path)


def centres(transform: Affine, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Map x and y of every pixel centre, each an array of the raster's shape."""
    rows, cols = shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    return transform.c + (c + 0.5) * transform.a, transform.f + (r + 0.5) * transform.e


def bilinear(z: np.ndarray, transform: Affine, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    """Bilinear in the pixel centres; past the outermost centres it extrapolates the edge cell."""
    col, row = ~transform @ (np.asarray(mx, dtype=np.float64), np.asarray(my, dtype=np.float64))
    col, row = col - 0.5, row - 0.5
    rows, cols = z.shape
    c0 = np.clip(np.floor(col), 0, cols - 2).astype(int)
    r0 = np.clip(np.floor(row), 0, rows - 2).astype(int)
    fc, fr = col - c0, row - r0
    z = z.astype(np.float64)
    top = z[r0, c0] * (1 - fc) + z[r0, c0 + 1] * fc
    bottom = z[r0 + 1, c0] * (1 - fc) + z[r0 + 1, c0 + 1] * fc
    return top * (1 - fr) + bottom * fr
