"""Where a north-up raster's pixels are, and its values between them."""

import numpy as np
from rasterio.transform import Affine


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
