"""The orbital map as the navigator uses it: an image to match, and where each map pixel is in the world.

The image is the CTX-like matching map (`fidelis.terrain.matching_map`). PnP needs a 3D
world point for every matched map pixel, so each one is placed in the mesh's world frame
(docs/camera-conventions.md): map metres from the map's geotransform, then ground metres
from mesh.json's origin and east_scale, and elevation from the DTM beneath it.

Elevation comes from the same DTM the mesh was built from, so the map's terrain model
is as good as the rendered terrain's. LVS's onboard elevation map was coarser; that
error isn't modelled.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

# Map footprint edges may sit this far outside the DTM's (float rounding in geotransforms).
FOOTPRINT_TOLERANCE_M = 1e-6


def _footprint(transform: Affine, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """(left, bottom, right, top) in map metres of a north-up raster."""
    left, top = transform @ (0, 0)
    right, bottom = transform @ (shape[1], shape[0])
    return left, bottom, right, top


@dataclass(frozen=True)
class OrbitalMap:
    image: np.ndarray  # H x W float32 in [0, 1]
    transform: Affine  # rasterio's: map pixel corner (col, row) to map metres
    dtm: np.ndarray  # elevation, metres
    dtm_transform: Affine
    origin: tuple[float, float]  # world (0, 0) in map metres, from mesh.json
    east_scale: float  # ground metres per map metre east-west, from mesh.json

    def __post_init__(self):
        if self.image.ndim != 2 or not (0 <= self.image.min() and self.image.max() <= 1):
            raise ValueError("map image must be a single-channel image in [0, 1]")
        left, bottom, right, top = _footprint(self.transform, self.image.shape)
        d_left, d_bottom, d_right, d_top = _footprint(self.dtm_transform, self.dtm.shape)
        tol = FOOTPRINT_TOLERANCE_M
        if left < d_left - tol or bottom < d_bottom - tol or right > d_right + tol or top > d_top + tol:
            raise ValueError(
                f"map footprint {(left, bottom, right, top)} reaches past the DTM's "
                f"{(d_left, d_bottom, d_right, d_top)}, where elevation is unknown"
            )

    def pixel_to_world(self, pixels: np.ndarray) -> np.ndarray:
        """N x 3 world points for N x 2 map pixels (col, row), OpenCV's centre-at-integer convention."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        mx, my = self.transform @ (pixels[:, 0] + 0.5, pixels[:, 1] + 0.5)
        x = (mx - self.origin[0]) * self.east_scale
        y = my - self.origin[1]
        return np.column_stack([x, y, self._elevation(mx, my)])

    def ground_height(self, east: float, north: float) -> float:
        """DTM elevation beneath a world position (ground metres east and north of world (0, 0))."""
        mx = np.array([east / self.east_scale + self.origin[0]])
        my = np.array([north + self.origin[1]])
        return float(self._elevation(mx, my)[0])

    def _elevation(self, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
        """Bilinear in the DTM's pixel centres; past the outermost centres it extrapolates the edge cell."""
        col, row = ~self.dtm_transform @ (mx, my)
        col, row = col - 0.5, row - 0.5
        rows, cols = self.dtm.shape
        c0 = np.clip(np.floor(col), 0, cols - 2).astype(int)
        r0 = np.clip(np.floor(row), 0, rows - 2).astype(int)
        fc, fr = col - c0, row - r0
        z = self.dtm.astype(np.float64)
        top = z[r0, c0] * (1 - fc) + z[r0, c0 + 1] * fc
        bottom = z[r0 + 1, c0] * (1 - fc) + z[r0 + 1, c0 + 1] * fc
        return top * (1 - fr) + bottom * fr


def load_orbital_map(map_path: Path, dtm_path: Path, mesh_json_path: Path) -> OrbitalMap:
    """The workflow's matching map (8-bit), the tile DTM, and the mesh's world frame."""
    with rasterio.open(map_path) as src:
        image = src.read(1)
        transform = src.transform
    if image.dtype != np.uint8:
        raise ValueError(f"{map_path} is {image.dtype}; the matching map is 8-bit")
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1)
        dtm_transform = src.transform
    frame = json.loads(Path(mesh_json_path).read_text())
    return OrbitalMap(
        image=image.astype(np.float32) / 255,
        transform=transform,
        dtm=dtm,
        dtm_transform=dtm_transform,
        origin=tuple(frame["origin"]),
        east_scale=frame["east_scale"],
    )
