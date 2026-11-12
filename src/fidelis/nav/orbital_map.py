"""The orbital map as the navigator uses it: an image to match, and where each map pixel is in the world.

PnP needs a 3D world point for every matched map pixel, so each one is placed in the mesh's
world frame (docs/camera-conventions.md): map metres from the map's geotransform, then
ground metres from mesh.json's origin and east_scale, and elevation from a DTM beneath it,
less the mesh's curvature drop. `Terrain` is that elevation lookup on its own.

There are two maps (#36), and the scene config's `matching_map.source` picks one:

- **The LVS map** (`load_lvs_map`): the USGS CTX orthomosaic (`fidelis.terrain.lvs_map`),
  with elevation from the CTX DTM everywhere. LVS's onboard map paired a CTX appearance map
  with a CTX-derived elevation map (docs/research/lvs-error-sources.md, section 2), so the navigator is given
  no HiRISE-derived information at all. Where the CTX DTM differs from the terrain the
  scene renders (median +0.32 m, sd 3.11 m over the tile; docs/benchmarks/wide-scene.md),
  that is real map elevation error and stays in. USGS's CTX DTM is 20 m/pixel, coarser than
  the 12 m LVS required of its elevation map.
- **The scene's own map** (`load_orbital_map`): a block average of the orthoimage the scene
  renders (`fidelis.terrain.matching_map`), with elevation from the DTMs the mesh was built
  from: the inner (HiRISE) DTM where it reaches, the CTX ring's beyond (#33). Map and scene
  agree exactly, which is why it was replaced; it stays so earlier results reproduce. Across
  the narrow stitch between the two meshes the ring DTM stands in for the stitch triangles;
  they meet the same heights at both edges.

The truth (where the camera really is) always comes from the scene's terrain (`load_terrain`),
whichever map the navigator holds.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from fidelis.terrain.raster import bilinear
from fidelis.terrain.scene import curvature_drop

# Map footprint edges may sit this far outside the DTM's (float rounding in geotransforms).
FOOTPRINT_TOLERANCE_M = 1e-6


def _footprint(transform: Affine, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """(left, bottom, right, top) in map metres of a north-up raster."""
    left, top = transform @ (0, 0)
    right, bottom = transform @ (shape[1], shape[0])
    return left, bottom, right, top


@dataclass(frozen=True)
class Terrain:
    """Elevation in the mesh's world frame: a DTM, an optional ring DTM around it, and the curvature drop."""

    dtm: np.ndarray  # elevation, metres
    dtm_transform: Affine
    origin: tuple[float, float]  # world (0, 0) in map metres, from mesh.json
    east_scale: float  # ground metres per map metre east-west, from mesh.json
    ring_dtm: np.ndarray | None = None  # the CTX ring's elevation, around and beyond dtm
    ring_dtm_transform: Affine | None = None
    curvature_radius_m: float | None = None  # from mesh.json; None for a flat frame

    def footprint(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) in map metres of everywhere elevation is known."""
        outer = (self.dtm, self.dtm_transform) if self.ring_dtm is None else (self.ring_dtm, self.ring_dtm_transform)
        return _footprint(outer[1], outer[0].shape)

    def world(self, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
        """N x 3 world points on the terrain beneath N map-metre positions."""
        x = (mx - self.origin[0]) * self.east_scale
        y = my - self.origin[1]
        return np.column_stack([x, y, self._elevation(mx, my) - self._drop(x, y)])

    def ground_height(self, east: float, north: float) -> float:
        """Elevation beneath a world position (ground metres east and north of world (0, 0))."""
        mx = np.array([east / self.east_scale + self.origin[0]])
        my = np.array([north + self.origin[1]])
        return float(self._elevation(mx, my)[0] - self._drop(east, north))

    def _drop(self, x, y):
        return 0.0 if self.curvature_radius_m is None else curvature_drop(x, y, self.curvature_radius_m)

    def _elevation(self, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
        """DTM elevation, bilinear in pixel centres: the inner DTM's within its outermost centres, else the ring's.

        Without a ring, past the inner DTM's outermost centres it extrapolates the edge cell.
        """
        inner = bilinear(self.dtm, self.dtm_transform, mx, my)
        if self.ring_dtm is None:
            return inner
        col, row = ~self.dtm_transform @ (mx, my)
        rows, cols = self.dtm.shape
        inside = (col >= 0.5) & (col <= cols - 0.5) & (row >= 0.5) & (row <= rows - 0.5)
        return np.where(inside, inner, bilinear(self.ring_dtm, self.ring_dtm_transform, mx, my))


@dataclass(frozen=True)
class OrbitalMap:
    image: np.ndarray  # H x W float32 in [0, 1]
    transform: Affine  # rasterio's: map pixel corner (col, row) to map metres
    dtm: np.ndarray  # elevation, metres
    dtm_transform: Affine
    origin: tuple[float, float]  # world (0, 0) in map metres, from mesh.json
    east_scale: float  # ground metres per map metre east-west, from mesh.json
    ring_dtm: np.ndarray | None = None  # the CTX ring's elevation, around and beyond dtm
    ring_dtm_transform: Affine | None = None
    curvature_radius_m: float | None = None  # from mesh.json; None for a flat frame

    def __post_init__(self):
        if self.image.ndim != 2 or not (0 <= self.image.min() and self.image.max() <= 1):
            raise ValueError("map image must be a single-channel image in [0, 1]")
        left, bottom, right, top = _footprint(self.transform, self.image.shape)
        d_left, d_bottom, d_right, d_top = self.terrain.footprint()
        tol = FOOTPRINT_TOLERANCE_M
        if left < d_left - tol or bottom < d_bottom - tol or right > d_right + tol or top > d_top + tol:
            raise ValueError(
                f"map footprint {(left, bottom, right, top)} reaches past the DTM's "
                f"{(d_left, d_bottom, d_right, d_top)}, where elevation is unknown"
            )

    @property
    def terrain(self) -> Terrain:
        """The elevation beneath the map's pixels."""
        return Terrain(
            self.dtm,
            self.dtm_transform,
            self.origin,
            self.east_scale,
            self.ring_dtm,
            self.ring_dtm_transform,
            self.curvature_radius_m,
        )

    def pixel_to_world(self, pixels: np.ndarray) -> np.ndarray:
        """N x 3 world points for N x 2 map pixels (col, row), OpenCV's centre-at-integer convention."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        return self.terrain.world(*(self.transform @ (pixels[:, 0] + 0.5, pixels[:, 1] + 0.5)))

    def ground_height(self, east: float, north: float) -> float:
        """The map's DTM elevation beneath a world position (ground metres east and north of world (0, 0))."""
        return self.terrain.ground_height(east, north)


def load_terrain(dtm_path: Path, mesh_json_path: Path, ring: bool = True) -> Terrain:
    """A DTM in the mesh's world frame, with the ring DTM mesh.json names (relative to its folder) around it.

    `ring=False` leaves the ring out, for a DTM that already covers everything asked of it.
    """
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1)
        dtm_transform = src.transform
    frame = json.loads(Path(mesh_json_path).read_text())
    ring_dtm = ring_transform = None
    if ring and "ring" in frame:
        with rasterio.open(Path(mesh_json_path).parent / frame["ring"]["dtm"]) as src:
            ring_dtm, ring_transform = src.read(1), src.transform
    return Terrain(
        dtm=dtm,
        dtm_transform=dtm_transform,
        origin=tuple(frame["origin"]),
        east_scale=frame["east_scale"],
        ring_dtm=ring_dtm,
        ring_dtm_transform=ring_transform,
        curvature_radius_m=frame.get("curvature_radius_m"),
    )


def _map_on_terrain(map_path: Path, terrain: Terrain) -> OrbitalMap:
    """An 8-bit map image with `terrain` as the elevation beneath its pixels."""
    with rasterio.open(map_path) as src:
        image = src.read(1)
        transform = src.transform
    if image.dtype != np.uint8:
        raise ValueError(f"{map_path} is {image.dtype}; the matching map is 8-bit")
    return OrbitalMap(
        image=image.astype(np.float32) / 255,
        transform=transform,
        dtm=terrain.dtm,
        dtm_transform=terrain.dtm_transform,
        origin=terrain.origin,
        east_scale=terrain.east_scale,
        ring_dtm=terrain.ring_dtm,
        ring_dtm_transform=terrain.ring_dtm_transform,
        curvature_radius_m=terrain.curvature_radius_m,
    )


def load_orbital_map(map_path: Path, dtm_path: Path, mesh_json_path: Path) -> OrbitalMap:
    """The scene's own matching map (8-bit), the scene's DTMs, and the mesh's world frame."""
    return _map_on_terrain(map_path, load_terrain(dtm_path, mesh_json_path))


def load_lvs_map(map_path: Path, ctx_dtm_path: Path, mesh_json_path: Path) -> OrbitalMap:
    """The LVS map (8-bit CTX) with CTX elevation everywhere, in the mesh's world frame.

    `ctx_dtm_path` is the CTX DTM over the whole scene (the workflow's scene/ring_dtm.tif).
    """
    return _map_on_terrain(map_path, load_terrain(ctx_dtm_path, mesh_json_path, ring=False))
