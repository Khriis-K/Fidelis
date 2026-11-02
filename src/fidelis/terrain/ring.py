"""The CTX ring: a 20 m terrain mesh around the HiRISE tile, joined to its edge with no crack.

The ring is one vertex per CTX DTM pixel centre, like the tile's mesh, minus a rectangular
hole where the tile goes. The two grids don't line up (different spacing, and CTX sits a
fraction of a pixel off HiRISE), so a strip of triangles stitches the hole's edge to the
tile mesh's outermost vertices. The strip uses exact copies of those vertices, so the two
shapes share every edge along the seam and a ray can't slip between them.

Both meshes are in the scene's one local frame (`fidelis.terrain.scene.to_local`): the
tile's origin, east scale and curvature.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine

from fidelis.terrain.mesh import write_ply
from fidelis.terrain.raster import centres
from fidelis.terrain.scene import to_local
from fidelis.terrain.tile import EPS


def _zip(a: np.ndarray, ta: np.ndarray, b: np.ndarray, tb: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangles between two vertex chains running the same way along parallel lines.

    Steps along whichever chain's next vertex comes first (by t, distance along the
    side), so triangles never cross. Both chains start and end at corners.
    """
    i = j = 0
    triangles = []
    while i < len(a) - 1 or j < len(b) - 1:
        if j == len(b) - 1 or (i < len(a) - 1 and ta[i + 1] <= tb[j + 1]):
            triangles.append((a[i], a[i + 1], b[j]))
            i += 1
        else:
            triangles.append((a[i], b[j + 1], b[j]))
            j += 1
    return triangles


def _sides(r0: int, r1: int, c0: int, c1: int) -> list[tuple]:
    """A rectangle's four sides as (row index, col index) arrays, counter-clockwise from above.

    South (west to east), east (south to north), north (east to west), west (north to south).
    Grid rows run north to south.
    """
    c, r = np.arange(c0, c1 + 1), np.arange(r0, r1 + 1)
    return [
        (np.full(len(c), r1), c),
        (r[::-1], np.full(len(r), c1)),
        (np.full(len(c), r0), c[::-1]),
        (r, np.full(len(r), c0)),
    ]


def ring_mesh(
    ring_heights: np.ndarray,
    ring_transform: Affine,
    inner_heights: np.ndarray,
    inner_transform: Affine,
    origin: tuple[float, float],
    east_scale: float,
    radius_m: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions, UVs (over the ring raster's footprint, v up) and triangles of the ring and its stitch.

    The first rows x cols vertices are the ring grid in raster order; those in the hole are
    left unreferenced. Then come copies of the tile's outermost vertices.
    """
    rows, cols = ring_heights.shape
    rmx, rmy = centres(ring_transform, ring_heights.shape)
    imx, imy = centres(inner_transform, inner_heights.shape)
    inner = to_local(imx, imy, inner_heights, origin, east_scale, radius_m)

    # Drop ring vertices within half a ring pixel of the tile mesh's edge, so the stitch
    # strip is never thinner than that.
    gap = abs(ring_transform.a) / 2
    near = (rmx > imx.min() - gap) & (rmx < imx.max() + gap) & (rmy > imy.min() - gap) & (rmy < imy.max() + gap)
    hole_rows, hole_cols = np.flatnonzero(near.any(axis=1)), np.flatnonzero(near.any(axis=0))
    if not near.any() or hole_rows[0] < 1 or hole_cols[0] < 1 or hole_rows[-1] > rows - 2 or hole_cols[-1] > cols - 2:
        raise ValueError("the inner tile must sit inside the ring, at least a ring pixel clear of its edge")
    r0, r1, c0, c1 = hole_rows[0] - 1, hole_rows[-1] + 1, hole_cols[0] - 1, hole_cols[-1] + 1

    idx = np.arange(rows * cols).reshape(rows, cols)
    keep = ~(near[:-1, :-1] | near[:-1, 1:] | near[1:, :-1] | near[1:, 1:])
    nw, ne, sw, se = idx[:-1, :-1][keep], idx[:-1, 1:][keep], idx[1:, :-1][keep], idx[1:, 1:][keep]
    faces = [np.stack([sw, se, ne], -1), np.stack([sw, ne, nw], -1)]

    # The tile's outermost vertices, appended after the ring grid, in the same side order.
    in_rows, in_cols = inner_heights.shape
    edge = np.zeros(inner_heights.shape, bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    copy_index = np.full(inner_heights.shape, -1)
    copy_index[edge] = rows * cols + np.arange(edge.sum())
    positions = np.concatenate(
        [to_local(rmx, rmy, ring_heights, origin, east_scale, radius_m).reshape(-1, 3), inner[edge]]
    )
    mx = np.concatenate([rmx.ravel(), imx[edge]])
    my = np.concatenate([rmy.ravel(), imy[edge]])

    stitch = []
    outer_sides = _sides(r0, r1, c0, c1)
    inner_sides = _sides(0, in_rows - 1, 0, in_cols - 1)
    for side, ((ar, ac), (br, bc)) in enumerate(zip(inner_sides, outer_sides, strict=True)):
        a, b = copy_index[ar, ac], idx[br, bc]
        # Distance along the side: east, north, west, south in turn.
        axis, sign = (0, 1, 0, 1)[side], (1, 1, -1, -1)[side]
        stitch += _zip(a, sign * positions[a, axis], b, sign * positions[b, axis])
    stitch = np.array(stitch)
    # Wind every stitch triangle counter-clockwise seen from above, so normals point up.
    p = positions[stitch][..., :2].astype(np.float64)
    cross = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (
        p[:, 2, 0] - p[:, 0, 0]
    )
    stitch[cross < 0] = stitch[cross < 0][:, [0, 2, 1]]
    faces.append(stitch)

    left, top = ring_transform.c, ring_transform.f
    right, bottom = left + cols * ring_transform.a, top + rows * ring_transform.e
    uvs = np.stack([(mx - left) / (right - left), (my - bottom) / (top - bottom)], axis=-1)
    return positions, uvs.astype(np.float32), np.concatenate(faces).astype(np.int32)


def build_ring_mesh(
    ring_dtm: Path,
    ring_ortho: Path,
    inner_dtm: Path,
    out_dir: Path,
    origin: tuple[float, float],
    east_scale: float,
    radius_m: float | None,
) -> dict:
    """Write ring.ply and ring_texture.png into out_dir; return the ring's part of mesh.json."""
    with rasterio.open(ring_dtm) as src:
        heights, transform, bounds, nodata = src.read(1), src.transform, src.bounds, src.nodata
    with rasterio.open(ring_ortho) as src:
        image, ortho_bounds = src.read(1), src.bounds
    with rasterio.open(inner_dtm) as src:
        inner_heights, inner_transform = src.read(1), src.transform
    if any(abs(a - b) > EPS for a, b in zip(bounds, ortho_bounds, strict=True)):
        raise ValueError(f"ring orthoimage bounds {tuple(ortho_bounds)} differ from the ring DTM's {tuple(bounds)}")
    if nodata is not None and (heights == np.float32(nodata)).any():
        raise ValueError(f"{ring_dtm} has nodata pixels; the ring would have holes")

    positions, uvs, faces = ring_mesh(heights, transform, inner_heights, inner_transform, origin, east_scale, radius_m)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "ring_texture.tmp.png"
    if not cv2.imwrite(str(tmp), image):
        raise OSError(f"could not write {tmp}")
    tmp.replace(out_dir / "ring_texture.png")
    write_ply(out_dir / "ring.ply", positions, uvs, faces, "ring_texture.png")
    return {
        "ply": "ring.ply",
        "texture": "ring_texture.png",
        # Relative to the mesh folder, so the orbital map can find the ring's elevation.
        "dtm": os.path.relpath(Path(ring_dtm).resolve(), out_dir.resolve()).replace(os.sep, "/"),
        "ortho": str(ring_ortho),
        "bounds": list(bounds),
        "vertex_count": len(positions),
        "face_count": len(faces),
    }
