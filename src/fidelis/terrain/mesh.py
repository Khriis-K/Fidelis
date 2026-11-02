"""The ground-truth terrain: the tile's DTM as a triangle mesh, textured with its orthoimage.

One vertex per DTM pixel, at the pixel centre and the DTM's height there, so the mesh
reproduces the DTM exactly at its samples and interpolates linearly between them. The
mesh is half a pixel (0.5 m) smaller than the tile on each side; the texture still spans
the whole tile, and each vertex's UV points at the orthoimage under it.

Coordinates are local metres (x east, y north, z elevation) around the tile centre,
recorded as `origin` in mesh.json. Raw map metres near 4.6e6 m would leave float32
vertex positions only 0.5 m apart.

Output is a binary PLY with per-vertex UVs (v = 0 at the image's bottom row, the
convention MeshLab, Blender and three.js use) plus texture.png and mesh.json.

    uv run fidelis-mesh DTM ORTHO OUT_DIR
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio

from fidelis.terrain.tile import EPS

# Vertex heights are DTM samples stored as float32: rounding, not modelling, is the error.
ELEVATION_TOLERANCE_M = 1e-3


def _read(path: Path) -> tuple[np.ndarray, rasterio.coords.BoundingBox, rasterio.Affine, float | None, str]:
    with rasterio.open(path) as src:
        return src.read(1), src.bounds, src.transform, src.nodata, src.crs.to_wkt()


def grid_mesh(heights: np.ndarray, transform: rasterio.Affine) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Positions (local metres), UVs and triangles for a north-up height grid; also the origin."""
    rows, cols = heights.shape
    r, c = np.mgrid[:rows, :cols]
    x = transform.c + (c + 0.5) * transform.a
    y = transform.f + (r + 0.5) * transform.e
    left, top = transform.c, transform.f
    right, bottom = left + cols * transform.a, top + rows * transform.e
    origin = [(left + right) / 2, (bottom + top) / 2]

    positions = np.stack([x - origin[0], y - origin[1], heights], axis=-1).reshape(-1, 3).astype(np.float32)
    uvs = np.stack([(x - left) / (right - left), (y - bottom) / (top - bottom)], axis=-1).reshape(-1, 2)

    # Split each grid quad into two triangles, counter-clockwise seen from above (normals up).
    idx = np.arange(rows * cols).reshape(rows, cols)
    nw, ne, sw, se = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate([np.stack([sw, se, ne], -1).reshape(-1, 3), np.stack([sw, ne, nw], -1).reshape(-1, 3)])
    return positions, uvs.astype(np.float32), faces.astype(np.int32), origin


def write_ply(path: Path, positions: np.ndarray, uvs: np.ndarray, faces: np.ndarray, texture_name: str) -> None:
    vertex = np.empty(len(positions), dtype=[("p", "<f4", 3), ("uv", "<f4", 2)])
    vertex["p"], vertex["uv"] = positions, uvs
    face = np.empty(len(faces), dtype=[("n", "u1"), ("i", "<i4", 3)])
    face["n"], face["i"] = 3, faces
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment TextureFile {texture_name}\n"  # MeshLab's convention for a PLY's texture
        f"element vertex {len(vertex)}\n"
        # texture_u/v, not bare u/v: MeshLab drops bare u/v; Mitsuba and three.js read both.
        "property float x\nproperty float y\nproperty float z\nproperty float texture_u\nproperty float texture_v\n"
        f"element face {len(face)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    with open(tmp, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())
        f.write(face.tobytes())
    tmp.replace(path)


def build_terrain_mesh(dtm_path: Path, ortho_path: Path, out_dir: Path) -> dict:
    """Write mesh.ply, texture.png and mesh.json into out_dir; return the mesh.json record."""
    heights, bounds, transform, nodata, crs = _read(dtm_path)
    image, ortho_bounds, _, _, _ = _read(ortho_path)
    if any(abs(a - b) > EPS for a, b in zip(bounds, ortho_bounds, strict=True)):
        raise ValueError(f"orthoimage bounds {tuple(ortho_bounds)} differ from the DTM's {tuple(bounds)}")
    holes = ~np.isfinite(heights)
    if nodata is not None:
        holes |= heights == np.float32(nodata)
    if holes.any():
        raise ValueError(f"{dtm_path} has {holes.sum()} nodata pixels; the mesh would have holes")

    positions, uvs, faces, origin = grid_mesh(heights, transform)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "texture.tmp.png"
    if not cv2.imwrite(str(tmp), image):
        raise OSError(f"could not write {tmp}")
    tmp.replace(out_dir / "texture.png")
    write_ply(out_dir / "mesh.ply", positions, uvs, faces, "texture.png")

    record = {
        "dtm": str(dtm_path),
        "ortho": str(ortho_path),
        "crs": crs,
        "bounds": list(bounds),
        "origin": origin,
        "frame": "local metres: x east, y north from origin; z elevation as in the DTM",
        "uv": "v = 0 at the texture's bottom row (OpenGL convention)",
        "vertex_count": len(positions),
        "face_count": len(faces),
        "elevation_tolerance_m": ELEVATION_TOLERANCE_M,
    }
    tmp = out_dir / "mesh.json.tmp"
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(out_dir / "mesh.json")
    return record


def mitsuba_shape(mesh_dir: Path) -> dict:
    """The mesh as a Mitsuba shape dict, textured diffuse. Needs a Mitsuba variant set.

    Mitsuba's PLY loader keeps UVs as written, but its bitmaps put v = 0 at the image's
    top row, so the texture lookup flips v. The texture is loaded raw (no sRGB decoding):
    the orthoimage stores scaled reflectance, not a display-encoded photo.
    """
    import mitsuba as mi

    mesh_dir = Path(mesh_dir)
    return {
        "type": "ply",
        "filename": str(mesh_dir / "mesh.ply"),
        "bsdf": {
            "type": "diffuse",
            "reflectance": {
                "type": "bitmap",
                "filename": str(mesh_dir / "texture.png"),
                "raw": True,
                "to_uv": mi.ScalarTransform4f().translate([0, 1, 0]).scale([1, -1, 1]),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dtm", type=Path)
    parser.add_argument("ortho", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    record = build_terrain_mesh(args.dtm, args.ortho, args.out_dir)
    print(f"mesh {record['vertex_count']} vertices, {record['face_count']} triangles in {args.out_dir}")
    return 0
