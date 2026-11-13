"""The ground-truth terrain: the tile's DTM as a triangle mesh, textured with its orthoimage.

One vertex per DTM pixel, at the pixel centre and the DTM's height there, so the mesh
reproduces the DTM exactly at its samples and interpolates linearly between them. The
mesh is half a pixel smaller than the tile on each side; the texture still spans the
whole tile, and each vertex's UV points at the orthoimage under it.

Coordinates are local ground metres (x east, y north, z elevation) around the tile
centre, recorded as `origin` in mesh.json. The mosaics' equirectangular projection
stretches east-west by 1/cos(latitude), so x is map metres times `east_scale`
(cos latitude at the tile centre, from tile.json); y is map metres as they are. Raw map
metres near 4.6e6 m would also leave float32 vertex positions only 0.5 m apart.

With `curvature_radius_m` (the wide scene, #33) each vertex also drops by d^2 / 2R, d its
ground distance from the origin: Mars curving away from the tangent plane (25 m at 13 km).

Output is a binary PLY with per-vertex UVs (v = 0 at the image's bottom row, the
convention MeshLab, Blender and three.js use) plus texture.png and mesh.json. When the
input folder also holds a CTX ring (ring.json from fidelis-scene), the ring goes beside
it as ring.ply and ring_texture.png (`fidelis.terrain.ring`), in the same frame.

    uv run fidelis-mesh SCENE_DIR OUT_DIR
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio

from fidelis.terrain.matching_map import EPS
from fidelis.terrain.raster import centres
from fidelis.terrain.scene import to_local, write_json

# Vertex heights are DTM samples stored as float32: rounding, not modelling, is the error.
ELEVATION_TOLERANCE_M = 1e-3


def _read(path: Path) -> tuple[np.ndarray, rasterio.coords.BoundingBox, rasterio.Affine, float | None, str]:
    with rasterio.open(path) as src:
        return src.read(1), src.bounds, src.transform, src.nodata, src.crs.to_wkt()


def grid_mesh(
    heights: np.ndarray,
    transform: rasterio.Affine,
    east_scale: float,
    curvature_radius_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Positions (local ground metres), UVs and triangles for a north-up height grid; also the origin (its centre)."""
    rows, cols = heights.shape
    x, y = centres(transform, heights.shape)
    left, top = transform.c, transform.f
    right, bottom = left + cols * transform.a, top + rows * transform.e
    origin = [(left + right) / 2, (bottom + top) / 2]

    positions = to_local(x, y, heights, origin, east_scale, curvature_radius_m).reshape(-1, 3)
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


def read_ply_vertices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertex positions and UVs from a PLY that write_ply wrote."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    count = int(next(line.split()[2] for line in header if line.startswith("element vertex")))
    vertex = np.frombuffer(data, dtype=[("p", "<f4", 3), ("uv", "<f4", 2)], count=count, offset=end)
    return vertex["p"].copy(), vertex["uv"].copy()


def build_terrain_mesh(
    dtm_path: Path, ortho_path: Path, out_dir: Path, east_scale: float, curvature_radius_m: float | None = None
) -> dict:
    """Write mesh.ply, texture.png and mesh.json into out_dir; return the mesh.json record.

    east_scale is ground metres per map metre east-west (`tile.east_west_ground_scale`).
    With no curvature radius the mesh is flat.
    """
    heights, bounds, transform, nodata, crs = _read(dtm_path)
    image, ortho_bounds, _, _, _ = _read(ortho_path)
    if any(abs(a - b) > EPS for a, b in zip(bounds, ortho_bounds, strict=True)):
        raise ValueError(f"orthoimage bounds {tuple(ortho_bounds)} differ from the DTM's {tuple(bounds)}")
    holes = ~np.isfinite(heights)
    if nodata is not None:
        holes |= heights == np.float32(nodata)
    if holes.any():
        raise ValueError(f"{dtm_path} has {holes.sum()} nodata pixels; the mesh would have holes")

    positions, uvs, faces, origin = grid_mesh(heights, transform, east_scale, curvature_radius_m=curvature_radius_m)
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
        "east_scale": east_scale,
        "curvature_radius_m": curvature_radius_m,
        "frame": (
            "local ground metres: x east = (map x - origin x) * east_scale, y north = map y - origin y;"
            " z elevation as in the DTM, less (x^2 + y^2) / (2 curvature_radius_m) when that is set"
        ),
        "uv": "v = 0 at the texture's bottom row (OpenGL convention)",
        "vertex_count": len(positions),
        "face_count": len(faces),
        "elevation_tolerance_m": ELEVATION_TOLERANCE_M,
    }
    write_json(out_dir / "mesh.json", record)
    return record


def mitsuba_shape(mesh_dir: Path, ply: str = "mesh.ply", texture: str = "texture.png") -> dict:
    """The mesh as a Mitsuba shape dict, textured diffuse. Needs a Mitsuba variant set.

    Mitsuba's PLY loader keeps UVs as written, but its bitmaps put v = 0 at the image's
    top row, so the texture lookup flips v. The texture is loaded raw (no sRGB decoding):
    the orthoimage stores scaled reflectance, not a display-encoded photo.
    """
    import mitsuba as mi

    mesh_dir = Path(mesh_dir)
    return {
        "type": "ply",
        "filename": str(mesh_dir / ply),
        "bsdf": {
            "type": "diffuse",
            "reflectance": {
                "type": "bitmap",
                "filename": str(mesh_dir / texture),
                "raw": True,
                "to_uv": mi.ScalarTransform4f().translate([0, 1, 0]).scale([1, -1, 1]),
            },
        },
    }


def terrain_shapes(mesh_dir: Path, texture: str = "texture.png") -> dict:
    """Every terrain shape mesh.json lists, as Mitsuba shape dicts: the tile, and the CTX ring if built.

    `texture` is the tile's texture file in mesh_dir: the orthoimage as photographed, or
    `texture_deshaded.png` (`fidelis.terrain.deshade`). The ring keeps its own.
    """
    mesh_dir = Path(mesh_dir)
    shapes = {"terrain": mitsuba_shape(mesh_dir, texture=texture)}
    ring = json.loads((mesh_dir / "mesh.json").read_text()).get("ring")
    if ring is not None:
        shapes["ring"] = mitsuba_shape(mesh_dir, ring["ply"], ring["texture"])
    return shapes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scene_dir", type=Path, help="dtm.tif, ortho.tif and tile.json (fidelis-tile), plus the ring if fidelis-scene"
    )
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    scene = args.scene_dir
    tile = json.loads((scene / "tile.json").read_text())
    # fidelis-tile's flat tile records no radius; fidelis-scene's blended one does.
    radius = tile.get("curvature_radius_m")
    record = build_terrain_mesh(
        scene / "dtm.tif", scene / "ortho.tif", args.out_dir, tile["east_west_ground_scale"], radius
    )
    print(f"mesh {record['vertex_count']} vertices, {record['face_count']} triangles in {args.out_dir}")
    if (scene / "ring.json").exists():
        # Imported here: the ring module builds on this one.
        from fidelis.terrain.ring import build_ring_mesh

        record["ring"] = build_ring_mesh(
            scene / "ring_dtm.tif",
            scene / "ring_ortho.tif",
            scene / "dtm.tif",
            args.out_dir,
            tuple(record["origin"]),
            record["east_scale"],
            radius,
        )
        write_json(args.out_dir / "mesh.json", record)
        ring = record["ring"]
        print(f"ring {ring['vertex_count']} vertices, {ring['face_count']} triangles")
    return 0
