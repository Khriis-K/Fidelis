import json
import sys
from collections import Counter

import mitsuba as mi
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fidelis.terrain.mesh import grid_mesh, main, terrain_shapes
from fidelis.terrain.ring import build_ring_mesh, ring_mesh

MARS_RADIUS_M = 3396190.0
MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"
EAST_SCALE = 0.9486
# A 600 m CTX-like ring grid at 20 m around a 100 m HiRISE-like tile at 4 m, whose edges
# don't fall on ring grid lines (like the real mosaics, which are off by a fraction of a pixel).
RING_LEFT, RING_TOP, RING_RES, RING_N = 4585000.0, 1098000.0, 20.0, 30
IN_LEFT, IN_TOP, IN_RES, IN_N = RING_LEFT + 203.25, RING_TOP - 197.5, 4.0, 25
ORIGIN = (IN_LEFT + IN_N * IN_RES / 2, IN_TOP - IN_N * IN_RES / 2)


@pytest.fixture(scope="module")
def grids():
    rng = np.random.default_rng(3)
    ring = (-2500 + rng.normal(0, 5, (RING_N, RING_N))).astype(np.float32)
    inner = (-2500 + rng.normal(0, 5, (IN_N, IN_N))).astype(np.float32)
    return (
        ring,
        from_origin(RING_LEFT, RING_TOP, RING_RES, RING_RES),
        inner,
        from_origin(IN_LEFT, IN_TOP, IN_RES, IN_RES),
    )


@pytest.fixture(scope="module")
def meshes(grids):
    ring, ring_t, inner, inner_t = grids
    inner_mesh = grid_mesh(inner, inner_t, EAST_SCALE, origin=ORIGIN, curvature_radius_m=MARS_RADIUS_M)[:3]
    ring_m = ring_mesh(ring, ring_t, inner, inner_t, ORIGIN, EAST_SCALE, MARS_RADIUS_M)
    return inner_mesh, ring_m


def edge_counts(*meshes):
    """How many triangles use each edge, with vertices identified by their exact float32 position."""
    counts = Counter()
    for positions, _, faces in meshes:
        keys = [p.tobytes() for p in positions]
        for tri in faces:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                counts[frozenset((keys[tri[a]], keys[tri[b]]))] += 1
    return counts


def test_the_ring_and_the_inner_tile_join_without_a_crack(meshes):
    inner, ring = meshes
    counts = edge_counts(inner, ring)

    # Every edge is shared by exactly two triangles, except along the scene's outer edge.
    assert max(counts.values()) == 2
    boundary = [e for e, n in counts.items() if n == 1]
    corners = ring[0][:, :2]
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    for edge in boundary:
        for key in edge:
            x, y, _ = np.frombuffer(key, dtype=np.float32)
            assert x in (lo[0], hi[0]) or y in (lo[1], hi[1])
    assert len(boundary) == 4 * (RING_N - 1)


def test_the_ring_alone_has_a_hole_where_the_inner_tile_goes(meshes):
    # Null case for the join test: without the inner tile, the ring's inner edge is open.
    _, ring = meshes
    assert sum(n == 1 for n in edge_counts(ring).values()) > 4 * (RING_N - 1)


def test_the_two_meshes_tile_the_scene_with_no_overlap(meshes):
    # Holes or overlaps would change the total area the triangles cover from above.
    inner, ring = meshes
    area = 0.0
    for positions, _, faces in (inner, ring):
        a, b, c = (positions[faces[:, i], :2].astype(np.float64) for i in range(3))
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        assert (cross > 0).all(), "every triangle faces up (counter-clockwise from above)"
        area += cross.sum() / 2

    span = (RING_N - 1) * RING_RES
    assert area == pytest.approx(span * EAST_SCALE * span, rel=1e-6)


def test_ring_vertices_sit_at_ctx_pixel_centres_and_heights_with_curvature(grids, meshes):
    ring, _, _, _ = grids
    _, (positions, _, _) = meshes
    grid = positions[: RING_N * RING_N].reshape(RING_N, RING_N, 3).astype(np.float64)
    cols, rows = np.meshgrid(np.arange(RING_N), np.arange(RING_N))
    x = (RING_LEFT + (cols + 0.5) * RING_RES - ORIGIN[0]) * EAST_SCALE
    y = RING_TOP - (rows + 0.5) * RING_RES - ORIGIN[1]

    np.testing.assert_allclose(grid[..., 0], x, atol=1e-3)
    np.testing.assert_allclose(grid[..., 1], y, atol=1e-3)
    np.testing.assert_allclose(grid[..., 2], ring - (x**2 + y**2) / (2 * MARS_RADIUS_M), atol=1e-3)


def test_ring_uvs_place_every_vertex_on_the_ring_texture_under_it(meshes):
    _, (positions, uvs, _) = meshes
    mx = positions[:, 0].astype(np.float64) / EAST_SCALE + ORIGIN[0]
    my = positions[:, 1].astype(np.float64) + ORIGIN[1]
    span = RING_N * RING_RES

    np.testing.assert_allclose(uvs[:, 0], (mx - RING_LEFT) / span, atol=1e-6)
    np.testing.assert_allclose(uvs[:, 1], (my - (RING_TOP - span)) / span, atol=1e-6)


def test_an_inner_tile_reaching_the_ring_edge_is_refused(grids):
    ring, ring_t, inner, _ = grids
    with pytest.raises(ValueError, match="inside"):
        ring_mesh(ring, ring_t, inner, from_origin(RING_LEFT + 5.0, IN_TOP, IN_RES, IN_RES), ORIGIN, 1.0, None)


# --- files on disk, through the mesh CLI ------------------------------------------------


def write_raster(path, data, left, top, res):
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": data.dtype,
        "crs": MARS_EQC,
        "transform": from_origin(left, top, res, res),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


@pytest.fixture
def scene_dir(grids, tmp_path):
    ring, _, inner, _ = grids
    scene = tmp_path / "scene"
    scene.mkdir()
    write_raster(scene / "dtm.tif", inner, IN_LEFT, IN_TOP, IN_RES)
    write_raster(scene / "ortho.tif", np.full((IN_N * 2, IN_N * 2), 90, np.uint8), IN_LEFT, IN_TOP, IN_RES / 2)
    write_raster(scene / "ring_dtm.tif", ring, RING_LEFT, RING_TOP, RING_RES)
    ortho_n = int(RING_N * RING_RES / 6)
    write_raster(scene / "ring_ortho.tif", np.full((ortho_n, ortho_n), 120, np.uint8), RING_LEFT, RING_TOP, 6.0)
    (scene / "tile.json").write_text(
        json.dumps({"east_west_ground_scale": EAST_SCALE, "curvature_radius_m": MARS_RADIUS_M})
    )
    (scene / "ring.json").write_text(json.dumps({"blend_band_m": 40.0}))
    return scene


def test_build_ring_mesh_writes_a_textured_ply(grids, scene_dir, tmp_path):
    record = build_ring_mesh(
        scene_dir / "ring_dtm.tif",
        scene_dir / "ring_ortho.tif",
        scene_dir / "dtm.tif",
        tmp_path / "mesh",
        ORIGIN,
        EAST_SCALE,
        MARS_RADIUS_M,
    )

    header = (tmp_path / "mesh" / "ring.ply").read_bytes().split(b"end_header")[0]
    assert b"comment TextureFile ring_texture.png" in header
    assert (tmp_path / "mesh" / "ring_texture.png").exists()
    assert record["face_count"] > 0


def test_the_mesh_cli_builds_the_ring_beside_the_inner_tile_in_one_frame(scene_dir, tmp_path, monkeypatch):
    out = tmp_path / "mesh"
    monkeypatch.setattr(sys, "argv", ["fidelis-mesh", str(scene_dir), str(out)])

    assert main() == 0

    record = json.loads((out / "mesh.json").read_text())
    assert record["curvature_radius_m"] == MARS_RADIUS_M
    assert record["ring"]["ply"] == "ring.ply"
    # The orbital map finds the ring's elevation from here, relative to the mesh folder.
    assert (out / record["ring"]["dtm"]).resolve() == (scene_dir / "ring_dtm.tif").resolve()

    mi.set_variant("scalar_rgb")
    shapes = terrain_shapes(out)
    assert set(shapes) == {"terrain", "ring"}
    scene = mi.load_dict({"type": "scene", **shapes})
    assert len(scene.shapes()) == 2


def test_without_a_ring_the_scene_is_the_inner_tile_alone(scene_dir, tmp_path, monkeypatch):
    (scene_dir / "ring.json").unlink()
    out = tmp_path / "mesh"
    monkeypatch.setattr(sys, "argv", ["fidelis-mesh", str(scene_dir), str(out)])

    assert main() == 0

    assert "ring" not in json.loads((out / "mesh.json").read_text())
    mi.set_variant("scalar_rgb")
    assert set(terrain_shapes(out)) == {"terrain"}
