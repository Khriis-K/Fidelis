import json
import sys

import mitsuba as mi
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fidelis.terrain.mesh import build_terrain_mesh, main, mitsuba_shape

MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"
# Map metres far from zero, like the real tile (~4.59e6 m east), so float32 precision bites.
LEFT, TOP = 4589870.0, 1094306.0
DTM_SHAPE = (8, 10)  # rows, cols; not square, so a transposed grid can't pass
ORTHO_SCALE = 4  # 0.25 m ortho pixels per 1 m DTM pixel, as in the Jezero tile
EAST_SCALE = 0.9486  # ground metres per map metre east-west at the Jezero landing site


def write_raster(path, data, res, nodata=None):
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": data.dtype,
        "crs": MARS_EQC,
        "transform": from_origin(LEFT, TOP, res, res),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


@pytest.fixture
def tile(tmp_path):
    """A rough DTM (so any shift or flip changes the heights) and a ramp orthoimage
    (each pixel's value says which column and row it is)."""
    rng = np.random.default_rng(0)
    heights = (-2600.0 + 20.0 * rng.random(DTM_SHAPE)).astype(np.float32)
    rows, cols = DTM_SHAPE[0] * ORTHO_SCALE, DTM_SHAPE[1] * ORTHO_SCALE
    r, c = np.mgrid[:rows, :cols]
    ortho = (c + 5 * r).astype(np.uint8)  # max 39 + 5 * 31 = 194, fits uint8
    write_raster(tmp_path / "dtm.tif", heights, 1.0, nodata=-3.4e38)
    write_raster(tmp_path / "ortho.tif", ortho, 1.0 / ORTHO_SCALE, nodata=0)
    return tmp_path / "dtm.tif", tmp_path / "ortho.tif", heights


@pytest.fixture
def built(tile, tmp_path):
    dtm, ortho, heights = tile
    record = build_terrain_mesh(dtm, ortho, tmp_path / "mesh", east_scale=EAST_SCALE)
    mi.set_variant("scalar_rgb")
    scene = mi.load_dict({"type": "scene", "terrain": mitsuba_shape(tmp_path / "mesh")})
    (shape,) = scene.shapes()
    return record, shape, heights


def map_xy(shape, record):
    """Vertex positions in map metres, from the mesh's ground-metre frame and its recorded origin and scale."""
    local = np.array(shape.vertex_positions_buffer()).reshape(-1, 3)
    ox, oy = record["origin"]
    return local[:, 0] / record["east_scale"] + ox, local[:, 1] + oy, local[:, 2]


def test_east_west_map_distances_shrink_to_ground_metres_and_north_south_ones_dont(built):
    # Neighbouring DTM pixels are 1 map metre apart both ways; on the ground, east-west
    # that's EAST_SCALE (cos latitude) metres, north-south a full metre.
    record, shape, _ = built
    local = np.array(shape.vertex_positions_buffer()).reshape(*DTM_SHAPE, 3)

    np.testing.assert_allclose(np.diff(local[:, :, 0], axis=1), EAST_SCALE, atol=1e-5)
    np.testing.assert_allclose(np.diff(local[:, :, 1], axis=0), -1.0, atol=1e-5)  # rows run north to south
    across = local[0, -1, 0] - local[0, 0, 0]
    assert across == pytest.approx(EAST_SCALE * (DTM_SHAPE[1] - 1), abs=1e-5)
    assert record["east_scale"] == EAST_SCALE


def test_the_mesh_has_one_vertex_per_dtm_pixel_and_two_triangles_per_quad(built):
    _, shape, _ = built
    rows, cols = DTM_SHAPE

    assert shape.vertex_count() == rows * cols
    assert shape.face_count() == 2 * (rows - 1) * (cols - 1)


def test_vertices_sit_on_dtm_pixel_centres_at_the_dtm_height(built):
    # The elevation acceptance check: every vertex is a DTM sample, so the stated
    # tolerance is float32 rounding (1 mm), checked at every vertex of the test tile.
    record, shape, heights = built
    x, y, z = map_xy(shape, record)

    col = (x - LEFT) - 0.5  # pixel-centre coordinates on the 1 m DTM grid
    row = (TOP - y) - 0.5
    np.testing.assert_allclose(col, np.round(col), atol=1e-3)
    np.testing.assert_allclose(row, np.round(row), atol=1e-3)
    np.testing.assert_allclose(z, heights[np.round(row).astype(int), np.round(col).astype(int)], atol=1e-3)
    assert record["elevation_tolerance_m"] == pytest.approx(1e-3)


def test_local_coordinates_stay_small_so_float32_keeps_millimetres(built):
    # Map metres near 4.6e6 have a float32 step of 0.5 m; the local frame avoids that.
    record, shape, _ = built
    local = np.array(shape.vertex_positions_buffer()).reshape(-1, 3)

    assert np.abs(local[:, :2]).max() < 10.0
    left, bottom, right, top = record["bounds"]
    assert record["origin"] == pytest.approx([(left + right) / 2, (bottom + top) / 2])


def test_every_triangle_faces_up(built):
    # Downward faces would render black under a sun from above.
    _, shape, _ = built
    p = np.array(shape.vertex_positions_buffer()).reshape(-1, 3)
    f = np.array(shape.faces_buffer()).reshape(-1, 3)

    normal_z = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])[:, 2]
    assert (normal_z > 0).all()


def test_the_texture_mitsuba_samples_at_each_vertex_is_the_orthoimage_under_it(built):
    # Registration on known reference points, through the renderer's own texture lookup.
    # The ramp orthoimage's value is col + 5 * row, and bilinear filtering of a ramp is
    # exact, so the sampled value pins down where in the image each vertex lands.
    record, shape, _ = built
    x, y, _ = map_xy(shape, record)
    uv = np.array(shape.vertex_texcoords_buffer()).reshape(-1, 2)

    bsdf = shape.bsdf()
    si = mi.SurfaceInteraction3f()
    sampled = []
    for u, v in uv:
        si.uv = mi.Point2f(u, v)
        sampled.append(bsdf.eval_diffuse_reflectance(si)[0])

    col = (x - LEFT) * ORTHO_SCALE - 0.5  # ortho pixel-centre coordinates of each vertex
    row = (TOP - y) * ORTHO_SCALE - 0.5
    np.testing.assert_allclose(np.array(sampled) * 255, col + 5 * row, atol=0.05)


def test_uvs_follow_the_standard_convention_with_v_up(built):
    # Viewers (MeshLab, Blender, three.js) put v = 0 at the image's bottom row.
    record, shape, _ = built
    _, y, _ = map_xy(shape, record)
    v = np.array(shape.vertex_texcoords_buffer()).reshape(-1, 2)[:, 1]
    _, bottom, _, top = record["bounds"]

    np.testing.assert_allclose(v, (y - bottom) / (top - bottom), atol=1e-6)


def test_the_ply_names_its_texture_for_viewers(tile, tmp_path):
    dtm, ortho, _ = tile
    build_terrain_mesh(dtm, ortho, tmp_path / "mesh", east_scale=EAST_SCALE)

    header = (tmp_path / "mesh" / "mesh.ply").read_bytes().split(b"end_header")[0]
    assert b"comment TextureFile texture.png" in header
    assert (tmp_path / "mesh" / "texture.png").exists()


def test_the_ply_uses_the_uv_names_meshlab_reads(tile, tmp_path):
    # MeshLab ignores bare u/v and shows the mesh untextured; texture_u/texture_v is
    # what it writes itself, and Mitsuba and three.js read it too.
    dtm, ortho, _ = tile
    build_terrain_mesh(dtm, ortho, tmp_path / "mesh", east_scale=EAST_SCALE)

    header = (tmp_path / "mesh" / "mesh.ply").read_bytes().split(b"end_header")[0]
    assert b"property float texture_u\nproperty float texture_v\n" in header


def test_the_record_is_written_beside_the_mesh(tile, tmp_path):
    dtm, ortho, _ = tile
    record = build_terrain_mesh(dtm, ortho, tmp_path / "mesh", east_scale=EAST_SCALE)

    assert json.loads((tmp_path / "mesh" / "mesh.json").read_text()) == record
    assert record["crs"]


def test_the_cli_takes_the_east_west_scale_from_the_tile_record(tile, tmp_path, monkeypatch):
    # fidelis-tile writes the scale; a key mismatch here would only show up in the pipeline.
    (tmp_path / "tile.json").write_text(json.dumps({"east_west_ground_scale": 0.9}))
    monkeypatch.setattr(sys, "argv", ["fidelis-mesh", str(tmp_path), str(tmp_path / "mesh")])

    assert main() == 0

    record = json.loads((tmp_path / "mesh" / "mesh.json").read_text())
    assert record["east_scale"] == 0.9
    local = np.frombuffer(
        (tmp_path / "mesh" / "mesh.ply").read_bytes().split(b"end_header\n")[1],
        dtype="<f4",
        count=record["vertex_count"] * 5,
    ).reshape(*DTM_SHAPE, 5)
    np.testing.assert_allclose(np.diff(local[:, :, 0], axis=1), 0.9, atol=1e-5)


def test_a_dtm_with_nodata_is_rejected(tile, tmp_path):
    # A hole would become a spike to -3.4e38 m; refuse instead of meshing it.
    dtm, ortho, heights = tile
    heights = heights.copy()
    heights[3, 4] = -3.4e38
    write_raster(dtm, heights, 1.0, nodata=-3.4e38)

    with pytest.raises(ValueError, match="nodata"):
        build_terrain_mesh(dtm, ortho, tmp_path / "mesh", east_scale=EAST_SCALE)


def test_an_orthoimage_with_other_bounds_is_rejected(tile, tmp_path):
    # UVs assume the image spans exactly the DTM's footprint.
    dtm, _, _ = tile
    small = np.zeros((DTM_SHAPE[0] * ORTHO_SCALE - 4, DTM_SHAPE[1] * ORTHO_SCALE), dtype=np.uint8)
    write_raster(tmp_path / "small.tif", small, 1.0 / ORTHO_SCALE)

    with pytest.raises(ValueError, match="bounds"):
        build_terrain_mesh(dtm, tmp_path / "small.tif", tmp_path / "mesh", east_scale=EAST_SCALE)
