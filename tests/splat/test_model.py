"""The Gaussian scene: started on the mesh's own vertices, saved and loaded as a checkpoint,
and exported as a standard 3DGS PLY that browser splat renderers read. CPU only."""

import math

import cv2
import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from fidelis.render.mesh_render import Sun
from fidelis.splat.model import SH_C0, export_ply, gaussians, init_from_mesh, load_checkpoint, save_checkpoint
from fidelis.terrain.mesh import build_terrain_mesh

MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"
LEFT, TOP = 4589870.0, 1094306.0
SIZE = 20  # DTM pixels a side, 2 m each
SUN = Sun(257.0, 40.0)


def write_raster(path, data, res):
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": data.dtype,
        "crs": MARS_EQC,
        "transform": from_origin(LEFT, TOP, res, res),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


@pytest.fixture
def mesh_dir(tmp_path):
    """A gently sloping 40 m tile at 2 m spacing, textured with a west-east brightness ramp at 1 m."""
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    write_raster(tmp_path / "dtm.tif", (-2500.0 + 0.5 * cols + 0.25 * rows).astype(np.float32), 2.0)
    ramp = np.tile(np.linspace(40, 220, 2 * SIZE).astype(np.uint8), (2 * SIZE, 1))
    write_raster(tmp_path / "ortho.tif", ramp, 1.0)
    build_terrain_mesh(tmp_path / "dtm.tif", tmp_path / "ortho.tif", tmp_path / "mesh", east_scale=0.95)
    return tmp_path / "mesh"


def read_vertices(path):
    """The mesh PLY's vertex positions and UVs (fidelis.terrain.mesh.write_ply's layout)."""
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    count = int(next(line.split()[2] for line in data[:end].decode().splitlines() if line.startswith("element vertex")))
    vertex = np.frombuffer(data, dtype=[("p", "<f4", 3), ("uv", "<f4", 2)], count=count, offset=end)
    return vertex["p"].astype(np.float64), vertex["uv"].astype(np.float64)


def read_splat_ply(path):
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode().splitlines()
    assert header[1] == "format binary_little_endian 1.0"
    count = int(next(line.split()[2] for line in header if line.startswith("element vertex")))
    names = [line.split()[2] for line in header if line.startswith("property")]
    assert all(line.split()[1] == "float" for line in header if line.startswith("property"))
    values = np.frombuffer(data, dtype="<f4", count=count * len(names), offset=end).reshape(count, len(names))
    assert end + values.nbytes == len(data), "trailing bytes after the vertex data"
    return names, values


# --- starting on the mesh


def test_one_gaussian_starts_on_every_mesh_vertex(mesh_dir):
    params = init_from_mesh(mesh_dir, SUN)
    positions, _ = read_vertices(mesh_dir / "mesh.ply")
    np.testing.assert_allclose(params["means"].numpy(), positions, atol=1e-3)


def test_gaussians_start_with_the_texture_under_them_lit_by_the_sun(mesh_dir):
    # Flat ground under a sun at elevation e has radiance albedo * sin(e) (mesh_render's
    # radiometry), so that's where each Gaussian's grey starts.
    params = init_from_mesh(mesh_dir, SUN)
    _, uvs = read_vertices(mesh_dir / "mesh.ply")
    texture = cv2.imread(str(mesh_dir / "texture.png"), cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255
    height, width = texture.shape
    # Each 2 m vertex sits on the corner between four 1 m texture pixels; the ramp runs west-east only.
    col = uvs[:, 0] * width - 0.5
    left = np.floor(col).astype(int)
    expected_albedo = texture[0, left] + (col - left) * (texture[0, left + 1] - texture[0, left])
    expected = expected_albedo * math.sin(math.radians(SUN.elevation_deg))
    np.testing.assert_allclose(params["colors"].numpy()[:, 0], expected, atol=1e-4)


def test_the_textures_top_row_lights_the_northern_edge(mesh_dir):
    # PNG rows run north to south while UV v runs south to north: a north-south ramp
    # catches a lookup that forgets to flip v, which the west-east ramp above can't.
    ramp = np.repeat(np.linspace(220, 40, 2 * SIZE).astype(np.uint8)[:, None], 2 * SIZE, axis=1)
    cv2.imwrite(str(mesh_dir / "texture.png"), ramp)
    params = init_from_mesh(mesh_dir, SUN)
    positions, uvs = read_vertices(mesh_dir / "mesh.ply")
    row = (1 - uvs[:, 1]) * 2 * SIZE - 0.5
    top = np.floor(row).astype(int)
    texture = ramp[:, 0].astype(np.float64) / 255
    expected = (texture[top] + (row - top) * (texture[top + 1] - texture[top])) * math.sin(math.radians(40.0))
    np.testing.assert_allclose(params["colors"].numpy()[:, 0], expected, atol=1e-4)
    north = positions[:, 1] == positions[:, 1].max()
    assert params["colors"][north, 0].mean() > params["colors"][~north, 0].mean()


def test_gaussians_start_round_and_as_wide_as_the_vertex_spacing(mesh_dir):
    params = init_from_mesh(mesh_dir, SUN)
    scales = params["scales"].exp().numpy()
    # 2 m DTM pixels; 1.9 m east-west at east_scale 0.95: the spacing is between the two.
    assert np.all(scales[:, 0] == scales[:, 1]) and np.all(scales[:, 1] == scales[:, 2])
    assert 1.85 < scales[0, 0] < 2.05


def test_the_ring_gets_gaussians_too_when_the_mesh_has_one(mesh_dir):
    import json

    from fidelis.terrain.mesh import write_ply

    record = json.loads((mesh_dir / "mesh.json").read_text())
    ring = np.array([[100.0, 0.0, -2490.0], [120.0, 0.0, -2490.0], [140.0, 0.0, -2490.0]])
    write_ply(mesh_dir / "ring.ply", ring, np.full((3, 2), 0.5, np.float32), np.zeros((0, 3), np.int32), "t.png")
    cv2.imwrite(str(mesh_dir / "ring_texture.png"), np.full((4, 4), 255, np.uint8))
    record["ring"] = {"ply": "ring.ply", "texture": "ring_texture.png"}
    (mesh_dir / "mesh.json").write_text(json.dumps(record))

    params = init_from_mesh(mesh_dir, SUN)
    positions, _ = read_vertices(mesh_dir / "mesh.ply")
    assert len(params["means"]) == len(positions) + 3
    np.testing.assert_allclose(params["means"][-3:].numpy(), ring, atol=1e-3)
    np.testing.assert_allclose(params["colors"][-3:, 0].numpy(), math.sin(math.radians(40.0)), atol=1e-4)
    # The ring's 20 m spacing, not the tile's 2 m.
    assert params["scales"][-1, 0].exp() == pytest.approx(20.0, rel=1e-4)


# --- the parameters, a checkpoint and the browser export


def two_gaussians():
    return gaussians(
        means=np.array([[1.0, 2.0, -2500.0], [-3.0, 4.0, -2501.0]]),
        greys=np.array([0.25, 0.6]),
        spacing=np.array([2.0, 4.0]),
    )


def test_parameters_are_stored_unconstrained_as_gsplat_trains_them():
    params = two_gaussians()
    assert set(params) == {"means", "scales", "quats", "opacities", "colors"}
    np.testing.assert_allclose(params["scales"].exp().numpy(), [[2.0] * 3, [4.0] * 3], rtol=1e-6)
    np.testing.assert_allclose(params["quats"].numpy(), [[1, 0, 0, 0]] * 2)
    assert torch.all((torch.sigmoid(params["opacities"]) > 0) & (torch.sigmoid(params["opacities"]) < 1))
    assert params["colors"].shape == (2, 1)
    assert all(p.dtype == torch.float32 for p in params.values())


def test_a_checkpoint_loads_back_exactly_with_its_metadata(tmp_path):
    params = two_gaussians()
    params["quats"] = torch.tensor([[0.9, 0.1, 0.2, 0.3], [0.5, -0.5, 0.5, 0.5]])
    path = tmp_path / "sub" / "splat.pt"
    save_checkpoint(path, params, {"sun_index": 0, "steps": 10})
    loaded, meta = load_checkpoint(path)
    assert meta == {"sun_index": 0, "steps": 10}
    assert set(loaded) == set(params)
    for key in params:
        assert torch.equal(loaded[key], params[key]), key
    assert not list(path.parent.glob("*.tmp*")), "a temp file was left behind"


def test_the_browser_export_is_a_standard_3dgs_ply(tmp_path):
    params = two_gaussians()
    params["quats"] = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 4.0]])
    export_ply(tmp_path / "splat.ply", params)
    names, values = read_splat_ply(tmp_path / "splat.ply")
    # The layout 3DGS writes and Spark, SuperSplat and other viewers read.
    assert names == [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]  # fmt: skip
    col = {name: values[:, i] for i, name in enumerate(names)}
    np.testing.assert_allclose(np.stack([col["x"], col["y"], col["z"]], 1), params["means"].numpy())
    # Viewers decode colour as 0.5 + SH_C0 * f_dc, the same grey in all three channels.
    for c in ("f_dc_0", "f_dc_1", "f_dc_2"):
        np.testing.assert_allclose(0.5 + SH_C0 * col[c], [0.25, 0.6], atol=1e-6)
    # Opacity and scales stay as logit and log, as viewers expect; rotations as unit quaternions.
    np.testing.assert_allclose(col["opacity"], params["opacities"].numpy())
    np.testing.assert_allclose(col["scale_0"], np.log([2.0, 4.0]), atol=1e-6)
    np.testing.assert_allclose(np.stack([col[f"rot_{i}"] for i in range(4)], 1), [[1, 0, 0, 0], [0, 0.6, 0, 0.8]])
