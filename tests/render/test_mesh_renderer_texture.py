"""The mesh renderer renders from the texture it's given (#43: the de-shaded texture, #19)."""

import math

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fidelis.render.camera import Intrinsics, look_at
from fidelis.render.mesh_render import MeshRenderer, Sun
from fidelis.terrain.mesh import build_terrain_mesh

MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"
SIZE = 40
NADIR = look_at(eye=[0.0, 0.0, 100.0], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
K = Intrinsics.from_fov(width=64, height=64, fov_x_deg=math.degrees(2 * math.atan(10 / 100)))


def write_raster(path, data, res):
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": data.dtype,
        "crs": MARS_EQC,
        "transform": from_origin(4589870.0, 1094306.0, res, res),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


@pytest.fixture
def flat_mesh(tmp_path):
    """Flat ground of albedo 128/255, with a second texture beside it at half that."""
    write_raster(tmp_path / "dtm.tif", np.zeros((SIZE, SIZE), np.float32), 1.0)
    write_raster(tmp_path / "ortho.tif", np.full((SIZE * 2, SIZE * 2), 128, np.uint8), 0.5)
    mesh = tmp_path / "mesh"
    build_terrain_mesh(tmp_path / "dtm.tif", tmp_path / "ortho.tif", mesh, east_scale=1.0)
    photo = cv2.imread(str(mesh / "texture.png"), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(mesh / "texture_half.png"), (photo // 2).astype(photo.dtype))
    return mesh


@pytest.mark.gpu
def test_the_renderer_renders_the_texture_it_is_given(flat_mesh):
    zenith = [Sun(0.0, 90.0)]
    photo = MeshRenderer(flat_mesh, zenith, spp=16)(NADIR, K, 0, seed=1)
    half = MeshRenderer(flat_mesh, zenith, spp=16, texture="texture_half.png")(NADIR, K, 0, seed=1)
    # Zenith sun on flat ground: radiance is the albedo, so half the texture is half the light.
    assert np.median(photo) == pytest.approx(128 / 255, rel=0.05)
    assert np.median(half) == pytest.approx(np.median(photo) / 2, rel=0.05)
