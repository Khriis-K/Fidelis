import math

import cv2
import mitsuba as mi
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fidelis.render.camera import Intrinsics, look_at
from fidelis.render.mesh_render import RENDER_DETERMINISM_TOLERANCE, Sun, render, terrain_scene
from fidelis.terrain.mesh import build_terrain_mesh

MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"
LEFT, TOP = 4589870.0, 1094306.0
SIZE = 40  # 40 x 40 m flat tile at 1 m/px; the mesh's local frame spans -20..20 m
PLATEAU_H = 6.0
PLATEAU_EDGE = 2.5  # the plateau's flat top ends 2.5 m from the centre on every side
ALBEDO = 128 / 255


@pytest.fixture(autouse=True)
def variant():
    mi.set_variant("scalar_rgb")


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
def plateau_mesh(tmp_path):
    """Flat ground at z = 0 with a 6 m tall, 6 x 6 px plateau in the middle, uniform albedo."""
    heights = np.zeros((SIZE, SIZE), dtype=np.float32)
    heights[17:23, 17:23] = PLATEAU_H  # pixel centres at -2.5..2.5 m
    write_raster(tmp_path / "dtm.tif", heights, 1.0)
    write_raster(tmp_path / "ortho.tif", np.full((SIZE * 2, SIZE * 2), 128, dtype=np.uint8), 0.5)
    build_terrain_mesh(tmp_path / "dtm.tif", tmp_path / "ortho.tif", tmp_path / "mesh")
    return tmp_path / "mesh"


# Nadir camera 100 m up, north at the top, framing the 40 m tile at 0.25 m/px on the ground.
NADIR = look_at(eye=[0.0, 0.0, 100.0], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
NADIR_K = Intrinsics.from_fov(width=160, height=160, fov_x_deg=math.degrees(2 * math.atan(20 / 100)))


def project(points, pose, k):
    """Pixel coordinates (OpenCV convention) of world points, by OpenCV itself."""
    w2c = pose.world_to_camera()
    rvec, _ = cv2.Rodrigues(w2c[:3, :3])
    px, _ = cv2.projectPoints(np.asarray(points, dtype=np.float64), rvec, w2c[:3, 3], k.matrix(), None)
    return px.reshape(-1, 2)


def ground_radiance(image, x, y):
    """Mean radiance in a 3 x 3 pixel window around the ground point (x, y, 0)."""
    ((u, v),) = project([[x, y, 0.0]], NADIR, NADIR_K)
    c, r = round(u), round(v)
    return image[r - 1 : r + 2, c - 1 : c + 2].mean()


@pytest.mark.parametrize(
    ("point", "radius"),
    [([5.0, 10.0, -3.0], 2.0), ([13.0, 13.0, -2.0], 2.0), ([-1.0, 5.0, -1.0], 2.0)],
)
def test_a_point_renders_where_opencv_projects_it(point, radius):
    # The pose and intrinsics mean the same thing to the renderer as to the navigator's
    # PnP: a small glowing sphere's image centroid lands on cv2.projectPoints of its
    # centre. The camera is oblique and rolled, and its principal point is off-centre in
    # both axes, so a flipped axis, a half-pixel shift or a swapped cx/cy all fail.
    pose = look_at(eye=[30.0, -40.0, 120.0], target=[5.0, 10.0, -3.0], up=[1.0, 1.0, 0.0])
    k = Intrinsics(width=96, height=80, fx=70.0, fy=70.0, cx=40.3, cy=45.8)
    scene = mi.load_dict(
        {
            "type": "scene",
            "ball": {"type": "sphere", "center": point, "radius": radius, "emitter": {"type": "area"}},
        }
    )
    (expected,) = project([point], pose, k)
    assert 5 < expected[0] < k.width - 5 and 5 < expected[1] < k.height - 5

    image = render(scene, pose, k, spp=512)

    assert image.shape == (k.height, k.width)
    rows, cols = np.mgrid[: k.height, : k.width]
    centroid = np.array([(image * cols).sum(), (image * rows).sum()]) / image.sum()
    np.testing.assert_allclose(centroid, expected, atol=0.1)


@pytest.mark.parametrize(
    ("azimuth", "elevation"),
    [(270.0, 45.0), (0.0, 30.0)],  # sun in the west, then in the north and lower
)
def test_the_plateau_casts_its_shadow_away_from_the_sun_with_length_height_over_tan_elevation(
    plateau_mesh, azimuth, elevation
):
    scene = terrain_scene(plateau_mesh, Sun(azimuth_deg=azimuth, elevation_deg=elevation))
    image = render(scene, NADIR, NADIR_K, spp=16)

    away = -np.array([math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth))])  # horizontal
    side = np.array([away[1], -away[0]])
    length = PLATEAU_H / math.tan(math.radians(elevation))
    lit = ALBEDO * math.sin(math.radians(elevation))

    def at(d, s=0.0):
        x, y = (PLATEAU_EDGE + d) * away + s * side
        return ground_radiance(image, x, y)

    for d in (2.0, length - 1.5):  # inside the shadow
        assert at(d) < 0.05 * lit, f"{d:.1f} m downsun of the plateau should be in shadow"
    assert at(length + 2.5) == pytest.approx(lit, rel=0.03)  # past the shadow's tip
    # The sunward side, 15 m out: nearer in, light bounced off the plateau's sunlit wall
    # brightens the ground (measured at 6 m from its centre: 5.9% under the 45 deg sun,
    # 9.8% under the 30 deg sun; under 3% beyond 12 m).
    assert at(-(PLATEAU_EDGE + 15.0)) == pytest.approx(lit, rel=0.03)
    assert at(2.0, s=6.0) == pytest.approx(lit, rel=0.03)  # beside the shadow


def test_light_bounced_off_a_sunlit_wall_brightens_the_ground_in_front_of_it(plateau_mesh):
    # Interreflection: ground 2.5 m in front of the plateau's sunlit wall also sees the
    # wall, so it's brighter than open ground (measured 5.9% at 64 spp). Without the
    # bounce it is exactly albedo * sin(elevation). Nearer than about 1.5 m, smooth
    # shading normals brighten the ground far more, bounce or not, so test beyond that.
    scene = terrain_scene(plateau_mesh, Sun(azimuth_deg=270.0, elevation_deg=45.0))
    image = render(scene, NADIR, NADIR_K, spp=64)
    lit = ALBEDO * math.sin(math.radians(45.0))

    assert ground_radiance(image, -(PLATEAU_EDGE + 1.0 + 2.5), 0.0) > 1.03 * lit


def test_the_same_seed_gives_the_same_image_and_another_seed_gives_new_noise(plateau_mesh):
    # At 4 spp the shadow edges and plateau rims are noisy, so a seed that changed
    # nothing would show up as identical images.
    scene = terrain_scene(plateau_mesh, Sun(azimuth_deg=300.0, elevation_deg=40.0))

    first = render(scene, NADIR, NADIR_K, spp=4, seed=7)
    again = render(scene, NADIR, NADIR_K, spp=4, seed=7)
    other = render(scene, NADIR, NADIR_K, spp=4, seed=8)

    assert np.abs(first - again).max() <= RENDER_DETERMINISM_TOLERANCE
    assert np.abs(first - other).max() > 10 * RENDER_DETERMINISM_TOLERANCE + 1e-3


@pytest.mark.gpu
def test_the_same_seed_gives_the_same_image_on_the_gpu_within_the_stated_tolerance(plateau_mesh):
    # The CUDA film adds samples atomically in no fixed order, so the GPU, which real
    # renders use, isn't bit-for-bit repeatable the way scalar_rgb is.
    mi.set_variant("cuda_ad_rgb")
    scene = terrain_scene(plateau_mesh, Sun(azimuth_deg=300.0, elevation_deg=40.0))

    first = render(scene, NADIR, NADIR_K, spp=64, seed=7)
    again = render(scene, NADIR, NADIR_K, spp=64, seed=7)
    other = render(scene, NADIR, NADIR_K, spp=64, seed=8)

    assert np.abs(first - again).max() <= RENDER_DETERMINISM_TOLERANCE
    assert np.abs(first - other).max() > 10 * RENDER_DETERMINISM_TOLERANCE + 1e-3


def test_sun_direction_follows_compass_azimuth_and_elevation_above_the_horizon():
    np.testing.assert_allclose(Sun(azimuth_deg=90.0, elevation_deg=0.0).direction(), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(Sun(azimuth_deg=0.0, elevation_deg=90.0).direction(), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(
        Sun(azimuth_deg=180.0, elevation_deg=30.0).direction(), [0, -math.sqrt(3) / 2, 0.5], atol=1e-12
    )


def test_a_sun_below_the_horizon_is_rejected():
    with pytest.raises(ValueError, match="elevation"):
        Sun(azimuth_deg=0.0, elevation_deg=-5.0)


def test_non_square_pixels_are_rejected(plateau_mesh):
    # Mitsuba's perspective camera has one field of view and square pixels.
    scene = terrain_scene(plateau_mesh, Sun(azimuth_deg=0.0, elevation_deg=45.0))
    k = Intrinsics(width=64, height=64, fx=50.0, fy=55.0, cx=31.5, cy=31.5)

    with pytest.raises(ValueError, match="square"):
        render(scene, NADIR, k, spp=1)
