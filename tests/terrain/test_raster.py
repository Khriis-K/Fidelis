import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fidelis.terrain.raster import bilinear, centres, read_raster, write_raster

MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"
LEFT, TOP = 4589000.0, 1094000.0


def test_centres_are_half_a_pixel_in_with_rows_running_south():
    mx, my = centres(from_origin(LEFT, TOP, 20.0, 20.0), (2, 3))

    np.testing.assert_array_equal(mx, [[LEFT + 10, LEFT + 30, LEFT + 50]] * 2)
    np.testing.assert_array_equal(my, [[TOP - 10] * 3, [TOP - 30] * 3])


def test_bilinear_hits_every_pixel_value_at_its_centre():
    z = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = from_origin(LEFT, TOP, 6.0, 6.0)

    np.testing.assert_allclose(bilinear(z, t, *centres(t, z.shape)), z)


def test_bilinear_extends_the_edge_cell_past_the_outermost_centres():
    # A ramp along x: half a pixel past the last centre continues the ramp.
    z = np.tile(np.arange(4, dtype=np.float64), (3, 1))
    t = from_origin(LEFT, TOP, 1.0, 1.0)

    assert bilinear(z, t, np.array([LEFT + 4.0]), np.array([TOP - 1.5]))[0] == pytest.approx(3.5)


def test_write_raster_takes_its_shape_from_the_data_and_drops_tiling(tmp_path):
    profile = {
        "driver": "GTiff",
        "width": 999,
        "height": 999,
        "count": 1,
        "dtype": "uint8",
        "crs": MARS_EQC,
        "transform": from_origin(LEFT, TOP, 6.0, 6.0),
        "blockxsize": 256,
        "blockysize": 256,
        "tiled": False,
    }
    data = np.arange(15, dtype=np.uint8).reshape(3, 5)

    write_raster(tmp_path / "sub" / "out.tif", data, profile)

    back, transform, written = read_raster(tmp_path / "sub" / "out.tif")
    np.testing.assert_array_equal(back, data)
    assert transform == profile["transform"]
    assert (written["width"], written["height"]) == (5, 3)
    assert not list(tmp_path.glob("sub/*.tmp.tif"))
    with rasterio.open(tmp_path / "sub" / "out.tif") as src:
        assert src.crs.to_proj4().startswith("+proj=eqc")
