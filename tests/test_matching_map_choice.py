"""Which map the navigator matches against (#36), and the offset, elevation and truth that go with it.

The real LVS map (CTX) is the default. The old map, a block average of the orthoimage the
scene renders, stays selectable so earlier results reproduce; its offset is zero by construction.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from test_simulation import FLAT_MAP, GROUND_Z, K, render_flat, scenario

from fidelis.map_frame import MapOffset, load_map_offset, matching_map_source
from fidelis.nav.navigator import Navigator
from fidelis.simulation import Simulator, build_simulator, load_matching_map

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "jezero.yaml"
MARS_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +R=3396190 +units=m +no_defs"
LEFT, TOP = 4589000.0, 1094000.0
ORIGIN = (LEFT + 384.0, TOP - 384.0)
CTX_HEIGHT, SCENE_HEIGHT = -2500.0, -2493.0


def jezero():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# --- the config: the CTX map is live, with its measured offset -----------------------------


def test_the_jezero_navigator_matches_against_the_ctx_map():
    assert matching_map_source(jezero()) == "ctx"


def test_the_live_offset_is_the_one_measured_between_the_ctx_map_and_the_scene():
    # #40's measurement, in ground metres: docs/benchmarks/map-offset.md.
    offset = load_map_offset(jezero())
    assert (offset.east_m, offset.north_m) == (1.89, -1.49)
    assert "docs/benchmarks/map-offset.md" in offset.source


def test_the_live_offset_cannot_drift_from_the_recorded_measurement():
    scene = jezero()
    measured, live = scene["ctx_map_offset"], scene["map_offset"]
    assert (live["east_m"], live["north_m"]) == (measured["east_m"], measured["north_m"])


def test_the_old_scene_map_has_a_zero_offset_whatever_the_config_holds_for_ctx():
    # The scene's own map shares the scene's frame exactly; CTX's 1.89 m must not follow it.
    scene = jezero() | {"matching_map": {"source": "scene"}}
    offset = load_map_offset(scene)
    assert (offset.east_m, offset.north_m) == (0.0, 0.0)
    assert "construction" in offset.source


def test_a_run_can_ask_for_the_old_map_and_gets_its_offset_with_it():
    offset = load_map_offset(jezero(), source="scene")
    assert (offset.east_m, offset.north_m) == (0.0, 0.0)


def test_an_unknown_map_name_is_refused():
    with pytest.raises(ValueError, match="hirise"):
        matching_map_source({"matching_map": {"source": "hirise"}})
    with pytest.raises(ValueError, match="hirise"):
        load_map_offset(jezero(), source="hirise")


# --- loading the chosen map from the workflow's folders ------------------------------------


def write_raster(path, data, res, left=LEFT, top=TOP):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def textured(seed):
    return np.random.default_rng(seed).integers(1, 255, (128, 128)).astype(np.uint8)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """A small stand-in for data/jezero: both maps, and two DTMs 7 m apart."""
    data = tmp_path_factory.mktemp("jezero")
    write_raster(data / "lvs_map" / "map.tif", textured(1), 6.0)
    write_raster(data / "matching_map" / "map.tif", textured(2), 6.0)
    write_raster(data / "scene" / "ring_dtm.tif", np.full((40, 40), CTX_HEIGHT, np.float32), 20.0)
    write_raster(data / "scene" / "dtm.tif", np.full((192, 192), SCENE_HEIGHT, np.float32), 4.0)
    (data / "mesh").mkdir()
    frame = {"origin": list(ORIGIN), "east_scale": 0.95, "ring": {"dtm": "../scene/ring_dtm.tif"}}
    (data / "mesh" / "mesh.json").write_text(json.dumps(frame))
    return data


def test_the_ctx_choice_loads_the_lvs_map_with_ctx_elevation(data_dir):
    loaded = load_matching_map(data_dir, "ctx")

    np.testing.assert_allclose(loaded.image, textured(1) / 255, atol=1e-7)
    assert loaded.pixel_to_world(np.array([[64.0, 64.0]]))[0, 2] == pytest.approx(CTX_HEIGHT)


def test_the_scene_choice_loads_the_old_map_with_the_scenes_elevation(data_dir):
    loaded = load_matching_map(data_dir, "scene")

    np.testing.assert_allclose(loaded.image, textured(2) / 255, atol=1e-7)
    assert loaded.pixel_to_world(np.array([[64.0, 64.0]]))[0, 2] == pytest.approx(SCENE_HEIGHT)


def test_the_simulator_is_built_on_the_configs_map_with_its_offset(data_dir):
    sim = build_simulator(CONFIG, data_dir, fidelities=(), device="cpu")

    assert sim.matching_map == "ctx"
    assert (sim.map_offset.east_m, sim.map_offset.north_m) == (1.89, -1.49)
    np.testing.assert_allclose(sim.navigator.map.image, textured(1) / 255, atol=1e-7)


def test_the_simulator_can_be_built_on_the_old_map_so_earlier_results_reproduce(data_dir):
    sim = build_simulator(CONFIG, data_dir, fidelities=(), device="cpu", matching_map="scene")

    assert sim.matching_map == "scene"
    assert (sim.map_offset.east_m, sim.map_offset.north_m) == (0.0, 0.0)
    np.testing.assert_allclose(sim.navigator.map.image, textured(2) / 255, atol=1e-7)


@pytest.mark.parametrize("choice", ["ctx", "scene"])
def test_the_true_ground_is_the_scenes_whichever_map_the_navigator_holds(data_dir, choice):
    # The truth can't depend on the navigator's map: the camera flies over the rendered terrain.
    sim = build_simulator(CONFIG, data_dir, fidelities=(), device="cpu", matching_map=choice)

    assert sim.ground_height(0.0, 0.0) == pytest.approx(SCENE_HEIGHT)


# --- the simulator: truth from the scene, the map named on every record --------------------


@pytest.fixture(scope="module")
def navigator():
    return Navigator(FLAT_MAP, device="cpu")


def test_the_camera_flies_above_the_scenes_ground_not_the_maps(navigator):
    # The map's DTM says GROUND_Z; the scene's ground is 9 m higher there.
    sim = Simulator({"flat": render_flat}, navigator, K, ground_height=lambda east, north: GROUND_Z + 9.0)

    record = sim.run(scenario(altitude_m=192.0), "flat").record

    assert record.true_position[2] == pytest.approx(GROUND_Z + 9.0 + 192.0)


def test_without_a_scene_terrain_the_maps_ground_is_the_truth(navigator):
    record = Simulator({"flat": render_flat}, navigator, K).run(scenario(altitude_m=192.0), "flat").record

    assert record.true_position[2] == pytest.approx(GROUND_Z + 192.0)


def test_every_record_names_the_map_it_was_matched_against(navigator):
    offset = MapOffset(1.89, -1.49, "test")
    sim = Simulator({"flat": render_flat}, navigator, K, map_offset=offset, matching_map="ctx")

    assert sim.run(scenario(), "flat").record.matching_map == "ctx"


def test_a_record_from_before_the_ctx_map_still_loads_and_names_no_map(navigator):
    from dataclasses import asdict

    from fidelis.simulation import SimulationRecord

    old = asdict(Simulator({"flat": render_flat}, navigator, K).run(scenario(), "flat").record)
    del old["matching_map"]

    assert SimulationRecord(**old).matching_map is None
