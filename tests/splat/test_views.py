"""Training and held-out views for a splat scene: poses the descent camera could really take,
on seeds no Monte Carlo run uses, rendered and stored with everything needed to reuse them."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from fidelis.render.camera import Intrinsics
from fidelis.scenario import draw, load_config
from fidelis.simulation import scenario_pose
from fidelis.splat.views import MC_SEED_LIMIT, load_views, render_views, view_poses, view_seeds, write_views

REPO = Path(__file__).resolve().parents[2]
CONFIG = load_config(REPO / "config" / "jezero.yaml")
K = Intrinsics.from_fov(32, 24, 90)


def bumpy_ground(east, north):
    return -2500.0 + 30.0 * math.sin(east / 700.0) * math.cos(north / 900.0)


def test_training_and_held_out_seeds_never_overlap_each_other_or_monte_carlo_seeds():
    train, heldout = set(view_seeds("train", 500)), set(view_seeds("heldout", 50))
    assert len(train) == 500 and len(heldout) == 50
    assert not train & heldout
    assert min(train | heldout) >= MC_SEED_LIMIT


def test_seed_ranges_are_fixed_so_a_rerun_renders_the_same_views():
    assert list(view_seeds("train", 10)) == list(view_seeds("train", 10))
    assert list(view_seeds("train", 5)) == list(view_seeds("train", 10))[:5]


def test_an_unknown_split_is_refused():
    with pytest.raises(ValueError, match="split"):
        view_seeds("test", 3)


def test_each_view_is_the_true_camera_its_seeds_scenario_describes():
    seeds = view_seeds("train", 20)
    poses = view_poses(CONFIG, seeds, bumpy_ground)
    for seed, pose in zip(seeds, poses, strict=True):
        s = draw(CONFIG, seed)
        expected = scenario_pose(s, bumpy_ground(s.offset_east_m, s.offset_north_m))
        np.testing.assert_allclose(pose.rotation, expected.rotation)
        np.testing.assert_allclose(pose.position, expected.position)


def test_views_spread_over_the_scenario_range_not_one_spot():
    poses = view_poses(CONFIG, view_seeds("train", 200), bumpy_ground)
    east = np.array([p.position[0] for p in poses])
    north = np.array([p.position[1] for p in poses])
    # The offsets' sd is 1100 m (config): 200 draws spread well past a kilometre each way.
    assert east.std() > 600 and north.std() > 600


def test_render_views_stacks_each_poses_render_under_the_one_sun():
    calls = []

    def renderer(pose, intrinsics, sun_index, seed):
        calls.append((sun_index, seed))
        return np.full((intrinsics.height, intrinsics.width), pose.position[0], np.float32)

    seeds = view_seeds("heldout", 3)
    poses = view_poses(CONFIG, seeds, bumpy_ground)
    images = render_views(renderer, poses, K, sun_index=2, seeds=seeds)
    assert images.shape == (3, K.height, K.width)
    assert [c[0] for c in calls] == [2, 2, 2]
    # Each view's render noise comes from its own seed.
    assert [c[1] for c in calls] == list(seeds)
    np.testing.assert_allclose(images[:, 0, 0], [p.position[0] for p in poses], rtol=1e-3)


def test_written_views_load_back_with_their_poses_intrinsics_and_sun(tmp_path):
    rng = np.random.default_rng(0)
    splits = {}
    for split, n in (("train", 4), ("heldout", 2)):
        seeds = view_seeds(split, n)
        splits[split] = (seeds, view_poses(CONFIG, seeds, bumpy_ground), rng.random((n, K.height, K.width)))
    write_views(tmp_path, splits, K, sun_index=1, meta={"spp": 16})

    for split, (seeds, poses, images) in splits.items():
        views = load_views(tmp_path, split)
        assert views.seeds == list(seeds)
        assert views.intrinsics == K
        assert views.sun_index == 1
        # Stored as float16: about 3 decimal digits, far finer than the camera's noise.
        np.testing.assert_allclose(views.images, images, atol=1e-3)
        for loaded, pose in zip(views.poses, poses, strict=True):
            np.testing.assert_allclose(loaded.rotation, pose.rotation, atol=1e-12)
            np.testing.assert_allclose(loaded.position, pose.position, atol=1e-9)
    assert json.loads((tmp_path / "views.json").read_text())["spp"] == 16
    assert not list(tmp_path.glob("*.tmp*")), "a temp file was left behind"
