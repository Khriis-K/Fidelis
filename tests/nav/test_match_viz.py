from pathlib import Path

import numpy as np
import pytest

from fidelis.nav.match_viz import INLIER_BGR, KEYPOINT_BGR, OUTLIER_BGR, draw_matches

DESCENT = np.full((200, 300), 0.5, np.float32)
MAP = np.full((400, 500), 0.5, np.float32)  # scaled by 0.5 to the descent image's height


def pair(image_xy, map_xy, inlier):
    return np.array([image_xy], float), np.array([map_xy], float), np.array([inlier])


def midpoint(image_xy, map_xy):
    """Where the line between a matched pair crosses, in the side-by-side canvas."""
    canvas_map = (300 + 0.5 * map_xy[0], 0.5 * map_xy[1])
    return round((image_xy[0] + canvas_map[0]) / 2), round((image_xy[1] + canvas_map[1]) / 2)


def test_the_canvas_is_the_descent_image_beside_the_map_scaled_to_its_height():
    canvas = draw_matches(DESCENT, MAP, *pair((10, 10), (20, 20), True))
    assert canvas.shape[:2] == (200, 300 + 250)
    assert canvas.shape[2] == 3 and canvas.dtype == np.uint8


def test_an_inlier_and_an_outlier_are_drawn_in_their_own_colours():
    for inlier, colour in ((True, INLIER_BGR), (False, OUTLIER_BGR)):
        image_xy, map_xy = (60.0, 100.0), (200.0, 160.0)
        canvas = draw_matches(DESCENT, MAP, *pair(image_xy, map_xy, inlier))
        x, y = midpoint(image_xy, map_xy)
        assert tuple(canvas[y, x]) == colour


def test_the_inlier_and_outlier_colours_differ():
    assert INLIER_BGR != OUTLIER_BGR


def test_keypoints_that_found_no_match_are_drawn_on_both_sides():
    empty = np.zeros((0, 2))
    canvas = draw_matches(
        DESCENT,
        MAP,
        empty,
        empty,
        np.zeros(0, bool),
        image_keypoints=np.array([[40.0, 50.0]]),
        map_keypoints=np.array([[100.0, 300.0]]),
    )
    assert tuple(canvas[50, 40]) == KEYPOINT_BGR
    assert tuple(canvas[150, 300 + 50]) == KEYPOINT_BGR  # map pixel (100, 300) at half scale
    assert KEYPOINT_BGR not in (INLIER_BGR, OUTLIER_BGR)


def test_a_run_with_no_matches_still_gets_a_picture():
    # Failed descents are the ones most worth looking at.
    empty = np.zeros((0, 2))
    no_inliers = np.zeros(0, bool)
    canvas = draw_matches(DESCENT, MAP, empty, empty, no_inliers, caption="seed 3: no_solution")
    assert canvas.shape[:2] == (200, 550)
    assert (canvas != draw_matches(DESCENT, MAP, empty, empty, no_inliers)).any()  # the caption was drawn


class StopBeforeRunning(Exception):
    pass


@pytest.mark.parametrize(
    ("flags", "zeroed"),
    [
        ([], set()),
        (["--clear-air"], {"dust_tau"}),
        (["--exact-camera"], {"focal_error"}),
    ],
)
def test_the_cli_pictures_the_descent_with_exactly_the_errors_its_flags_leave_out(monkeypatch, tmp_path, flags, zeroed):
    # The picture must show the run it reproduces; those from before #37 and #38 had no haze and the exact camera.
    import fidelis.nav.match_viz as match_viz
    import fidelis.simulation

    seen = []

    class Simulator:
        def run(self, scenario, fidelity):
            seen.append(scenario)
            raise StopBeforeRunning

    monkeypatch.setattr(fidelis.simulation, "build_simulator", lambda *args, **kwargs: Simulator())
    config_path = str(Path(__file__).resolve().parents[2] / "config" / "jezero.yaml")
    monkeypatch.setattr("sys.argv", ["fidelis-match-viz", config_path, "data", "7", str(tmp_path / "m.png"), *flags])
    with pytest.raises(StopBeforeRunning):
        match_viz.main()
    [scenario] = seen
    for name in ("dust_tau", "focal_error"):
        assert (getattr(scenario, name) == 0.0) == (name in zeroed), name
