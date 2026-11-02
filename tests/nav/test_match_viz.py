import numpy as np

from fidelis.nav.match_viz import INLIER_BGR, OUTLIER_BGR, draw_matches

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


def test_a_run_with_no_matches_still_gets_a_picture():
    # Failed descents are the ones most worth looking at.
    empty = np.zeros((0, 2))
    no_inliers = np.zeros(0, bool)
    canvas = draw_matches(DESCENT, MAP, empty, empty, no_inliers, caption="seed 3: no_solution")
    assert canvas.shape[:2] == (200, 550)
    assert (canvas != draw_matches(DESCENT, MAP, empty, empty, no_inliers)).any()  # the caption was drawn
