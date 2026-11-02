"""A picture of one descent's matching: the camera image beside the orbital map, matches drawn.

RANSAC's inliers in green, the matches it rejected in red, each line from a keypoint in
the descent image to its match in the map. The map is scaled to the descent image's
height. A failed descent gets a picture too: those are the ones worth looking at.

    uv run fidelis-match-viz config/jezero.yaml data/jezero 12 results/plain_mc/matches/seed_12.png

reruns seed 12 from the config and draws it.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

INLIER_BGR = (0, 200, 0)
OUTLIER_BGR = (0, 0, 230)
CAPTION_BGR = (0, 255, 255)


def _grey_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def draw_matches(
    image: np.ndarray,
    map_image: np.ndarray,
    image_points: np.ndarray,
    map_points: np.ndarray,
    inliers: np.ndarray,
    caption: str = "",
) -> np.ndarray:
    """A BGR uint8 canvas. Images are single-channel in [0, 1]; points are OpenCV pixels (x, y)."""
    height, width = image.shape
    scale = height / map_image.shape[0]
    map_size = (round(map_image.shape[1] * scale), height)
    canvas = np.hstack([_grey_bgr(image), _grey_bgr(cv2.resize(map_image, map_size, interpolation=cv2.INTER_AREA))])
    ends = np.asarray(map_points, float) * scale + [width, 0]
    # Outliers first, so inliers stay visible on top.
    for keep, colour in ((~np.asarray(inliers, bool), OUTLIER_BGR), (np.asarray(inliers, bool), INLIER_BGR)):
        for a, b in zip(np.asarray(image_points, float)[keep], ends[keep], strict=True):
            a, b = tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b)
            cv2.line(canvas, a, b, colour, 1, cv2.LINE_8)
            cv2.circle(canvas, a, 2, colour, -1)
            cv2.circle(canvas, b, 2, colour, -1)
    if caption:
        cv2.putText(canvas, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CAPTION_BGR, 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("data_dir", type=Path, help="the workflow's scene data, e.g. data/jezero")
    parser.add_argument("seed", type=int)
    parser.add_argument("out", type=Path, help="a .png")
    parser.add_argument("--fidelity", default="mesh")
    args = parser.parse_args()

    from fidelis.scenario import draw, load_config
    from fidelis.simulation import build_mesh_simulator

    simulator = build_mesh_simulator(args.config, args.data_dir)
    run = simulator.run(draw(load_config(args.config), args.seed), args.fidelity)
    r = run.record
    error = "no fix" if r.error_m is None else f"error {r.error_m:.1f} m"
    caption = f"seed {r.seed}, {r.fidelity}: {r.status}, {r.inlier_count}/{r.match_count} inliers, {error}"
    canvas = draw_matches(
        run.image, simulator.navigator.map.image, run.nav.image_points, run.nav.map_points, run.nav.inliers, caption
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.stem + ".tmp" + args.out.suffix)
    if not cv2.imwrite(str(tmp), canvas):
        raise OSError(f"could not write {tmp}")
    tmp.replace(args.out)
    print(f"{args.out}: {caption}")
    return 0
