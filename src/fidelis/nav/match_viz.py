"""A picture of one descent's matching: the camera image beside the orbital map, matches drawn.

RANSAC's inliers in green, the matches it rejected in red, each line from a keypoint in
the descent image to its match in the map; keypoints that found no match are blue dots.
The map is scaled to the descent image's height. A failed descent gets a picture too:
those are the ones worth looking at.

    uv run fidelis-match-viz config/jezero.yaml data/jezero 12 results/plain_mc/matches/seed_12.png

reruns seed 12 from the config and draws it. On the GPU the rerun's error can differ from
the seed's record by up to the navigator's jitter (docs/adr/0001-gpu-reproducibility-is-image-level.md).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

INLIER_BGR = (0, 200, 0)
OUTLIER_BGR = (0, 0, 230)
KEYPOINT_BGR = (255, 160, 0)  # keypoints with no match
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
    image_keypoints: np.ndarray | None = None,
    map_keypoints: np.ndarray | None = None,
) -> np.ndarray:
    """A BGR uint8 canvas. Images are single-channel in [0, 1]; points are OpenCV pixels (x, y).

    `image_keypoints` and `map_keypoints`, if given, are every keypoint on each side, matched
    or not; they're drawn first, as single pixels, so matches sit on top.
    """
    height, width = image.shape
    scale = height / map_image.shape[0]
    map_size = (round(map_image.shape[1] * scale), height)
    canvas = np.hstack([_grey_bgr(image), _grey_bgr(cv2.resize(map_image, map_size, interpolation=cv2.INTER_AREA))])
    for points, offset, point_scale in ((image_keypoints, 0, 1.0), (map_keypoints, width, scale)):
        if points is not None and len(points):
            cols, rows = np.round(np.asarray(points, float) * point_scale + [offset, 0]).astype(int).T
            keep = (cols >= 0) & (cols < canvas.shape[1]) & (rows >= 0) & (rows < height)
            canvas[rows[keep], cols[keep]] = KEYPOINT_BGR
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
    record, nav = run.record, run.nav
    error = "no fix" if record.error_m is None else f"error {record.error_m:.1f} m"
    caption = (
        f"seed {record.seed}, {record.fidelity}: {record.status}, {record.inlier_count}/{record.match_count}"
        f" inliers, {error}; config {(record.config_sha256 or 'unknown')[:8]}"
    )
    canvas = draw_matches(
        run.image,
        simulator.navigator.map.image,
        nav.image_points,
        nav.map_points,
        nav.inliers,
        caption,
        image_keypoints=nav.keypoints,
        map_keypoints=simulator.navigator.map_keypoints,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.stem + ".tmp" + args.out.suffix)
    if not cv2.imwrite(str(tmp), canvas):
        raise OSError(f"could not write {tmp}")
    tmp.replace(args.out)
    print(f"{args.out}: {caption}")
    return 0
