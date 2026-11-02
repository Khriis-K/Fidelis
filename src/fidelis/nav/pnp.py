"""Camera pose from 2D-3D correspondences: OpenCV PnP inside RANSAC.

The solve fails softly. Too few correspondences, or no pose that enough of them agree
on, returns a PnpSolution with a failure status instead of raising, so a Monte Carlo
batch records the failed descent and carries on.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from fidelis.render.camera import CameraPose, Intrinsics

# A fix needs this many correspondences, and this many RANSAC inliers. Twice PnP-RANSAC's
# 5-point minimal sample: random matches rarely agree on a pose beyond the sample itself
# (a stray point lands within 8 px of its projection with odds of about 2e-4), so 10
# inliers mean real agreement, not chance.
MIN_CORRESPONDENCES = 10
# OpenCV's default. At 2.2 km a descent pixel is 4.3 m, so 8 px is 34 m on the ground,
# about 6 CTX map pixels: room for the map keypoints' error, and tight against a wrong
# match, which lands anywhere in a 4.4 km view.
REPROJECTION_PX = 8.0
RANSAC_ITERATIONS = 1000
RANSAC_CONFIDENCE = 0.999


@dataclass(frozen=True)
class PnpSolution:
    status: str  # "ok", "too_few_matches" or "no_solution"
    pose: CameraPose | None  # None unless status is "ok"
    inliers: np.ndarray  # one bool per correspondence


def solve_pose(image_points: np.ndarray, world_points: np.ndarray, intrinsics: Intrinsics) -> PnpSolution:
    """Fit the camera pose to N image pixels and their N world points (metres)."""
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    world_points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
    n = len(image_points)
    if len(world_points) != n:
        raise ValueError(f"need the same number of image and world points, got {n} and {len(world_points)}")
    none = np.zeros(n, dtype=bool)
    if n < MIN_CORRESPONDENCES:
        return PnpSolution("too_few_matches", None, none)

    # RANSAC's sampler is seeded inside OpenCV, so the same input gives the same pose.
    found, rvec, tvec, index = cv2.solvePnPRansac(
        world_points,
        image_points,
        intrinsics.matrix(),
        None,
        iterationsCount=RANSAC_ITERATIONS,
        reprojectionError=REPROJECTION_PX,
        confidence=RANSAC_CONFIDENCE,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not found or index is None or len(index) < MIN_CORRESPONDENCES:
        return PnpSolution("no_solution", None, none)
    inliers = none.copy()
    inliers[index.ravel()] = True
    # OpenCV's (rvec, tvec) is world-to-camera; CameraPose is camera-to-world.
    r = cv2.Rodrigues(rvec)[0]
    pose = CameraPose(rotation=r.T, position=-r.T @ tvec.ravel())
    return PnpSolution("ok", pose, inliers)
