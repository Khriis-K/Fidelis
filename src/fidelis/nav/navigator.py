"""The system under test: ALIKED keypoints, LightGlue matches to the orbital map, PnP for position.

`Navigator(orbital_map)` extracts the map's features once; `locate(image, intrinsics)`
then matches one descent image against them and solves the camera pose. The result
carries the matched pixel pairs and RANSAC's inlier mask, so a failed descent can be
drawn and replayed, not just counted.

Two things LightGlue's defaults would get wrong here:

- ALIKED resizes every image to 1024 px on its long side unless told not to. That
  would quietly coarsen the 1670 px CTX-like map from 6 m to about 10 m per pixel, so
  both images are extracted at their own resolution (`resize=None`).
- ALIKED's dense features for a whole map don't fit on an 8 GB GPU (a 5010 px map
  needs about 13 GB for one 128-channel layer), so the map is extracted in tiles.

Weights: ALIKED (BSD-3-Clause) and LightGlue's ALIKED model (Apache-2.0), both
permissive. LightGlue's other detector option has non-commercial weights; a test keeps
it out of the package by name.

The descent image is single-channel, in [0, 1] (the camera's output scaled to full
scale). The navigator doesn't rescale it, so exposure and clipping stay what the camera made.

ALIKED and LightGlue aren't rotation-invariant: a view turned 90 degrees from north-up
doesn't match the map at all. Like LVS, which knew its attitude from the IMU, `locate`
takes the attitude the spacecraft believes it has and warps the image to a nadir,
north-up view before extracting features. Keypoints go back to the camera's own pixels
before PnP, so the warp never reaches the pose or the returned matches.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from lightglue import ALIKED, LightGlue

from fidelis.nav.orbital_map import OrbitalMap
from fidelis.nav.pnp import solve_pose
from fidelis.render.camera import CameraPose, Intrinsics

# Per descent image and per map tile. On the 6 m map a full 1024 px tile reaches the cap
# (the whole 1670 px map gives 14,290 keypoints in four tiles); the descent images
# measured here matched 1,900-3,500 of them (docs/benchmarks/navigator.md).
MAX_KEYPOINTS = 4096
# A whole sweep, its 2 m map in 25 of these tiles, peaked at 3.8 GB of torch memory.
MAP_TILE_PX = 1024
# Keypoints this close to the warped image's empty border are dropped: the border's
# straight edge is exactly what a detector fires on, and it isn't in the map.
BORDER_PX = 8
# Camera-to-world rotation of a nadir camera whose image top points north (look_at's convention).
NADIR_NORTH_UP = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])


def nadir_homography(attitude: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """The 3 x 3 map from a camera's pixels to those of a nadir, north-up camera at the same place.

    Only the rotation differs, so this is exact whatever the terrain: K R_nadirᵀ R K⁻¹.
    """
    k = intrinsics.matrix()
    return k @ NADIR_NORTH_UP.T @ np.asarray(attitude, dtype=np.float64) @ np.linalg.inv(k)


def warp_to_nadir(image: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The image as a nadir, north-up camera would see it, on a canvas just big enough for all of it.

    Returns the warped image, the homography from its pixels back to the original's,
    and a mask of where it holds image content at least BORDER_PX from the empty part.
    """
    height, width = image.shape
    right, bottom = width - 0.5, height - 0.5
    corners = np.array([[-0.5, -0.5, 1], [right, -0.5, 1], [right, bottom, 1], [-0.5, bottom, 1]])
    projected = corners @ homography.T
    if (projected[:, 2] <= 0).any():
        raise ValueError("the view reaches the horizon, so it has no nadir image to warp to")
    # Rounded so float fuzz can't add a row: a 90-degree turn of a 256 px image stays 256 px.
    warped_corners = np.round(projected[:, :2] / projected[:, 2:], 6)
    low = np.floor(warped_corners.min(axis=0) + 0.5)
    size = tuple(int(s) for s in np.ceil(warped_corners.max(axis=0) + 0.5 - low))
    to_canvas = np.array([[1, 0, -low[0]], [0, 1, -low[1]], [0, 0, 1]]) @ homography
    warped = cv2.warpPerspective(image.astype(np.float32), to_canvas, size, flags=cv2.INTER_LINEAR)
    covered = cv2.warpPerspective(np.ones_like(image, np.uint8), to_canvas, size, flags=cv2.INTER_NEAREST)
    # Beyond the canvas counts as covered: only the warp's empty corners are new edges.
    valid = cv2.erode(covered, np.ones((2 * BORDER_PX + 1, 2 * BORDER_PX + 1), np.uint8), borderValue=1)
    return warped, np.linalg.inv(to_canvas), valid.astype(bool)


@dataclass(frozen=True)
class NavResult:
    status: str  # "ok", "too_few_matches" or "no_solution"
    pose: CameraPose | None  # the fix; None unless status is "ok"
    image_points: np.ndarray  # N x 2 matched descent-image pixels (OpenCV convention)
    map_points: np.ndarray  # N x 2 matched map pixels, same order
    inliers: np.ndarray  # N bools, RANSAC's inlier mask

    @property
    def position(self) -> np.ndarray | None:
        return None if self.pose is None else self.pose.position

    @property
    def match_count(self) -> int:
        return len(self.image_points)

    @property
    def inlier_count(self) -> int:
        return int(self.inliers.sum())


class Navigator:
    def __init__(self, orbital_map: OrbitalMap, device: str = "cuda"):
        self.map = orbital_map
        self.device = device
        self.extractor = ALIKED(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)
        self.matcher = LightGlue(features="aliked").eval().to(device)
        self.map_features = self._extract_tiled(orbital_map.image)

    def locate(self, image: np.ndarray, intrinsics: Intrinsics, attitude: np.ndarray | None = None) -> NavResult:
        """Match one descent image to the map and solve the camera pose.

        `attitude` is the camera-to-world rotation the spacecraft believes it has (an IMU's
        estimate, errors and all). Given one, the image is warped to nadir and north-up
        before matching; without one it is matched as it is, which works only near north-up.
        """
        if image.shape != (intrinsics.height, intrinsics.width):
            raise ValueError(
                f"image is {image.shape}, intrinsics say {(intrinsics.height, intrinsics.width)} (rows, cols)"
            )
        to_image = None
        if attitude is not None:
            image, to_image, valid = warp_to_nadir(image, nadir_homography(attitude, intrinsics))
        with torch.inference_mode():
            features = self.extractor.extract(self._tensor(image), resize=None)
            if to_image is not None:
                features = self._keep_valid(features, valid)
            if features["keypoints"].shape[1] == 0:
                pairs = np.zeros((0, 2), dtype=int)
            else:
                matches = self.matcher({"image0": features, "image1": self.map_features})["matches"][0]
                pairs = matches.cpu().numpy()
        image_points = features["keypoints"][0].cpu().numpy().astype(np.float64)[pairs[:, 0]]
        if to_image is not None and len(image_points):
            image_points = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), to_image).reshape(-1, 2)
        map_points = self.map_features["keypoints"][0].cpu().numpy().astype(np.float64)[pairs[:, 1]]

        solution = solve_pose(image_points, self.map.pixel_to_world(map_points), intrinsics)
        return NavResult(solution.status, solution.pose, image_points, map_points, solution.inliers)

    def _keep_valid(self, features: dict, valid: np.ndarray) -> dict:
        """Only the keypoints that land where `valid` is true."""
        cols, rows = features["keypoints"][0].round().long().cpu().numpy().T
        rows, cols = rows.clip(0, valid.shape[0] - 1), cols.clip(0, valid.shape[1] - 1)
        keep = torch.from_numpy(valid[rows, cols]).to(self.device)
        return {**features, **{k: features[k][:, keep] for k in ("keypoints", "descriptors", "keypoint_scores")}}

    def _tensor(self, gray: np.ndarray) -> torch.Tensor:
        # ALIKED takes RGB; a grey image is three equal channels.
        return torch.from_numpy(np.ascontiguousarray(gray, dtype=np.float32)).to(self.device).expand(3, -1, -1)

    def _extract_tiled(self, image: np.ndarray) -> dict:
        """ALIKED features of the whole map, one tile at a time, in whole-map pixel coordinates."""
        keypoints, descriptors, scores = [], [], []
        with torch.inference_mode():
            for r0 in range(0, image.shape[0], MAP_TILE_PX):
                for c0 in range(0, image.shape[1], MAP_TILE_PX):
                    tile = image[r0 : r0 + MAP_TILE_PX, c0 : c0 + MAP_TILE_PX]
                    f = self.extractor.extract(self._tensor(tile), resize=None)
                    keypoints.append(f["keypoints"][0] + torch.tensor([c0, r0], device=self.device))
                    descriptors.append(f["descriptors"][0])
                    scores.append(f["keypoint_scores"][0])
        height, width = image.shape
        return {
            "keypoints": torch.cat(keypoints)[None],
            "descriptors": torch.cat(descriptors)[None],
            "keypoint_scores": torch.cat(scores)[None],
            "image_size": torch.tensor([[width, height]], device=self.device, dtype=torch.float32),
        }
