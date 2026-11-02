"""Camera poses and intrinsics, in the one convention Fidelis uses everywhere.

World frame: the terrain mesh's local ground metres (x east, y north, z up), as in mesh.json.

Camera frame: OpenCV's, x right, y down, z forward along the optical axis. A pose is
the camera-to-world rotation (its columns are the camera's axes in world coordinates)
plus the camera centre in world metres. `world_to_camera()` is the inverse as a 4 x 4
matrix: what gsplat takes as `viewmats`, and what cv2.projectPoints and solvePnP mean
by rvec and tvec.

Pixels: OpenCV's, pixel (0, 0)'s centre at (0, 0). So the principal point of a centred
1024-pixel-wide camera is cx = 511.5.

See docs/camera-conventions.md.
"""

import math
from dataclasses import dataclass

import numpy as np

# How far RᵀR may be from the identity: room for float32 storage (gsplat, JSON round
# trips; about 1e-7), far below any real error such as a scaled or sheared matrix.
ROTATION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class CameraPose:
    rotation: np.ndarray  # 3 x 3 camera-to-world; columns are camera x, y, z in the world frame
    position: np.ndarray  # camera centre, world metres

    def __post_init__(self):
        r = np.asarray(self.rotation, dtype=np.float64)
        if r.shape != (3, 3) or not np.allclose(r.T @ r, np.eye(3), atol=ROTATION_TOLERANCE) or np.linalg.det(r) < 0:
            raise ValueError(f"camera-to-world matrix is not a rotation:\n{r}")
        object.__setattr__(self, "rotation", r)
        object.__setattr__(self, "position", np.asarray(self.position, dtype=np.float64).reshape(3))

    def world_to_camera(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotation.T
        m[:3, 3] = -self.rotation.T @ self.position
        return m


def look_at(eye, target, up) -> CameraPose:
    """A camera at `eye` looking at `target`, with `up` pointing towards the image's top."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    z = target - eye
    if np.linalg.norm(z) < 1e-9:
        raise ValueError("target is at the eye, so the view direction is undefined")
    z /= np.linalg.norm(z)
    x = np.cross(-up, z)
    if np.linalg.norm(x) < 1e-9:
        raise ValueError("up is parallel to the view direction, so the camera's roll is undefined")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return CameraPose(rotation=np.stack([x, y, z], axis=1), position=eye)


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, fov_x_deg: float) -> "Intrinsics":
        """Square pixels, principal point on the image centre."""
        f = width / (2 * math.tan(math.radians(fov_x_deg) / 2))
        return cls(width, height, f, f, (width - 1) / 2, (height - 1) / 2)

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)
