"""What warping by attitude costs on the CPU tests' synthetic nadir view (issue #34).

The analytic fixture from tests/nav/test_navigator.py (a 256 px nadir view, 1.5 m
descent pixels, of a 2 m noise-texture map) at several headings, each located with
its true attitude. Turns by multiples of 90 degrees only move whole pixels; others
resample the image. Then eight more camera positions, and the north-up view resampled
45 degrees and back, matched without a warp, to see how much of the cost is resampling.
Bilinear (the navigator's) against bicubic warping. The numbers in
docs/benchmarks/navigator.md, "Warping by attitude".

    uv run python scripts/check_attitude_warp.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "nav"))
import test_navigator as fixture  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

import fidelis.nav.navigator as nav  # noqa: E402
from fidelis.nav.orbital_map import OrbitalMap  # noqa: E402

f = fixture
flat_map = OrbitalMap(
    image=f.texture(f.MAP_SIZE),
    transform=from_origin(f.LEFT, f.TOP, f.MAP_RES, f.MAP_RES),
    dtm=np.full((8, 8), f.GROUND_Z, np.float32),
    dtm_transform=from_origin(f.LEFT, f.TOP, f.MAP_RES * f.MAP_SIZE / 8, f.MAP_RES * f.MAP_SIZE / 8),
    origin=f.ORIGIN,
    east_scale=f.EAST_SCALE,
)
navigator = nav.Navigator(flat_map, device="cpu")
truth = f.TRUE_POSITION.copy()


def error(heading, attitude=True):
    """Position error (m) of the fixture's view at `heading`, or None without a fix."""
    view = np.clip(f.nadir_view(flat_map, heading), 0, 1)
    result = navigator.locate(view, f.K, f.attitude(heading) if attitude else None)
    return None if result.pose is None else float(np.linalg.norm(result.position - f.TRUE_POSITION))


def fmt(e):
    return "-" if e is None else f"{e:.3f}"


bilinear = nav.cv2.warpPerspective


def bicubic(image, homography, size, flags=cv2.INTER_LINEAR):
    return bilinear(image, homography, size, flags=cv2.INTER_CUBIC if flags == cv2.INTER_LINEAR else flags)


headings = (0, 15, 30, 45, 60, 90, 135, 180, 270)
print(f"North-up, no warp: {fmt(error(0, attitude=False))} m\n")
print("| Heading | Bilinear warp (m) | Bicubic warp (m) |\n|---|---|---|")
for h in headings:
    linear = error(h)
    nav.cv2.warpPerspective = bicubic
    cubic = error(h)
    nav.cv2.warpPerspective = bilinear
    print(f"| {h} | {fmt(linear)} | {fmt(cubic)} |")

print("\nEight more positions (bilinear)\n| East, north (m) | 0 deg | 45 deg | 15 deg |\n|---|---|---|---|")
rng = np.random.default_rng(1)
for _ in range(8):
    f.TRUE_POSITION[:] = truth + np.r_[rng.uniform(-60, 60, 2), 0]
    east, north = f.TRUE_POSITION[:2]
    print(f"| {east:.1f}, {north:.1f} | {fmt(error(0))} | {fmt(error(45))} | {fmt(error(15))} |")
f.TRUE_POSITION[:] = truth

# Resampling alone: the north-up view turned 45 degrees and back, its empty corners zeroed.
view = np.clip(f.nadir_view(flat_map), 0, 1).astype(np.float32)
size = (f.K.width, f.K.height)
there, back = (cv2.getRotationMatrix2D((f.K.cx, f.K.cy), a, 1.0) for a in (45, -45))
twice = cv2.warpAffine(cv2.warpAffine(view, there, size), back, size)
covered = cv2.warpAffine(
    cv2.warpAffine(np.ones_like(view), there, size, flags=cv2.INTER_NEAREST), back, size, flags=cv2.INTER_NEAREST
)
result = navigator.locate(np.where(covered > 0, twice, 0), f.K)
print(f"\nNorth-up resampled 45 deg and back, no warp: {np.linalg.norm(result.position - truth):.3f} m")

# Attitude-knowledge error: the believed attitude is the true one turned by `e` degrees
# about world up (a heading error) or world east (a tilt error). Each row compares with
# the same view's perfect-attitude row (e = 0), not with north-up.
print("\nAttitude error (bilinear)")
print("| View heading | Error axis | Error (deg) | Status | Inliers | Position error (m) |\n|---|---|---|---|---|---|")
for heading in (0, 45):
    view = np.clip(f.nadir_view(flat_map, heading), 0, 1)
    for axis_name, axis in (("heading", [0, 0, 1]), ("tilt", [1, 0, 0])):
        for e in (0, 1, 3, 10, 20):
            turn = cv2.Rodrigues(np.radians(e) * np.array(axis, dtype=np.float64))[0]
            result = navigator.locate(view, f.K, turn @ f.attitude(heading))
            err = None if result.pose is None else float(np.linalg.norm(result.position - truth))
            print(f"| {heading} | {axis_name} | {e} | {result.status} | {result.inlier_count} | {fmt(err)} |")
