# Camera conventions

One pose convention for every part of Fidelis that places a camera: the Mitsuba mesh renderer (issue #5), splat training and rendering (gsplat), and the navigator's PnP (OpenCV). Code: `src/fidelis/render/camera.py`.

## World frame

The terrain mesh's local frame, in metres (`mesh.json`):

- **x east, y north**, measured from the tile centre (`origin` in `mesh.json`, in map metres of the mosaic's equirectangular projection).
- **z up**, the DTM elevation itself (around −2570 m at Jezero), not height above ground.

So a camera 900 m above Jezero's mean ground sits at z ≈ −1672 m. Map metres → local: subtract `origin` from x and y.

## Camera frame and pose

OpenCV's camera frame: **x right, y down, z forward** along the optical axis.

A `CameraPose` holds:

- `rotation`: the 3 × 3 **camera-to-world** rotation. Its columns are the camera's x, y and z axes written in world coordinates. A proper rotation (det +1); anything else is rejected.
- `position`: the camera centre in world metres.

`pose.world_to_camera()` returns the inverse as a 4 × 4 matrix `[Rᵀ | −Rᵀ·position]`. That is exactly:

- gsplat's `viewmats` ("the world-to-cam transformation of the cameras", `gsplat/rendering.py`), with `Ks` from `Intrinsics.matrix()`;
- OpenCV's extrinsics: `rvec = cv2.Rodrigues(w2c[:3, :3])[0]`, `tvec = w2c[:3, 3]`, as `cv2.projectPoints` takes them and `cv2.solvePnP` returns them. The navigator's PnP output converts back with `rotation = R.T`, `position = -R.T @ tvec`.

`look_at(eye, target, up)` builds a pose; `up` is the world direction that appears towards the top of the image. A nadir camera with `up = [0, 1, 0]` gives a north-up image (right = east, down = south), like a map. `up` parallel to the view direction is rejected, since the roll is then undefined.

## Intrinsics and pixels

`Intrinsics(width, height, fx, fy, cx, cy)` is OpenCV's pinhole K, in pixels. **Pixel (0, 0)'s centre is at (0, 0)**, so a centred principal point is `((W − 1) / 2, (H − 1) / 2)`: 511.5 for a 1024-pixel image, not 512. `Intrinsics.from_fov(W, H, fov_x_deg)` builds square pixels with a centred principal point.

No lens distortion. The Mitsuba renderer needs `fx == fy` (square pixels) and raises otherwise.

The descent camera mirrors Perseverance's LCAM: **1024 × 1024 pixels, 90° × 90° field of view** (Maki et al. 2020, "The Mars 2020 Engineering Cameras and Microphone on the Perseverance Rover", Space Science Reviews 216:137, section on LCAM requirements: "a field of view 90° by 90°, a format of 1024 by 1024 pixels"; open access, PMC7686239). That is `Intrinsics.from_fov(1024, 1024, 90)`: fx = fy = 512, cx = cy = 511.5.

## Sun

`Sun(azimuth_deg, elevation_deg)` in `src/fidelis/render/mesh_render.py`: azimuth is the **compass bearing of the sun** (0 north, 90 east, clockwise), elevation is the angle **above the horizon**, from 0 to 90. `sun.direction()` is the unit vector from the ground towards the sun in the world frame.

## How it's checked

`tests/render/test_mesh_render.py` renders small glowing spheres with an oblique, rolled camera whose principal point is off-centre in both axes, and requires each sphere's image centroid to land within 0.1 px of `cv2.projectPoints` of its centre. That pins the axes, the pixel origin, the half-pixel convention, and cx/cy together, against OpenCV's own projection rather than our reading of it. The first run of this test found that Mitsuba's `principal_point_offset` has the opposite sign to pixel coordinates.

## Gotchas

- **Mitsuba's camera frame is not OpenCV's.** Mitsuba's perspective camera looks along +z with +y up and +x to the left. `mitsuba_sensor()` converts with `diag(−1, −1, 1)`; don't build Mitsuba cameras by hand.
- **Mitsuba's film puts pixel centres at half-integers**, OpenCV at integers; `mitsuba_sensor()` accounts for it.
- **The field of view limits altitude on the 2 km tile.** A nadir 90° camera sees 2 × altitude across, so above about 1000 m it sees past the mesh's edge, into empty black. See `docs/benchmarks/mesh-render.md`.
