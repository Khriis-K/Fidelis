# Camera conventions

One pose convention for every part of Fidelis that places a camera: the Mitsuba mesh renderer (issue #5), splat training and rendering (gsplat), and the navigator's PnP (OpenCV). Code: `src/fidelis/render/camera.py`.

## World frame

The terrain mesh's local frame, in metres (`mesh.json`):

- **x east, y north, in ground metres**, measured from the tile centre (`origin` in `mesh.json`, in map metres of the mosaic's equirectangular projection).
- **z up**, the DTM elevation itself (around −2550 m at Jezero), not height above ground.

So a camera 2.2 km above the 10 km tile's mean ground (−2553.2 m) sits at z ≈ −353 m.

**Map metres are not ground metres east–west.** The mosaics' projection has its standard parallel at the equator (`lat_ts=0`), so it stretches east–west distances by 1/cos(latitude): 5.4% at Jezero. The mesh corrects for it with one factor, `east_scale` in `mesh.json` (cos latitude at the tile centre, 0.94863). Map metres → local: x = (map x − origin x) × `east_scale`, y = map y − origin y. Local → map: x / `east_scale` + origin x. North–south needs no correction.

**The world frame is flat; Mars isn't.** The tile is a flat projection of a sphere, so true ground falls away from it by d² / 2R: **3.7 m at 5 km from the centre** (R = 3,396,190 m). That's below a descent-camera pixel at LVS altitudes (4.3–8.2 m), and it isn't modelled. A long, low, oblique view would see the horizon in the wrong place.

## Camera frame and pose

OpenCV's camera frame: **x right, y down, z forward** along the optical axis.

A `CameraPose` holds:

- `rotation`: the 3 × 3 **camera-to-world** rotation. Its columns are the camera's x, y and z axes written in world coordinates. A proper rotation (det +1): RᵀR must be within 1e-6 of the identity, which allows for float32 storage (gsplat, JSON round trips) but rejects a scaled or sheared matrix. Anything else is rejected.
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
- **The field of view limits altitude and tilt.** A nadir 90° camera sees 2 × altitude across. The 10 km mesh is 9.50 km of ground east–west, so at LVS's top altitude (4.2 km) a nadir view leaves only about 550 m spare on each side (about 3.5° of tilt); at 2.2 km it leaves about 2.5 km. That's for a north-up view over ground at the mesh's mean height: turning the camera swings its corners outward (past about 8° of heading at 4.2 km a corner leaves the mesh), and ground lower than the mean (down to 99 m lower here) widens the footprint. Past the edge the image is empty black. `require_no_blank_lines()` catches whole blank rows or columns, not a clipped corner. See `docs/benchmarks/mesh-render.md`.
