"""Path-traced renders of the terrain mesh from a camera pose under a chosen sun.

Mitsuba 3's path tracer traces a shadow ray to the sun from every shading point, so
shadows fall where the geometry puts them. The sun is a directional light (a point on
the sky; the real sun's 0.35 degree disc would soften shadow edges by about 6 mm per
metre of occluder height). There is no sky light: shadows get only light bounced off
the terrain (`MAX_DEPTH`), so they are darker than on Mars, where a dusty sky fills them.
Shading uses Mitsuba's smooth per-vertex normals; shadow rays use the true triangles.

Radiometry: the sun's irradiance is pi, so flat ground with the sun at the zenith has
radiance equal to its albedo (the orthoimage value / 255), and sunlit flat ground in
general has albedo * sin(elevation). Exposure is left to the caller.

Needs a Mitsuba variant set by the caller (`cuda_ad_rgb` for real renders). See
docs/camera-conventions.md for the pose and pixel conventions.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fidelis.render.camera import CameraPose, Intrinsics
from fidelis.terrain.mesh import terrain_shapes

MAX_DEPTH = 3  # camera ray, direct sun, and one bounce off the terrain
# Same scene, pose, spp and seed on the same variant and machine: the largest pixel
# difference allowed between two renders. scalar_rgb repeats bit for bit; cuda_ad_rgb's
# film adds samples atomically in no fixed order, which moved pixels by up to 3.0e-7 on
# the 10 km Jezero mesh (docs/benchmarks/mesh-render.md).
RENDER_DETERMINISM_TOLERANCE = 1e-6
# Mitsuba's perspective camera looks along its local +z with +y up and +x to the left;
# OpenCV's has +y down and +x to the right.
_OPENCV_TO_MITSUBA = np.diag([-1.0, -1.0, 1.0])
# The tile texture as photographed (fidelis.terrain.mesh); fidelis.stress names the others.
PHOTO_TEXTURE = "texture.png"


@dataclass(frozen=True)
class Sun:
    azimuth_deg: float  # compass bearing of the sun: 0 north, 90 east
    elevation_deg: float  # above the horizon

    def __post_init__(self):
        if not 0 <= self.elevation_deg <= 90:
            raise ValueError(f"sun elevation {self.elevation_deg} deg must be in [0, 90]")

    def direction(self) -> np.ndarray:
        """Unit vector from the ground towards the sun, in the mesh's world frame."""
        az, el = math.radians(self.azimuth_deg), math.radians(self.elevation_deg)
        return np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])


def config_suns(scene: dict) -> list[Sun]:
    """The scene config's sun angles, in sun_index order."""
    return [Sun(s["azimuth_deg"], s["elevation_deg"]) for s in scene["suns"]]


def terrain_scene(mesh_dir: Path, sun: Sun, texture: str = PHOTO_TEXTURE):
    """The textured terrain (the tile, and the CTX ring if built) lit by the sun.

    Load once per mesh and sun, render many poses. `texture` names the tile's texture
    file (`fidelis.terrain.mesh.terrain_shapes`).
    """
    import mitsuba as mi

    return mi.load_dict(
        {
            "type": "scene",
            **terrain_shapes(mesh_dir, texture),
            "sun": {"type": "directional", "direction": (-sun.direction()).tolist(), "irradiance": math.pi},
        }
    )


def mitsuba_sensor(pose: CameraPose, intrinsics: Intrinsics, spp: int) -> dict:
    """A Mitsuba perspective camera that projects exactly as the OpenCV pinhole (pose, K) does."""
    import mitsuba as mi

    if not math.isclose(intrinsics.fx, intrinsics.fy, rel_tol=1e-9):
        raise ValueError(f"fx = {intrinsics.fx} and fy = {intrinsics.fy}: Mitsuba renders square pixels only")
    to_world = np.eye(4)
    to_world[:3, :3] = pose.rotation @ _OPENCV_TO_MITSUBA
    to_world[:3, 3] = pose.position
    return {
        "type": "perspective",
        "fov": math.degrees(2 * math.atan(intrinsics.width / (2 * intrinsics.fx))),
        "fov_axis": "x",
        # Mitsuba measures the principal point from the film's centre, in film widths and
        # heights, with the opposite sign to pixel coordinates (found by the projection
        # test). The film's centre is OpenCV pixel ((W - 1) / 2, (H - 1) / 2).
        "principal_point_offset_x": -(intrinsics.cx - (intrinsics.width - 1) / 2) / intrinsics.width,
        "principal_point_offset_y": -(intrinsics.cy - (intrinsics.height - 1) / 2) / intrinsics.height,
        "to_world": mi.ScalarTransform4f(to_world),
        "near_clip": 0.1,
        "far_clip": 1e6,
        "film": {
            "type": "hdrfilm",
            "width": intrinsics.width,
            "height": intrinsics.height,
            "pixel_format": "luminance",
            # Each sample counts only towards its own pixel, as in a pinhole camera.
            "rfilter": {"type": "box"},
        },
        "sampler": {"type": "independent", "sample_count": spp},
    }


def render(scene, pose: CameraPose, intrinsics: Intrinsics, spp: int, seed: int = 0) -> np.ndarray:
    """A height x width float32 image of linear radiance."""
    import mitsuba as mi

    sensor = mi.load_dict(mitsuba_sensor(pose, intrinsics, spp))
    integrator = mi.load_dict({"type": "path", "max_depth": MAX_DEPTH})
    image = mi.render(scene, sensor=sensor, integrator=integrator, spp=spp, seed=seed)
    return np.array(image, dtype=np.float32)[..., 0]


class MeshRenderer:
    """The simulator's "mesh" fidelity: the mesh, with the tile texture named, under one of a fixed list of suns.

    Keeps one sun's scene loaded at a time (about 2 s to load the Jezero mesh); callers
    running many descents should group them by sun. Sets the `cuda_ad_rgb` variant.
    """

    def __init__(self, mesh_dir: Path, suns: list[Sun], spp: int, texture: str = PHOTO_TEXTURE):
        import mitsuba as mi

        mi.set_variant("cuda_ad_rgb")
        self.mesh_dir, self.suns, self.spp, self.texture = Path(mesh_dir), list(suns), spp, texture
        self._scene, self._sun_index = None, None

    def use_sun(self, sun_index: int) -> None:
        """Load that sun's scene, unless it's the one loaded. The simulator calls this untimed."""
        if not 0 <= sun_index < len(self.suns):
            raise ValueError(f"sun index {sun_index}; this renderer has {len(self.suns)} suns")
        if sun_index != self._sun_index:
            import drjit as dr

            self._scene = None
            dr.flush_malloc_cache()
            self._scene = terrain_scene(self.mesh_dir, self.suns[sun_index], self.texture)
            self._sun_index = sun_index

    def __call__(self, pose: CameraPose, intrinsics: Intrinsics, sun_index: int, seed: int) -> np.ndarray:
        self.use_sun(sun_index)
        image = render(self._scene, pose, intrinsics, self.spp, seed=seed)
        require_no_blank_lines(image)
        return image


def require_no_blank_lines(image: np.ndarray) -> None:
    """Raise if any image row or column is exactly zero.

    A 10 km mesh at 2 m spacing filled the laptop's 8 GB GPU, and Mitsuba returned
    streaked, partly black images without raising (likely out of GPU memory). Terrain
    lit by a sun above the horizon doesn't render a whole line of exact zeros. Empty
    space past the mesh's edge does (there's no sky light), so this also fails a view
    that runs off the mesh along a whole edge of the image, but not one that only clips
    a corner. Not checked: whether that out-of-memory render's damage was exact zeros
    along whole lines; it was described as streaks and a black wedge.
    """
    for axis, name in ((1, "rows"), (0, "columns")):
        blank = np.flatnonzero(~np.any(image != 0, axis=axis))
        if blank.size:
            raise ValueError(
                f"render has {blank.size} blank {name} (first {blank[0]}): out of GPU memory, or off the mesh?"
            )
