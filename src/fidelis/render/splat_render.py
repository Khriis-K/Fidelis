"""Renders of a trained splat scene from a camera pose: the simulator's "splat" fidelity.

gsplat rasterizes the Gaussians with the pose's world-to-camera matrix as its `viewmats`
(docs/camera-conventions.md). One convention differs: gsplat puts pixel (0, 0)'s centre
at (0.5, 0.5), OpenCV at (0, 0), so the principal point moves half a pixel on the way in
(pinned by tests/render/test_splat_render.py against cv2.projectPoints).

Radiance comes out as the splat learned it from the mesh renders: linear, one channel,
black where no Gaussian covers the view. There's no sampling noise, so the seed is unused.
"""

from pathlib import Path

import numpy as np
import torch

from fidelis.render.camera import CameraPose, Intrinsics
from fidelis.splat.model import load_checkpoint


def render(params: dict[str, torch.Tensor], pose: CameraPose, intrinsics: Intrinsics, info: dict | None = None):
    """A height x width radiance tensor on the parameters' device, differentiable in them.

    gsplat's per-Gaussian bookkeeping (what training's densification needs) goes into `info`.
    """
    from fidelis.gsplat_loader import import_gsplat

    gsplat = import_gsplat()
    device = params["means"].device
    k = intrinsics.matrix()
    k[0, 2] += 0.5
    k[1, 2] += 0.5
    image, _, meta = gsplat.rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=params["colors"],
        viewmats=torch.tensor(pose.world_to_camera(), dtype=torch.float32, device=device)[None],
        Ks=torch.tensor(k, dtype=torch.float32, device=device)[None],
        width=intrinsics.width,
        height=intrinsics.height,
        packed=False,
    )
    if info is not None:
        info.update(meta)
    return image[0, ..., 0]


class SplatRenderer:
    """The splat scenes of the suns that have one; keeps one sun's scene on the GPU at a time."""

    def __init__(self, checkpoints: dict[int, Path], device: str = "cuda"):
        self.checkpoints = {int(k): Path(v) for k, v in checkpoints.items()}
        self.device = device
        self._params, self._sun_index = None, None

    @property
    def suns(self) -> list[int]:
        return sorted(self.checkpoints)

    def use_sun(self, sun_index: int) -> None:
        """Load that sun's scene, unless it's the one loaded. The simulator calls this untimed."""
        if sun_index not in self.checkpoints:
            raise ValueError(f"no splat scene for sun {sun_index}; this renderer has suns {self.suns}")
        if sun_index != self._sun_index:
            self._params = None
            self._params, _ = load_checkpoint(self.checkpoints[sun_index], device=self.device)
            self._sun_index = sun_index

    def __call__(self, pose: CameraPose, intrinsics: Intrinsics, sun_index: int, seed: int) -> np.ndarray:
        self.use_sun(sun_index)
        with torch.no_grad():
            image = render(self._params, pose, intrinsics)
        return image.cpu().numpy().astype(np.float32)
