"""The Gaussians of a splat scene: where they start, how they're saved, how browsers get them.

Parameters are kept as gsplat trains them, unconstrained: `means` (world metres,
docs/camera-conventions.md), `scales` (log metres), `quats` (wxyz, not normalised),
`opacities` (logits) and `colors`, one grey channel of linear radiance. One channel
because LCAM is monochrome and the mesh is a diffuse (Lambertian) surface, so a point
looks the same from every direction and spherical harmonics would have nothing to fit.

The scene starts on the true surface: one Gaussian per mesh vertex (the tile, and the
CTX ring if built), round and as wide as the vertex spacing, with the texture's albedo
under it times sin(sun elevation), the radiance of flat ground under that sun
(fidelis.render.mesh_render). Training then only refines a scene that's nearly right.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from fidelis.render.mesh_render import Sun
from fidelis.terrain.mesh import read_ply_vertices

# The zeroth spherical-harmonic basis constant: viewers decode colour as 0.5 + SH_C0 * f_dc.
SH_C0 = 0.28209479177387814
# gsplat's simple_trainer starting opacity.
INITIAL_OPACITY = 0.1


def gaussians(means: np.ndarray, greys: np.ndarray, spacing: np.ndarray) -> dict[str, torch.Tensor]:
    """Round Gaussians at `means` with grey `greys`, each `spacing` metres wide."""
    n = len(means)
    log_scale = np.log(np.asarray(spacing, np.float64))
    return {
        "means": torch.tensor(np.asarray(means), dtype=torch.float32),
        "scales": torch.tensor(np.repeat(log_scale[:, None], 3, axis=1), dtype=torch.float32),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, dtype=torch.float32).reshape(n, 4),
        "opacities": torch.full((n,), math.log(INITIAL_OPACITY / (1 - INITIAL_OPACITY)), dtype=torch.float32),
        "colors": torch.tensor(np.asarray(greys).reshape(n, 1), dtype=torch.float32),
    }


def _bilinear(texture: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """Sample a texture at UVs with v = 0 on its bottom row (fidelis.terrain.mesh's convention)."""
    height, width = texture.shape
    col = np.clip(uvs[:, 0] * width - 0.5, 0, width - 1)
    row = np.clip((1 - uvs[:, 1]) * height - 0.5, 0, height - 1)
    c0, r0 = np.minimum(col.astype(int), width - 2), np.minimum(row.astype(int), height - 2)
    fc, fr = col - c0, row - r0
    top = texture[r0, c0] * (1 - fc) + texture[r0, c0 + 1] * fc
    bottom = texture[r0 + 1, c0] * (1 - fc) + texture[r0 + 1, c0 + 1] * fc
    return top * (1 - fr) + bottom * fr


def _spacing(positions: np.ndarray) -> float:
    """A grid mesh's vertex spacing: the median gap between consecutive vertices (row ends are the outliers)."""
    return float(np.median(np.linalg.norm(np.diff(positions.astype(np.float64), axis=0), axis=1)))


def init_from_mesh(mesh_dir: Path, sun: Sun) -> dict[str, torch.Tensor]:
    mesh_dir = Path(mesh_dir)
    record = json.loads((mesh_dir / "mesh.json").read_text())
    shapes = [("mesh.ply", "texture.png")]
    if record.get("ring") is not None:
        shapes.append((record["ring"]["ply"], record["ring"]["texture"]))
    lit = math.sin(math.radians(sun.elevation_deg))
    means, greys, spacing = [], [], []
    for ply, texture_name in shapes:
        positions, uvs = read_ply_vertices(mesh_dir / ply)
        texture = cv2.imread(str(mesh_dir / texture_name), cv2.IMREAD_GRAYSCALE)
        if texture is None:
            raise OSError(f"could not read {mesh_dir / texture_name}")
        means.append(positions)
        greys.append(_bilinear(texture.astype(np.float64) / 255, uvs.astype(np.float64)) * lit)
        spacing.append(np.full(len(positions), _spacing(positions)))
    return gaussians(np.concatenate(means), np.concatenate(greys), np.concatenate(spacing))


def save_checkpoint(path: Path, params: dict[str, torch.Tensor], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    torch.save({"params": {k: v.detach().cpu() for k, v in params.items()}, "meta": meta}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[dict[str, torch.Tensor], dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    return checkpoint["params"], checkpoint["meta"]


PLY_PROPERTIES = (
    ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)


def export_ply(path: Path, params: dict[str, torch.Tensor]) -> None:
    """The scene as a 3DGS PLY (the layout of Kerbl et al.'s reference code), which browser viewers read.

    The grey goes into all three colour channels as a degree-0 spherical harmonic. Colours
    are the renderer's linear radiance, not exposed for display.
    """
    p = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in params.items()}
    n = len(p["means"])
    f_dc = np.repeat((p["colors"] - 0.5) / SH_C0, 3, axis=1)
    quats = p["quats"] / np.linalg.norm(p["quats"], axis=1, keepdims=True)
    columns = [p["means"], np.zeros((n, 3), np.float32), f_dc, p["opacities"][:, None], p["scales"], quats]
    values = np.concatenate(columns, axis=1).astype("<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n" + "".join(f"property float {name}\n" for name in PLY_PROPERTIES) + "end_header\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    with open(tmp, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(values.tobytes())
    tmp.replace(path)
