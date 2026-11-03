"""How closely a splat render matches the mesh render it was trained from: PSNR and SSIM.

Both are measured against the camera's full scale, the mesh render's 99.9th percentile,
where degrade.expose puts full scale. Renderer radiance has no absolute meaning (white
flat ground under a zenith sun is 1), so a fixed peak of 1 would flatter dark terrain.

SSIM is Wang et al. 2004's: an 11-pixel Gaussian window with sigma 1.5, C1 = (0.01 L)^2
and C2 = (0.03 L)^2 for peak L. The same function is training's SSIM loss, so it works on
torch tensors and keeps gradients.

Every report carries a constant-image baseline (each view replaced by its mean
brightness) beside the splat's numbers: a gate the baseline can pass means nothing.
"""

import math
from dataclasses import dataclass
from statistics import median

import numpy as np
import torch
import torch.nn.functional as F

from fidelis.render.degrade import EXPOSURE_PERCENTILE

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5


def full_scale(reference: np.ndarray) -> float:
    return float(np.percentile(reference, EXPOSURE_PERCENTILE))


def psnr(image: torch.Tensor, reference: torch.Tensor, peak: float) -> float:
    mse = float(torch.mean((image.double() - reference.double()) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(peak**2 / mse)


def _window(device, dtype) -> torch.Tensor:
    x = torch.arange(SSIM_WINDOW, device=device, dtype=dtype) - SSIM_WINDOW // 2
    g = torch.exp(-(x**2) / (2 * SSIM_SIGMA**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(image: torch.Tensor, reference: torch.Tensor, peak: float) -> torch.Tensor:
    """Mean SSIM of two H x W (or B x H x W) images, as a differentiable scalar tensor."""
    x, y = image.reshape(-1, 1, *image.shape[-2:]), reference.reshape(-1, 1, *reference.shape[-2:])
    w = _window(x.device, x.dtype)

    def blur(t):
        # Reflect at the border, so a flat image stays flat to the edge.
        return F.conv2d(F.pad(t, [SSIM_WINDOW // 2] * 4, mode="reflect"), w)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x**2
    var_y = blur(y * y) - mu_y**2
    cov = blur(x * y) - mu_x * mu_y
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return s.mean()


@dataclass(frozen=True)
class QualityGate:
    psnr_db_min: float
    ssim_min: float

    def passes(self, view: dict) -> bool:
        return view["psnr_db"] >= self.psnr_db_min and view["ssim"] >= self.ssim_min


def _scores(images, references) -> list[dict]:
    views = []
    for image, reference in zip(images, references, strict=True):
        peak = full_scale(reference)
        a, b = torch.as_tensor(np.asarray(image, np.float32)), torch.as_tensor(np.asarray(reference, np.float32))
        views.append({"psnr_db": psnr(a, b, peak), "ssim": float(ssim(a, b, peak))})
    return views


def _summary(views: list[dict]) -> dict:
    return {
        "worst": {k: min(v[k] for v in views) for k in ("psnr_db", "ssim")},
        "median": {k: median(v[k] for v in views) for k in ("psnr_db", "ssim")},
    }


def quality_report(images, references, gate: QualityGate) -> dict:
    """Per-view PSNR and SSIM of `images` against `references`; passed only if every view passes the gate."""
    views = _scores(images, references)
    baseline = _scores([np.full_like(r, np.mean(r)) for r in references], references)
    return {
        "gate": {"psnr_db_min": gate.psnr_db_min, "ssim_min": gate.ssim_min},
        "passed": all(gate.passes(v) for v in views),
        **_summary(views),
        "views": views,
        "constant_baseline": _summary(baseline),
    }
