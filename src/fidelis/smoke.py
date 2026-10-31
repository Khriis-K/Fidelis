"""Environment smoke test: run a minimal real example of each core library.

`uv run fidelis-smoke` runs every check and prints PASS/FAIL per library.
"""

from collections.abc import Callable
from dataclasses import dataclass

Check = Callable[[], str]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def run_checks(checks: dict[str, Check]) -> list[CheckResult]:
    """Run each check in order; an exception fails that check only."""
    results = []
    for name, check in checks.items():
        try:
            results.append(CheckResult(name, True, check()))
        except Exception as exc:
            results.append(CheckResult(name, False, f"{type(exc).__name__}: {exc}"))
    return results


# Each check imports its library itself, so one broken install fails only its own
# check, and asserts against an independent ground truth rather than "it ran".


def check_torch() -> str:
    """A matmul on the GPU agrees with the same matmul on the CPU."""
    import torch

    a = torch.randn(512, 512, generator=torch.Generator().manual_seed(0))
    gpu = (a.cuda() @ a.cuda()).cpu()
    # Default matmul precision on Ampere+ may use TF32, hence the loose tolerance.
    if not torch.allclose(gpu, a @ a, rtol=1e-2, atol=1e-1):
        raise AssertionError("GPU matmul disagrees with CPU matmul")
    major, minor = torch.cuda.get_device_capability()
    return f"torch {torch.__version__} on {torch.cuda.get_device_name()} (sm_{major}{minor})"


def check_mitsuba() -> str:
    """The CUDA variant path-traces a small Cornell box into a lit, non-flat image."""
    import mitsuba as mi
    import numpy as np

    mi.set_variant("cuda_ad_rgb")
    scene_dict = mi.cornell_box()
    scene_dict["sensor"]["film"]["width"] = 64
    scene_dict["sensor"]["film"]["height"] = 64
    image = np.array(mi.render(mi.load_dict(scene_dict), spp=16))
    if image.shape != (64, 64, 3) or not np.isfinite(image).all():
        raise AssertionError(f"bad render: shape {image.shape}")
    if image.mean() <= 0 or image.std() <= 0:
        raise AssertionError("render is black or flat")
    return f"mitsuba {mi.__version__}, cuda_ad_rgb, 64x64 at 16 spp"


def check_gsplat() -> str:
    """One Gaussian straight ahead of the camera rasterizes at the principal point."""
    import torch

    from fidelis.gsplat_loader import import_gsplat

    gsplat = import_gsplat()
    size = 64
    colors, _, _ = gsplat.rasterization(
        means=torch.tensor([[0.0, 0.0, 5.0]], device="cuda"),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        scales=torch.full((1, 3), 0.2, device="cuda"),
        opacities=torch.ones(1, device="cuda"),
        colors=torch.ones(1, 3, device="cuda"),
        viewmats=torch.eye(4, device="cuda")[None],
        Ks=torch.tensor([[[50.0, 0.0, size / 2], [0.0, 50.0, size / 2], [0.0, 0.0, 1.0]]], device="cuda"),
        width=size,
        height=size,
    )
    brightness = colors[0, ..., 0]
    row, col = divmod(int(brightness.argmax()), size)
    if abs(row - size / 2) > 1 or abs(col - size / 2) > 1:
        raise AssertionError(f"Gaussian landed at pixel ({row}, {col}), expected the center")
    if brightness[0, 0] >= brightness[row, col]:
        raise AssertionError("corner is as bright as the Gaussian")
    return f"gsplat {gsplat.__version__}, peak at pixel ({row}, {col})"


def check_lightglue() -> str:
    """ALIKED + LightGlue matches a texture to a warped copy, agreeing with the warp."""
    import cv2
    import numpy as np
    import torch
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import rbd

    # Multi-scale smoothed noise: a synthetic texture with corners at many scales.
    rng = np.random.default_rng(0)
    height, width = 480, 640
    texture = np.zeros((height, width), np.float32)
    for sigma in (2, 4, 8, 16):
        texture += cv2.GaussianBlur(rng.random((height, width), np.float32), (0, 0), sigma)
    texture = cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX)
    homography = np.array([[0.95, 0.05, 20.0], [-0.04, 0.97, 10.0], [0.0, 0.0, 1.0]])
    warped = cv2.warpPerspective(texture, homography, (width, height))

    def to_tensor(gray):
        return torch.from_numpy(np.repeat(gray[None], 3, axis=0)).cuda()

    extractor = ALIKED(max_num_keypoints=1024).eval().cuda()
    matcher = LightGlue(features="aliked").eval().cuda()
    with torch.inference_mode():
        feats0 = extractor.extract(to_tensor(texture))
        feats1 = extractor.extract(to_tensor(warped))
        result = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, result = rbd(feats0), rbd(feats1), rbd(result)
    pairs = result["matches"].cpu().numpy()
    kpts0 = feats0["keypoints"][pairs[:, 0]].cpu().numpy()
    kpts1 = feats1["keypoints"][pairs[:, 1]].cpu().numpy()

    projected = cv2.perspectiveTransform(kpts0[None].astype(np.float64), homography)[0]
    inliers = int((np.linalg.norm(projected - kpts1, axis=1) < 3.0).sum())
    if len(pairs) < 50 or inliers < 0.8 * len(pairs):
        raise AssertionError(f"{inliers}/{len(pairs)} matches agree with the known warp")
    return f"{inliers}/{len(pairs)} matches within 3 px of the known warp"


def check_mxmcpy() -> str:
    """MFMC and ACV allocations for correlated toy models beat plain MC at equal cost."""
    import numpy as np
    from mxmc import Optimizer

    costs = np.array([1.0, 0.1, 0.01])
    stdevs = np.array([1.0, 0.9, 0.8])
    correlation = np.array([[1.0, 0.95, 0.9], [0.95, 1.0, 0.92], [0.9, 0.92, 1.0]])
    covariance = correlation * np.outer(stdevs, stdevs)
    target_cost = 100.0
    plain_mc_variance = covariance[0, 0] / (target_cost / costs[0])

    optimizer = Optimizer(costs, covariance)
    variances = {}
    for algorithm in ("mfmc", "acvmf"):
        result = optimizer.optimize(algorithm, target_cost)
        variance = float(np.squeeze(result.variance))  # ACV returns a 1-element array
        if result.cost > target_cost * (1 + 1e-9):
            raise AssertionError(f"{algorithm} spent {result.cost} > target {target_cost}")
        if not 0 < variance < plain_mc_variance:
            raise AssertionError(f"{algorithm} variance {variance} not below plain MC {plain_mc_variance}")
        variances[algorithm] = variance
    summary = ", ".join(f"{a} {v:.2e}" for a, v in variances.items())
    return f"estimator variance {summary} vs plain MC {plain_mc_variance:.2e}"


CHECKS: dict[str, Check] = {
    "torch": check_torch,
    "mitsuba": check_mitsuba,
    "gsplat": check_gsplat,
    "lightglue": check_lightglue,
    "mxmcpy": check_mxmcpy,
}


def main(checks: dict[str, Check] | None = None) -> int:
    """Run the checks, print one PASS/FAIL line per library, return the exit code."""
    results = run_checks(CHECKS if checks is None else checks)
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name}: {r.detail}")
    return 0 if all(r.passed for r in results) else 1
