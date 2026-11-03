"""Train a splat scene from multi-view mesh renders, then check it against held-out views.

The recipe is gsplat's simple_trainer, cut to what Fidelis uses: one view per step, loss
(1 - w) L1 + w (1 - SSIM) with w = 0.2, Adam per parameter group at simple_trainer's
learning rates, positions' rate scaled by the scene's size and decaying 100x over the run,
and gsplat's DefaultStrategy growing, splitting and pruning Gaussians (from step 500 to
15,000, with an opacity reset every 3,000). The scene starts on the mesh's vertices
(fidelis.splat.model). SSIM's peak is each view's full scale (fidelis.splat.metrics).

    uv run fidelis-splat-train config/jezero.yaml data/jezero/splat/sun0/views data/jezero/mesh data/jezero/splat/sun0

writes splat.pt (the checkpoint the "splat" fidelity loads), splat.ply (for the browser
viewer) and quality.json (held-out PSNR and SSIM against the mesh, the gate from the
config, and the constant-image baseline).
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from fidelis.render.camera import CameraPose, Intrinsics
from fidelis.render.degrade import EXPOSURE_PERCENTILE
from fidelis.splat.metrics import QualityGate, quality_report, ssim

# gsplat simple_trainer's learning rates; means' is multiplied by the scene scale.
LEARNING_RATES = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3}
MEANS_LR_FINAL_FRACTION = 0.01


@dataclass(frozen=True)
class TrainSettings:
    steps: int = 30_000
    ssim_weight: float = 0.2
    log_every: int = 500  # 0: no progress lines

    @classmethod
    def from_config(cls, section: dict) -> "TrainSettings":
        return cls(**{k: section[k] for k in ("steps", "ssim_weight", "log_every")})


def scene_scale(poses: list[CameraPose]) -> float:
    centres = np.array([p.position for p in poses])
    return 1.1 * float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).max())


def train(
    params: dict[str, torch.Tensor],
    images: torch.Tensor,
    poses: list[CameraPose],
    intrinsics: Intrinsics,
    settings: TrainSettings,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Fit the Gaussians to `images` (views x height x width) seen from `poses`; returns the trained parameters."""
    from fidelis.gsplat_loader import import_gsplat
    from fidelis.render.splat_render import render

    import_gsplat()
    from gsplat.strategy import DefaultStrategy  # noqa: TID251 -- import_gsplat() has set up the build

    scale = scene_scale(poses)
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in params.items()})
    optimizers = {
        k: torch.optim.Adam([{"params": params[k], "lr": lr * (scale if k == "means" else 1), "name": k}], eps=1e-15)
        for k, lr in LEARNING_RATES.items()
    }
    means_decay = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=MEANS_LR_FINAL_FRACTION ** (1 / max(settings.steps, 1))
    )
    strategy = DefaultStrategy(verbose=settings.log_every > 0)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scale)
    images = images.to(device)
    # Each view's full scale, from every 4th pixel (torch.quantile refuses inputs over 2^24 values).
    peaks = [float(torch.quantile(image.float().flatten()[::4], EXPOSURE_PERCENTILE / 100)) for image in images]

    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for step in range(settings.steps):
        i = int(rng.integers(len(poses)))
        target = images[i].float()
        info = {}
        image = render(params, poses[i], intrinsics, info=info)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        l1 = torch.abs(image - target).mean()
        loss = (1 - settings.ssim_weight) * l1 + settings.ssim_weight * (1 - ssim(image, target, peaks[i]))
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        means_decay.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if settings.log_every and (step % settings.log_every == 0 or step == settings.steps - 1):
            print(
                f"step {step}: loss {loss.item():.5f}, L1 {l1.item():.5f}, {len(params['means'])} Gaussians,"
                f" {time.perf_counter() - start:.0f} s",
                flush=True,
            )
    return {k: v.detach() for k, v in params.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("views_dir", type=Path, help="fidelis-splat-views output")
    parser.add_argument("mesh_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    from fidelis.render.mesh_render import config_suns
    from fidelis.render.splat_render import render
    from fidelis.splat.model import export_ply, init_from_mesh, save_checkpoint
    from fidelis.splat.views import load_views
    from fidelis.terrain.fetch import sha256_of

    scene = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    section = scene["splat"]
    views, heldout = load_views(args.views_dir, "train"), load_views(args.views_dir, "heldout")
    params = init_from_mesh(args.mesh_dir, config_suns(scene)[views.sun_index])
    print(f"{len(params['means'])} Gaussians from the mesh; {len(views.poses)} training views", flush=True)

    start = time.perf_counter()
    trained = train(
        params, torch.from_numpy(views.images), views.poses, views.intrinsics, TrainSettings.from_config(section)
    )
    train_s = time.perf_counter() - start

    with torch.no_grad():
        renders = [render(trained, p, heldout.intrinsics).cpu().numpy() for p in heldout.poses]
    report = quality_report(
        renders, [im.astype(np.float32) for im in heldout.images], QualityGate(**section["quality"])
    )
    for view, seed in zip(report["views"], heldout.seeds, strict=True):
        view["seed"] = seed

    meta = {
        "sun_index": views.sun_index,
        "sun": scene["suns"][views.sun_index],
        "steps": section["steps"],
        "gaussians": len(trained["means"]),
        "train_s": train_s,
        "gpu": torch.cuda.get_device_name(),
        "config_sha256": sha256_of(args.config),
    }
    save_checkpoint(args.out_dir / "splat.pt", trained, meta)
    export_ply(args.out_dir / "splat.ply", trained)
    path = args.out_dir / "quality.json"
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    tmp.write_text(json.dumps(meta | {"heldout": report}, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

    worst, baseline = report["worst"], report["constant_baseline"]
    print(
        f"held-out worst PSNR {worst['psnr_db']:.2f} dB, SSIM {worst['ssim']:.4f}"
        f" (constant image: {baseline['worst']['psnr_db']:.2f} dB, {baseline['worst']['ssim']:.4f},"
        f" {'rejected' if baseline['rejected'] else 'NOT rejected'});"
        f" gate {'PASSED' if report['passed'] else 'FAILED'}"
    )
    return 0
