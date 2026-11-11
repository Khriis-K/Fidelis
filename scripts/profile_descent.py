"""Where a splat descent's time goes, stage by stage: the per-stage timing #10 needs first.

The splat Monte Carlo run cost 0.184 s per descent against the mesh's 0.234 s
(docs/benchmarks/splat.md), so swapping the renderer saved about 0.05 s and the other
~0.18 s looked like the camera model and the navigator. Cheaper splat levels can only cut
the render's share, so this splits `Simulator.run`'s timed span into its stages:

    render -> camera model -> nadir warp -> ALIKED extract -> LightGlue match -> PnP

It times the real code path: each stage's function is wrapped in place, with a CUDA
synchronize on both sides so GPU work lands in the stage that launched it (which the
syncs themselves slightly inflate). "other" is the record's cost_s minus the stages:
glue such as tensor uploads and moving keypoints back to camera pixels.

Seed 1's descent is run once untimed first, so gsplat's extension load and the GPU's
warm-up don't count. Every descent is under sun 0, the one trained splat.

Takes the splat as its .pt checkpoint, or as the browser PLY export, which holds the
same parameters (colours and quaternions round-trip to float precision; the renderer
normalises quaternions anyway).

    uv run python scripts/profile_descent.py config/jezero.yaml data/jezero [--samples 30]
"""

import argparse
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch

import fidelis.nav.navigator as navigator_module
import fidelis.simulation as simulation_module
from fidelis.render.splat_render import SplatRenderer
from fidelis.scenario import draw, load_config, pin_sun
from fidelis.simulation import build_simulator
from fidelis.splat.model import PLY_PROPERTIES, SH_C0, save_checkpoint

STAGES = ("render", "camera", "warp", "extract", "match", "pnp")


def params_from_ply(path: Path) -> dict[str, torch.Tensor]:
    """The inverse of fidelis.splat.model.export_ply."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    names = [line.split()[-1] for line in header if line.startswith("property")]
    if names != PLY_PROPERTIES:
        raise ValueError(f"{path} isn't a Fidelis splat export: properties {names}")
    values = np.frombuffer(data, dtype="<f4", offset=end).reshape(-1, len(names))
    column = {name: i for i, name in enumerate(names)}

    def cols(*keys):
        return torch.from_numpy(values[:, [column[k] for k in keys]].copy())

    return {
        "means": cols("x", "y", "z"),
        "scales": cols("scale_0", "scale_1", "scale_2"),
        "quats": cols("rot_0", "rot_1", "rot_2", "rot_3"),
        "opacities": cols("opacity")[:, 0],
        "colors": cols("f_dc_0") * SH_C0 + 0.5,
    }


class Clock:
    def __init__(self):
        self.times: dict[str, float] = {}

    def wrap(self, stage: str, fn):
        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            self.times[stage] = self.times.get(stage, 0.0) + time.perf_counter() - start
            return out

        return timed


class TimedRenderer:
    def __init__(self, inner, clock: Clock):
        self.use_sun = inner.use_sun
        self._call = clock.wrap("render", inner)

    def __call__(self, *args):
        return self._call(*args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--samples", type=int, default=30, help="timed descents, seeds 1..samples")
    parser.add_argument("--out", type=Path, help="write per-seed stage times here as JSON")
    args = parser.parse_args()

    sun_dir = args.data_dir / "splat" / "sun0"
    checkpoint = sun_dir / "splat.pt"
    if not checkpoint.exists():
        checkpoint = args.data_dir / "splat" / "profile_from_ply" / "splat.pt"
        if not checkpoint.exists():
            params = params_from_ply(sun_dir / "splat.ply")
            save_checkpoint(checkpoint, params, {"from": "splat.ply, by scripts/profile_descent.py"})
    gaussians = len(torch.load(checkpoint, weights_only=True)["params"]["means"])

    simulator = build_simulator(args.config, args.data_dir, fidelities=())
    clock = Clock()
    simulator.renderers["splat"] = TimedRenderer(SplatRenderer({0: checkpoint}), clock)
    simulation_module.camera_image = clock.wrap("camera", simulation_module.camera_image)
    navigator_module.warp_to_nadir = clock.wrap("warp", navigator_module.warp_to_nadir)
    navigator_module.solve_pose = clock.wrap("pnp", navigator_module.solve_pose)
    nav = simulator.navigator
    nav.extractor.extract = clock.wrap("extract", nav.extractor.extract)
    nav.matcher = clock.wrap("match", nav.matcher)

    config = pin_sun(load_config(args.config), 0)
    simulator.run(draw(config, 1), "splat")  # warm-up, untimed

    rows = []
    for seed in range(1, args.samples + 1):
        clock.times = {}
        record = simulator.run(draw(config, seed), "splat").record
        row = {"seed": seed, "cost_s": record.cost_s, "status": record.status, **clock.times}
        row["other"] = record.cost_s - sum(clock.times[s] for s in STAGES)
        rows.append(row)

    total = median(r["cost_s"] for r in rows)
    print(f"{simulator.gpu}, {gaussians:,} Gaussians, {len(rows)} descents (seeds 1-{args.samples}), sun 0")
    print(f"{'stage':<8} {'median s':>9} {'share of median cost':>21}")
    for stage in (*STAGES, "other"):
        m = median(r[stage] for r in rows)
        print(f"{stage:<8} {m:>9.4f} {m / total:>20.0%}")
    print(f"{'cost_s':<8} {total:>9.4f}")
    print(f"statuses: {sorted({r['status'] for r in rows})}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"gpu": simulator.gpu, "gaussians": gaussians, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
