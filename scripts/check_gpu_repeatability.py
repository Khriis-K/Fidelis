"""Why the same seed doesn't give the same error on the GPU: the numbers in docs/benchmarks/plain-mc.md.

Through the real seam (Jezero mesh, config suns, LCAM, 6 m map), for seed 10 and a few others:

1. Two processes render seed 10: how far apart are the renders?
2. One image, saved by the first process, located twice in each: does the navigator repeat,
   within a process and across them?
3. The same matched pairs solved five times: does PnP + RANSAC repeat?
4. That image with 1e-7 and 1e-6 noise added (four draws each), below and above the
   renders' jitter: how far does the error move?
5. Three runs of each seed in one process: camera-image difference and errors.
6. Mitsuba's samples_per_pass = 1 (one sample per pixel per launch): bit-exact across
   processes, and at what cost against the default, at 256 spp?

    uv run python scripts/check_gpu_repeatability.py data/jezero
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "jezero.yaml"
SEEDS = (10, 15, 0, 4)


def child(data: Path, out: Path) -> None:
    """One process's share: render and locate seed 10, save the arrays, time samples_per_pass."""
    import drjit as dr
    import mitsuba as mi
    import yaml

    from fidelis.nav.orbital_map import load_orbital_map
    from fidelis.nav.pnp import solve_pose
    from fidelis.render.camera import Intrinsics
    from fidelis.render.mesh_render import MAX_DEPTH, Sun, mitsuba_sensor, terrain_scene
    from fidelis.scenario import draw, load_config
    from fidelis.simulation import LCAM_PX, build_mesh_simulator, scenario_pose

    mi.set_variant("cuda_ad_rgb")
    scene_config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    s = draw(load_config(CONFIG), 10)
    ground = load_orbital_map(data / "matching_map" / "map.tif", data / "tile" / "dtm.tif", data / "mesh" / "mesh.json")
    pose = scenario_pose(s, ground.ground_height(s.offset_east_m, s.offset_north_m))
    intrinsics = Intrinsics.from_fov(LCAM_PX, LCAM_PX, scene_config["scenario"]["camera_fov_deg"])

    # Timed first and freed, so the simulator's own copy of the mesh doesn't share the GPU with it.
    scene = terrain_scene(data / "mesh", Sun(**scene_config["suns"][s.sun_index]))
    sensor = mi.load_dict(mitsuba_sensor(pose, intrinsics, 256))
    for per_pass in (None, 1):
        spec = {"type": "path", "max_depth": MAX_DEPTH} | ({"samples_per_pass": per_pass} if per_pass else {})
        integrator = mi.load_dict(spec)
        mi.render(scene, sensor=sensor, integrator=integrator, spp=256, seed=10)  # warm-up
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            image = np.array(mi.render(scene, sensor=sensor, integrator=integrator, spp=256, seed=10))
            times.append(time.perf_counter() - t0)
        np.save(out / f"radiance_per_pass_{per_pass}.npy", image[..., 0])
        if per_pass is None:
            print(f"  median radiance pixel {float(np.median(image)):.3f}", flush=True)
        print(f"  samples_per_pass={per_pass}: median {sorted(times)[1]:.3f} s at 256 spp", flush=True)

    del scene
    dr.flush_malloc_cache()

    sim = build_mesh_simulator(CONFIG, data)
    nav = sim.navigator
    run = sim.run(s, "mesh")
    np.save(out / "image.npy", run.image)
    # The first process saves its image as the fixed one; the second locates that same file.
    fixed_path = out.parent / "fixed_image.npy"
    if not fixed_path.exists():
        np.save(fixed_path, run.image)
    fixed = np.load(fixed_path)

    def error(result):
        return None if result.position is None else round(float(np.linalg.norm(result.position - pose.position)), 4)

    located, again = (nav.locate(fixed, sim.intrinsics, pose.rotation) for _ in range(2))
    same = np.array_equal(located.image_points, again.image_points) and np.array_equal(located.inliers, again.inliers)
    print(
        f"  fixed image located twice here: {error(located)} and {error(again)} m,"
        f" {located.inlier_count} inliers, identical matches {same}",
        flush=True,
    )
    world = nav.map.pixel_to_world(located.map_points)
    spread = np.ptp([solve_pose(located.image_points, world, sim.intrinsics).pose.position for _ in range(5)], axis=0)
    print(f"  same pairs solved 5x: spread {spread.max():.6f} m", flush=True)
    rng = np.random.default_rng(0)
    for sd in (1e-7, 1e-6):  # below and above the jitter the renders show
        nudged = [
            error(
                nav.locate(
                    np.clip(fixed + rng.normal(0, sd, fixed.shape).astype(np.float32), 0, 1),
                    sim.intrinsics,
                    pose.rotation,
                )
            )
            for _ in range(4)
        ]
        print(f"  fixed image + {sd:.0e} noise, four draws: {nudged} m", flush=True)


def in_process(data: Path) -> None:
    from fidelis.scenario import draw, load_config
    from fidelis.simulation import build_mesh_simulator

    sim = build_mesh_simulator(CONFIG, data)
    config = load_config(CONFIG)
    print("\nThree runs per seed in one process\n| Seed | Errors (m) | Max camera-image difference |\n|---|---|---|")
    for seed in SEEDS:
        runs = [sim.run(draw(config, seed), "mesh") for _ in range(3)]
        diff = max(float(np.abs(r.image - runs[0].image).max()) for r in runs)
        print(f"| {seed} | {', '.join(f'{r.record.error_m:.4f}' for r in runs)} | {diff:.2e} |", flush=True)


def main() -> None:
    data = Path(sys.argv[1])
    if len(sys.argv) == 4 and sys.argv[2] == "--child":
        child(data, Path(sys.argv[3]))
        return
    with tempfile.TemporaryDirectory() as tmp:
        outs = []
        for name in ("A", "B"):
            out = Path(tmp) / name
            out.mkdir()
            print(f"Process {name}", flush=True)
            subprocess.run([sys.executable, __file__, str(data), "--child", str(out)], check=True)
            outs.append(out)
        a, b = outs
        image_diff = np.abs(np.load(a / "image.npy") - np.load(b / "image.npy"))
        print(
            f"\nSeed 10's camera image, process A vs B: max difference {image_diff.max():.2e},"
            f" {int((image_diff > 0).sum())} of {image_diff.size} pixels differ"
        )
        for per_pass in (None, 1):
            ra, rb = (np.load(o / f"radiance_per_pass_{per_pass}.npy") for o in outs)
            print(
                f"samples_per_pass={per_pass} render, A vs B: max difference {np.abs(ra - rb).max():.2e},"
                f" bit-exact {np.array_equal(ra, rb)}"
            )
    in_process(data)


if __name__ == "__main__":
    main()
