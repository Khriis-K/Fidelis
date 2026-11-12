"""The real LVS map (#36, ADR 0003 decision 6.1): the USGS CTX orthomosaic, cut to the scene.

USGS calls its Mars 2020 TRN CTX orthomosaic "the Lander Vision System (LVS) map". Until
#36 the navigator matched against a block average of the HiRISE orthoimage the scene
renders (`fidelis.terrain.matching_map`), so map and descent image agreed on every rock
and shadow. Here the map is the CTX mosaic's own pixels under the scene's footprint:
different imagery, dates and lighting from what the camera sees, as LVS had.

Nothing HiRISE-derived goes in. The map is not resampled onto the scene's pixel grid
(LVS's map was "not enhanced, colour-balanced or smoothed"; Cheng et al. 2021, Table 1),
and not brightness-matched: the scene's ring texture (`ring_ortho.tif`) is CTX scaled to
HiRISE's mean and spread, so it isn't used either. The builder is given only the CTX
mosaic, the ring's bounds and the east-west ground scale.

    uv run fidelis-lvs-map CONFIG SCENE_DIR CTX_ORTHO OUT_TIF
"""

import argparse
import json
from pathlib import Path

import rasterio
import yaml

from fidelis.terrain.matching_map import require_same_resolution
from fidelis.terrain.scene import write_json
from fidelis.terrain.tile import Bounds, cut_tile


def build_lvs_map(ctx_ortho: Path, bounds: Bounds, dst_path: Path) -> dict:
    """Copy the CTX mosaic's pixels under bounds, which must be whole CTX pixels; return what was built."""
    info = cut_tile(ctx_ortho, bounds, dst_path)
    if info["nodata_pixels"]:
        Path(dst_path).unlink()
        raise ValueError(f"{ctx_ortho} has {info['nodata_pixels']} nodata pixels under {bounds}")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("scene_dir", type=Path, help="holds ring.json (the footprint) and tile.json (the scale)")
    parser.add_argument("ctx_ortho", type=Path, help="the raw USGS CTX orthomosaic")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    lvs_resolution = yaml.safe_load(args.config.read_text(encoding="utf-8"))["matching_map"]["lvs_map_resolution"]
    lvs_m = lvs_resolution["value_m"]
    with rasterio.open(args.ctx_ortho) as src:
        require_same_resolution(lvs_m, src.res[0])
    bounds = tuple(json.loads((args.scene_dir / "ring.json").read_text())["bounds"])
    info = build_lvs_map(args.ctx_ortho, bounds, args.out)
    info["lvs_map_resolution"] = lvs_resolution
    # Pixels are square in map metres, not on the ground (see fidelis.terrain.matching_map).
    scale = json.loads((args.scene_dir / "tile.json").read_text())["east_west_ground_scale"]
    info["east_west_ground_scale"] = scale
    info["ground_resolution_m"] = {"east_west": info["resolution_m"] * scale, "north_south": info["resolution_m"]}
    write_json(args.out.with_suffix(".json"), info)
    print(f"{args.out}: {info['shape']} at {info['resolution_m']} m/pixel, CTX's own pixels")
    return 0
