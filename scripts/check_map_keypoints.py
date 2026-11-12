"""How many ALIKED keypoints a matching map gives, per device and per map tile.

The records of the A40 runs count 70,749 keypoints on the scene's own map; the laptop's GPU
counts 70,726 on the same file (docs/benchmarks/ctx-map-scan.md). This extracts one map on
each device named, tile by tile as the navigator does, so the two can be compared with
nothing else changed. Extraction only: no matching, so it fits the laptop.

    uv run python scripts/check_map_keypoints.py data/jezero/matching_map/map.tif cuda cpu
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from lightglue import ALIKED

from fidelis.nav.navigator import MAP_TILE_PX, MAX_KEYPOINTS

path, devices = Path(sys.argv[1]), sys.argv[2:]
with rasterio.open(path) as src:
    image = src.read(1).astype(np.float32) / 255
print(path, image.shape)
for device in devices:
    extractor = ALIKED(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)
    counts = []
    with torch.inference_mode():
        for r0 in range(0, image.shape[0], MAP_TILE_PX):
            row = []
            for c0 in range(0, image.shape[1], MAP_TILE_PX):
                tile = np.ascontiguousarray(image[r0 : r0 + MAP_TILE_PX, c0 : c0 + MAP_TILE_PX])
                features = extractor.extract(torch.from_numpy(tile).to(device).expand(3, -1, -1), resize=None)
                row.append(features["keypoints"].shape[1])
            counts.append(row)
    name = torch.cuda.get_device_name() if device.startswith("cuda") else "cpu"
    print(f"{name}: {sum(map(sum, counts))} keypoints; per tile (capped at {MAX_KEYPOINTS}):")
    for row in counts:
        print("   ", row)
