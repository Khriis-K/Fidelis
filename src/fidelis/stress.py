"""The stress case (#43): suns beyond the realistic set, and the texture a run renders from.

A stress case is anything harsher, or other, than the scene the realistic failure rate is
measured on: a sun from the config's `stress_suns` list, or a texture other than the photo.
Its records and estimates are labelled so, and never pooled with realistic ones (ADR 0003,
decision 1; docs/benchmarks/stress-sun-scan.md).

The sun difference is the angle between the two sun directions in 3D, as in
docs/research/ctx-and-hirise-sun-angles.md, section 6, and in JPL's field tests' 100 deg line.
"""

import math
from dataclasses import replace

from fidelis.render.mesh_render import Sun, config_suns
from fidelis.scenario import Integers, ScenarioConfig

# The scene config's sun lists, by key: scenario.sun_index indexes one of them.
REALISTIC_SUNS = "suns"
STRESS_SUNS = "stress_suns"
SUN_LISTS = (REALISTIC_SUNS, STRESS_SUNS)

# The tile textures in the workflow's mesh dir: the orthoimage as photographed, and with its
# baked terrain shading divided out (fidelis.terrain.deshade, #19).
PHOTO_TEXTURE = "texture.png"
DESHADED_TEXTURE = "texture_deshaded.png"
TEXTURES = (PHOTO_TEXTURE, DESHADED_TEXTURE)


def sun_difference_deg(a: Sun, b: Sun) -> float:
    """The angle between two sun directions, degrees."""
    cos = sum(x * y for x, y in zip(a.direction(), b.direction(), strict=True))
    return math.degrees(math.acos(min(1.0, max(-1.0, cos))))


def ctx_map_sun(scene: dict) -> Sun:
    """The sun of the CTX image under the LVS map over the tile, from the scene config."""
    sun = scene["ctx_map_sun"]
    return Sun(sun["azimuth_deg"], sun["elevation_deg"])


def stress_suns(scene: dict) -> list[Sun]:
    """The scene config's stress suns, in index order."""
    return [Sun(s["azimuth_deg"], s["elevation_deg"]) for s in scene[STRESS_SUNS]]


def suns_for(scene: dict, sun_list: str) -> list[Sun]:
    """The suns a run's scenario.sun_index indexes: the realistic list, or the stress list."""
    if sun_list not in SUN_LISTS:
        raise ValueError(f"unknown sun list {sun_list!r}; expected one of {SUN_LISTS}")
    return stress_suns(scene) if sun_list == STRESS_SUNS else config_suns(scene)


def pin_stress_sun(config: ScenarioConfig, index: int, scene: dict) -> ScenarioConfig:
    """The config with every draw's sun_index set to stress sun `index`; a seed's other inputs don't change.

    The draws' sun_index then indexes `stress_suns`, so the simulator must render from that list.
    """
    count = len(scene[STRESS_SUNS])
    if not 0 <= index < count:
        raise ValueError(f"stress sun {index}; the config has {count} (0 to {count - 1})")
    return replace(config, sun_index=Integers(index, index))


def is_stress_case(record) -> bool:
    """True for a record lit by a stress sun or rendered from any texture but the photo.

    Records from before #43 carry neither label; they all used the realistic suns and the photo.
    """
    return (record.sun_list or REALISTIC_SUNS) != REALISTIC_SUNS or (record.texture or PHOTO_TEXTURE) != PHOTO_TEXTURE


def stress_label(record) -> tuple:
    """What an estimate must hold constant: the sun list and texture, and for a stress case the sun too.

    Realistic records pool across suns, as the sampler draws them; a stress estimate is one dose step.
    """
    sun = record.scenario.get("sun_index") if is_stress_case(record) else None
    return (record.sun_list or REALISTIC_SUNS, record.texture or PHOTO_TEXTURE, sun)
