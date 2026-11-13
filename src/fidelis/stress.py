"""The stress case (#43): suns beyond the realistic set, and the texture a run renders from.

A stress case is anything harsher, or other, than the scene the realistic failure rate is
measured on: a sun from the config's `stress_suns` list or the caller's own, or a texture
other than the photo. Its records and estimates are labelled so, and never pooled with
realistic ones (ADR 0003, decision 1; docs/benchmarks/stress-sun-scan.md).

The sun difference is the angle between the two sun directions in 3D, as in
docs/research/ctx-and-hirise-sun-angles.md, section 6, and in JPL's field tests' 100 deg line.
"""

import math
from dataclasses import replace

from fidelis.render.mesh_render import PHOTO_TEXTURE, Sun, config_suns
from fidelis.scenario import Integers, ScenarioConfig

# Which list scenario.sun_index indexes: the config's realistic suns, its stress suns, or
# suns a caller passed in (a test's zenith sun, say), which are never realistic.
REALISTIC_SUNS = "suns"
STRESS_SUNS = "stress_suns"
CUSTOM_SUNS = "custom"
SUN_LISTS = (REALISTIC_SUNS, STRESS_SUNS, CUSTOM_SUNS)

# The tile textures in the workflow's mesh dir: the orthoimage as photographed, and with its
# baked terrain shading divided out (fidelis.terrain.deshade, #19).
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


def check_sun_list(sun_list: str) -> None:
    if sun_list not in SUN_LISTS:
        raise ValueError(f"unknown sun list {sun_list!r}; expected one of {SUN_LISTS}")


def suns_for(scene: dict, sun_list: str) -> list[Sun]:
    """The config's suns that a run's scenario.sun_index indexes: the realistic list, or the stress list."""
    if sun_list == STRESS_SUNS:
        return stress_suns(scene)
    if sun_list == REALISTIC_SUNS:
        return config_suns(scene)
    raise ValueError(f"sun list {sun_list!r} isn't in the config; it has {REALISTIC_SUNS!r} and {STRESS_SUNS!r}")


def pin_stress_sun(config: ScenarioConfig, index: int, scene: dict) -> ScenarioConfig:
    """The config with every draw's sun_index set to stress sun `index`; a seed's other inputs don't change.

    The draws' sun_index then indexes `stress_suns`, so the simulator must render from that list.
    """
    count = len(scene[STRESS_SUNS])
    if not 0 <= index < count:
        raise ValueError(f"stress sun {index}; the config has {count} (0 to {count - 1})")
    return replace(config, sun_index=Integers(index, index))


def check_pinned_sun(texture: str, sun_index: int | None, stress_sun: int | None) -> None:
    """Refuse, before anything renders, a run off the photo texture whose sun isn't pinned.

    Such a run is a stress case, and a stress estimate holds one sun: unpinned, its draws
    span all the realistic suns, and `estimate` would refuse them only after every render.
    """
    if texture != PHOTO_TEXTURE and sun_index is None and stress_sun is None:
        raise ValueError(f"a run on {texture} is a stress case: pin its sun with --sun-index or --stress-sun")


def pinned_sun(scene: dict, sun_index: int | None, stress_sun: int | None) -> dict | None:
    """The one sun a pinned run used, with its angle from the map's sun to 0.1 deg; None if unpinned."""
    if stress_sun is not None:
        sun_list, index = STRESS_SUNS, stress_sun
    elif sun_index is not None:
        sun_list, index = REALISTIC_SUNS, sun_index
    else:
        return None
    entry = scene[sun_list][index]
    sun = Sun(entry["azimuth_deg"], entry["elevation_deg"])
    return {
        "sun_list": sun_list,
        "index": index,
        "azimuth_deg": entry["azimuth_deg"],
        "elevation_deg": entry["elevation_deg"],
        "sun_difference_deg": round(sun_difference_deg(sun, ctx_map_sun(scene)), 1),
    }


def is_stress(sun_list: str | None, texture: str | None) -> bool:
    """True unless the realistic suns lit the photo texture. None (labels from before #43) means those."""
    return (sun_list or REALISTIC_SUNS) != REALISTIC_SUNS or (texture or PHOTO_TEXTURE) != PHOTO_TEXTURE


def is_stress_case(record) -> bool:
    """A record lit by any sun but the realistic ones, or rendered from any texture but the photo."""
    return is_stress(record.sun_list, record.texture)


def pool_key(record) -> tuple:
    """What records in one estimate must share: the sun list and texture, and for a stress case the sun too.

    Realistic records pool across suns, as the sampler draws them; a stress estimate is one dose step.
    """
    sun = record.scenario.get("sun_index") if is_stress_case(record) else None
    return (record.sun_list or REALISTIC_SUNS, record.texture or PHOTO_TEXTURE, sun)
