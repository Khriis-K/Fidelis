"""The map frame (ADR 0003): a fix's error measured against the matching map, not the true scene.

The navigator's fix comes out in the matching map's coordinates. If the whole map sits
shifted from the rendered scene, a perfect navigator still reports the truth plus that
shift. LVS's map could be off by 150-300 m without harm, because its hazard map shared the
same frame (docs/research/lvs-error-sources.md). So the shift is removed before failures
are counted, and reported on its own.

The shift is measured once, independently, by registering the two orbital images (#40),
and recorded in the scene config's `map_offset`. It is never estimated from descent
results: subtracting the descents' mean error would also hide the navigator's own bias.
Local map distortion is not removed; it is part of the fix's real error.

The offset belongs to a map. The config's `matching_map.source` names the map the navigator
matches against: "ctx", the real LVS map (#36), whose offset is `map_offset`; or "scene",
the old map made from the orthoimage the scene renders, whose offset is zero by
construction. Asking for a map gives its offset with it, so the two can't be mismatched.

Convention: the offset is where the map puts a ground feature, less where the scene has
it, in metres east and north. It is horizontal, since image registration measures no height.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MapOffset:
    east_m: float
    north_m: float
    source: str  # how it was measured, or why it is zero


MATCHING_MAPS = ("ctx", "scene")

SCENE_MAP_OFFSET = MapOffset(
    0.0,
    0.0,
    "Zero by construction: the scene's own map is a block average of the same orthoimage "
    "the scene renders, so the two share one frame exactly.",
)


def checked_matching_map(source: str) -> str:
    """`source` if it names a matching map, else a ValueError."""
    if source not in MATCHING_MAPS:
        raise ValueError(f"unknown matching map {source!r}; expected one of {MATCHING_MAPS}")
    return source


def matching_map_source(scene: dict) -> str:
    """Which map the scene config has the navigator match against: "ctx" or "scene"."""
    return checked_matching_map(scene["matching_map"]["source"])


def load_map_offset(scene: dict, source: str | None = None) -> MapOffset:
    """The whole-map offset of a matching map: `source`, or else the scene config's own choice.

    The CTX map's is the config's `map_offset`, which must say where its numbers come
    from; the scene's own map's is zero by construction. A config that names no map has
    only its `map_offset`.
    """
    if source is None and "source" in scene.get("matching_map", {}):
        source = matching_map_source(scene)
    if source is not None and checked_matching_map(source) == "scene":
        return SCENE_MAP_OFFSET
    entry = scene["map_offset"]
    source = str(entry.get("source") or "").strip()
    if not source:
        raise ValueError("map_offset needs a source: how it was measured, or why it is zero")
    return MapOffset(float(entry["east_m"]), float(entry["north_m"]), source)


def map_frame_error(error_vector: list[float], offset: MapOffset) -> list[float]:
    """A true-frame error vector (estimate minus truth, east-north-up) with the whole-map shift removed."""
    east, north, up = error_vector
    return [east - offset.east_m, north - offset.north_m, up]
