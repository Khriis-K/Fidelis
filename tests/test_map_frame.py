from pathlib import Path

import pytest
import yaml

from fidelis.map_frame import MapOffset, load_map_offset, map_frame_error

REPO = Path(__file__).resolve().parents[1]


def test_the_map_frame_error_removes_the_offset_from_east_and_north():
    # The fix comes out in map coordinates: a perfect navigator reports truth + offset.
    offset = MapOffset(east_m=30.0, north_m=-20.0, source="test")
    assert map_frame_error([35.0, -18.0, 1.5], offset) == pytest.approx([5.0, 2.0, 1.5])


def test_the_offset_is_horizontal_so_the_height_error_is_untouched():
    offset = MapOffset(east_m=0.0, north_m=0.0, source="test")
    assert map_frame_error([0.0, 0.0, 7.0], offset) == pytest.approx([0.0, 0.0, 7.0])


def test_a_zero_offset_leaves_the_error_as_it_was():
    assert map_frame_error([3.0, -4.0, 1.0], MapOffset(0.0, 0.0, "test")) == pytest.approx([3.0, -4.0, 1.0])


def test_the_offset_is_read_from_the_scene_config_with_its_source():
    scene = {"map_offset": {"east_m": 12.5, "north_m": -3.0, "source": "registered CTX to HiRISE (#40)"}}
    assert load_map_offset(scene) == MapOffset(12.5, -3.0, "registered CTX to HiRISE (#40)")


def test_an_offset_without_a_source_is_refused():
    # ADR 0003: the offset is measured independently and reported; an unsourced number isn't one.
    with pytest.raises(ValueError, match="source"):
        load_map_offset({"map_offset": {"east_m": 1.0, "north_m": 2.0, "source": "  "}})


def test_the_jezero_config_has_a_zero_offset_with_a_source_until_the_ctx_map_arrives():
    # Exact today: the matching map is made from the same orthoimage the scene renders (#36 changes that).
    scene = yaml.safe_load((REPO / "config" / "jezero.yaml").read_text(encoding="utf-8"))
    offset = load_map_offset(scene)
    assert (offset.east_m, offset.north_m) == (0.0, 0.0)
    assert offset.source.strip()
