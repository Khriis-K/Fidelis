"""Pinning the sun: while only some suns have a splat scene (#9 trains one), a run can hold
sun_index at one value without changing anything else a seed draws."""

from dataclasses import asdict
from pathlib import Path

import pytest

from fidelis.scenario import FIELDS, draw, load_config, pin_sun

REPO = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO / "config" / "jezero.yaml")


@pytest.mark.parametrize("sun_index", [0, 3])
def test_every_seed_draws_the_pinned_sun(sun_index):
    pinned = pin_sun(CONFIG, sun_index)
    assert {draw(pinned, seed).sun_index for seed in range(200)} == {sun_index}


def test_pinning_the_sun_leaves_every_other_input_of_a_seed_unchanged():
    pinned = pin_sun(CONFIG, 0)
    for seed in range(50):
        free, held = asdict(draw(CONFIG, seed)), asdict(draw(pinned, seed))
        assert {k: free[k] for k in FIELDS if k != "sun_index"} == {k: held[k] for k in FIELDS if k != "sun_index"}


@pytest.mark.parametrize("sun_index", [-1, 5])
def test_a_sun_the_config_cannot_draw_is_refused(sun_index):
    with pytest.raises(ValueError, match="sun"):
        pin_sun(CONFIG, sun_index)
