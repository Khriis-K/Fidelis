"""Per-descent random inputs, each scenario reproducible from its seed.

A scenario is one simulated descent's operating conditions: where the camera truly is
(offset from the tile centre, altitude above the ground beneath it), how it's pointed
(tilt off nadir, the direction of that tilt, heading), how the image is degraded
(noise, motion blur, exposure), and which pre-rendered sun angle lights it.

Every input has its own random stream, seeded by (seed, a hash of the input's name), so
retuning one input's distribution leaves every other input of a seed unchanged. All
distributions are bounded, which lets `check_fits_tile` prove from the config alone that
no draw can see past the terrain tile. Distribution parameters live in the `scenario`
section of the scene config (config/jezero.yaml).

Pure Python and NumPy; no GPU.
"""

import math
import zlib
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import NormalDist

import numpy as np
import yaml


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float

    def __post_init__(self):
        if not self.low <= self.high:
            raise ValueError(f"uniform needs low <= high, got [{self.low}, {self.high}]")

    def bounds(self) -> tuple[float, float]:
        return self.low, self.high

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class TruncatedNormal:
    mean: float
    sd: float
    low: float
    high: float

    def __post_init__(self):
        if not self.sd > 0:
            raise ValueError(f"truncnorm needs sd > 0, got {self.sd}")
        if not self.low < self.high:
            raise ValueError(f"truncnorm needs low < high, got [{self.low}, {self.high}]")
        # Bounds so far out that the normal's CDF rounds to 0 or 1 would break the
        # inverse-CDF draw; such a config means a typo, so refuse it here.
        normal = NormalDist(self.mean, self.sd)
        if not (0 < normal.cdf(self.low) < normal.cdf(self.high) < 1):
            raise ValueError(
                f"truncnorm bounds [{self.low}, {self.high}] are too far into the tails of N({self.mean}, {self.sd})"
            )

    def bounds(self) -> tuple[float, float]:
        return self.low, self.high

    def sample(self, rng: np.random.Generator) -> float:
        # Inverse CDF: one uniform draw per sample, however narrow the bounds.
        normal = NormalDist(self.mean, self.sd)
        u = rng.uniform(normal.cdf(self.low), normal.cdf(self.high))
        return min(max(normal.inv_cdf(u), self.low), self.high)


@dataclass(frozen=True)
class Integers:
    low: int
    high: int  # inclusive

    def __post_init__(self):
        if not self.low <= self.high:
            raise ValueError(f"integers needs low <= high, got [{self.low}, {self.high}]")

    def bounds(self) -> tuple[int, int]:
        return self.low, self.high

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.low, self.high, endpoint=True))


Distribution = Uniform | TruncatedNormal | Integers

_KINDS = {
    "uniform": (Uniform, ("low", "high"), float),
    "truncnorm": (TruncatedNormal, ("mean", "sd", "low", "high"), float),
    "integers": (Integers, ("low", "high"), int),
}


def parse_distribution(spec: dict) -> Distribution:
    """A distribution from its config form, e.g. {dist: uniform, low: 0, high: 1}."""
    spec = dict(spec)
    kind = spec.pop("dist", None)
    if kind not in _KINDS:
        raise ValueError(f"unknown distribution {kind!r}; expected one of {sorted(_KINDS)}")
    cls, params, cast = _KINDS[kind]
    if set(spec) != set(params):
        raise ValueError(f"{kind} takes exactly {params}, got {sorted(spec)}")
    return cls(*(cast(spec[p]) for p in params))


@dataclass(frozen=True)
class Scenario:
    seed: int
    offset_east_m: float  # true horizontal position, ground metres from the tile centre
    offset_north_m: float
    altitude_m: float  # camera height above the ground beneath it
    tilt_deg: float  # optical axis off nadir
    tilt_azimuth_deg: float  # compass bearing the camera leans towards: 0 north, 90 east
    heading_deg: float  # compass bearing of the image's top edge
    noise_sd: float  # additive Gaussian noise, as a fraction of full scale
    blur_px: float  # linear motion-blur length, pixels
    blur_angle_deg: float  # direction of that blur in the image
    exposure_gain: float  # brightness multiplier on the rendered image
    sun_index: int  # which pre-rendered sun angle lights the scene


FIELDS = tuple(f.name for f in fields(Scenario) if f.name != "seed")


@dataclass(frozen=True)
class ScenarioConfig:
    offset_east_m: Distribution
    offset_north_m: Distribution
    altitude_m: Distribution
    tilt_deg: Distribution
    tilt_azimuth_deg: Distribution
    heading_deg: Distribution
    noise_sd: Distribution
    blur_px: Distribution
    blur_angle_deg: Distribution
    exposure_gain: Distribution
    sun_index: Integers

    @classmethod
    def from_dict(cls, section: dict) -> "ScenarioConfig":
        missing, unknown = set(FIELDS) - set(section), set(section) - set(FIELDS)
        if missing or unknown:
            raise ValueError(f"scenario inputs: missing {sorted(missing)}, unknown {sorted(unknown)}")
        config = cls(**{name: parse_distribution(section[name]) for name in FIELDS})
        if not isinstance(config.sun_index, Integers):
            raise ValueError("sun_index must use dist: integers, since it picks one of the sun angles")
        return config


def draw(config: ScenarioConfig, seed: int) -> Scenario:
    """The scenario for this seed. The same seed and config always give the same scenario."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    values = {}
    for name in FIELDS:
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        values[name] = getattr(config, name).sample(rng)
    return Scenario(seed=seed, **values)


def max_ground_reach_m(altitude_m: float, tilt_deg: float, fov_deg: float) -> float:
    """Farthest horizontal distance, over flat ground, that a square view can see.

    The image corner is the ray farthest off the optical axis; tilting can add the whole
    tilt to it, whichever way the camera leans or turns.
    """
    half = math.radians(fov_deg) / 2
    corner = math.atan(math.sqrt(2) * math.tan(half))
    angle = corner + math.radians(tilt_deg)
    if angle >= math.pi / 2:
        raise ValueError(f"a {fov_deg} deg view tilted {tilt_deg} deg reaches the horizon")
    return altitude_m * math.tan(angle)


def check_fits_tile(
    config: ScenarioConfig, half_width_east_m: float, half_width_north_m: float, fov_deg: float, relief_m: float
) -> None:
    """Refuse a config whose worst-case draw could see past the tile edge.

    Worst case: the highest altitude, plus the tile's full relief (the ground at the view's
    edge may be that much lower than beneath the camera), at the largest tilt, from the
    largest offset. Seeing past the tile would show missing terrain, a failure caused by
    the map running out rather than by navigation.
    """
    reach = max_ground_reach_m(config.altitude_m.bounds()[1] + relief_m, config.tilt_deg.bounds()[1], fov_deg)
    for axis, offset, half_width in (
        ("east", config.offset_east_m, half_width_east_m),
        ("north", config.offset_north_m, half_width_north_m),
    ):
        worst = max(abs(b) for b in offset.bounds()) + reach
        if worst > half_width:
            raise ValueError(
                f"a draw can see {worst:.0f} m {axis} of the tile centre, past the tile's {half_width:.0f} m;"
                " narrow the altitude, tilt or offset"
            )


def load_config(path: Path) -> ScenarioConfig:
    """The scene config's scenario distributions, checked to keep every view on the tile."""
    scene = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    section = dict(scene["scenario"])
    fov_deg = section.pop("camera_fov_deg")
    relief_m = section.pop("terrain_relief_m")
    config = ScenarioConfig.from_dict(section)
    # The tile's nominal size, before it's grown to the pixel grid: a slight underestimate.
    tile = scene["tile"]
    half_m = tile["size_m"] / 2
    check_fits_tile(
        config,
        half_width_east_m=half_m * math.cos(math.radians(tile["center_lat"])),
        half_width_north_m=half_m,
        fov_deg=fov_deg,
        relief_m=relief_m,
    )
    return config
