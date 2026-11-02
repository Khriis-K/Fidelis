"""From rendered radiance to what the descent camera records: exposure, motion blur, noise.

In that order, as in the camera: light smears across pixels during the exposure, the
sensor reads it out with noise, and the result clips to full scale.

Exposure. LCAM had no automatic exposure: it was set beforehand from radiometric
predictions, and flight images came out at about 0.6x their target
(docs/research/edl-dispersions-for-scenario-sampler.md). The renderer's absolute
radiance isn't physical (the orthoimage is relative brightness, docs/benchmarks/mesh-render.md),
so the nominal exposure is set from the image itself: its 99.9th percentile reaches full
scale, standing in for a prediction that was right. The scenario's exposure gain is then
how far off the prediction was.

Noise is additive Gaussian, a fraction of full scale. Its random stream comes from the
scenario's seed, under a name of its own, so it repeats with the seed without touching
the sampler's streams.
"""

import math
import zlib

import cv2
import numpy as np

from fidelis.scenario import Scenario

EXPOSURE_PERCENTILE = 99.9
# Shifted copies averaged per pixel of blur length: enough that a sub-pixel blur is smooth.
BLUR_TAPS_PER_PX = 8


def expose(radiance: np.ndarray, gain: float) -> np.ndarray:
    """Radiance scaled so its 99.9th percentile sits at gain x full scale. Not clipped."""
    reference = float(np.percentile(radiance, EXPOSURE_PERCENTILE))
    if reference <= 0:
        return np.zeros_like(radiance, dtype=np.float32)
    return (radiance * (gain / reference)).astype(np.float32)


def motion_blur(image: np.ndarray, length_px: float, angle_deg: float) -> np.ndarray:
    """A straight-line blur `length_px` long, centred on each pixel.

    `angle_deg` is the blur's direction in the image: 0 along x (columns), 90 along y (rows, down).
    """
    if length_px == 0:
        return image
    taps = max(2, math.ceil(length_px * BLUR_TAPS_PER_PX) + 1)
    direction = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
    height, width = image.shape
    out = np.zeros(image.shape, np.float64)
    for t in np.linspace(-length_px / 2, length_px / 2, taps):
        dx, dy = t * direction
        shift = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        out += cv2.warpAffine(image, shift, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return (out / taps).astype(image.dtype)


def add_noise(image: np.ndarray, sd: float, rng: np.random.Generator) -> np.ndarray:
    return (image + rng.normal(0.0, sd, image.shape)).astype(np.float32)


def camera_image(radiance: np.ndarray, scenario: Scenario) -> np.ndarray:
    """The descent image the navigator gets: float32, in [0, 1]."""
    rng = np.random.default_rng([scenario.seed, zlib.crc32(b"camera_noise")])
    image = motion_blur(radiance.astype(np.float32), scenario.blur_px, scenario.blur_angle_deg)
    image = add_noise(expose(image, scenario.exposure_gain), scenario.noise_sd, rng)
    return np.clip(image, 0, 1)
