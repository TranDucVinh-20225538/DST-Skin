"""HED stain jitter for Camelyon train-time covariate simulation.

Ruifrok & Johnston H&E matrix; Tellez-style multiplicative jitter in HED.
No skimage / staintools (not in torch-env). Train only; eval unchanged.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# skimage.color.rgb_from_hed (Ruifrok and Johnston 2001)
RGB_FROM_HED = np.array(
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=np.float64,
)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)
EPS = 1e-6


class HEDJitter:
    """Multiply H/E/D channels by exp(N(0, sigma^2)). PIL in, PIL out."""

    def __init__(self, sigma: float = 0.20, p: float = 1.0):
        self.sigma = float(sigma)
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p < 1.0 and np.random.random() > self.p:
            return img
        rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        rgb = np.clip(rgb, EPS, 1.0)
        hed = (-np.log(rgb)) @ HED_FROM_RGB
        scale = np.exp(np.random.normal(0.0, self.sigma, size=(3,)))
        hed = hed * scale.reshape(1, 1, 3)
        out = np.exp(-(hed @ RGB_FROM_HED))
        out = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")


STAIN_COV_SIGMA = 0.20
STAIN_COV_P = 1.0
