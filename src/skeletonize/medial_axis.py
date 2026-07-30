from __future__ import annotations

import numpy as np
from skimage.morphology import medial_axis


def extract_medial_axis(binary: np.ndarray) -> np.ndarray:
    """Return the original ink's one-pixel medial axis in project polarity."""
    axis = np.asarray(medial_axis(binary < 128, return_distance=False), dtype=bool)
    return np.where(axis, 0, 255).astype(np.uint8)
