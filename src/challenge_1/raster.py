from random import randint

import cv2
import numpy as np


def grayscale_on_white(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV image to grayscale, compositing alpha onto white."""
    if image.ndim == 2:
        return image

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    color = image[:, :, :3].astype(np.float64)
    alpha = image[:, :, 3:4].astype(np.float64) / 255.0
    composited = color * alpha + 255.0 * (1.0 - alpha)
    composited = np.rint(composited).astype(np.uint8)
    return cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)


def threshold_level(gray: np.ndarray, threshold: int | None) -> float:
    flags = cv2.THRESH_BINARY
    if threshold is None:
        threshold = 0
        flags |= cv2.THRESH_OTSU
    level, _ = cv2.threshold(gray, threshold, 255, flags)
    return float(level)


def random_contour_colors(count: int) -> tuple[str, ...]:
    colors = []
    used = set()
    while len(colors) < count:
        red, green, blue = (randint(32, 223) for _ in range(3))
        if max(red, green, blue) - min(red, green, blue) < 64:
            continue
        color = f"#{red:02x}{green:02x}{blue:02x}"
        if color in used:
            continue
        used.add(color)
        colors.append(color)
    return tuple(colors)
