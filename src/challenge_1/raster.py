from random import randint

import cv2
import numpy as np

from pictographic.curves import AxisPoint


def binarize(gray: np.ndarray, threshold: int | None) -> np.ndarray:
    threshold_type = cv2.THRESH_BINARY
    if threshold is None:
        threshold = 0
        threshold_type += cv2.THRESH_OTSU
    _, binary = cv2.threshold(gray, threshold, 255, threshold_type)
    return binary


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


def _bgr(color: str) -> tuple[int, int, int]:
    return int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16)


def draw_contours(
    shape: tuple[int, int],
    contours: tuple[tuple[AxisPoint, ...], ...],
    colors: tuple[str, ...],
) -> np.ndarray:
    image = np.full((*shape, 3), 255, dtype=np.uint8)
    for contour, color in zip(contours, colors, strict=True):
        points = np.rint(np.asarray(contour) * 256).astype(np.int32)
        cv2.polylines(
            image,
            [points],
            True,
            _bgr(color),
            1,
            cv2.LINE_AA,
            shift=8,
        )
    return image
