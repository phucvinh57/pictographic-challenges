from pathlib import Path

import cv2
from cv2.typing import MatLike
from skimage.measure import find_contours

from .types import Contour
import numpy as np


def _convert_to_grayscale(image: MatLike) -> np.ndarray:
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


def extract_contours(image_path: Path) -> tuple[Contour, ...]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    gray_image = _convert_to_grayscale(image)
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    field = np.pad(
        (thresh + 0.5) - gray_image.astype(np.float64),
        1,
        constant_values=-1.0,
    )

    contours: list[Contour] = []
    for contour in find_contours(
        field,
        level=0.0,
        fully_connected="high",
        positive_orientation="high",
    ):
        # scikit-image returns (row, column); the SVG pipeline uses (x, y).
        # The 0.5 offset is to center the contour on the pixel grid.
        points = tuple(
            (float(column - 0.5), float(row - 0.5)) for row, column in contour
        )
        if len(points) > 1 and points[0] == points[-1]:
            points = points[:-1]
        if len(points) >= 4:
            contours.append(points)
    return tuple(contours)

def limit_penalties(contour: Contour) -> Contour:
    """
    To reduce a closed contour to a smaller set of important points,
    while preserving points where the contour has significant curvature/change.
    """
    if len(contour) <= 3:
        return contour
    return ()
    # Calculate the angles between consecutive segments
