from pathlib import Path

import cv2
import numpy as np
from skimage.measure import find_contours

from common.vectorization import Contour


def read_image_in_gray_scale(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if image.ndim == 2:
        return image

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    alpha = image[:, :, 3:4].astype(np.float64) / 255.0
    composited = image[:, :, :3] * alpha
    composited += 255.0 * (1.0 - alpha)
    np.rint(composited, out=composited)
    return cv2.cvtColor(composited.astype(np.uint8), cv2.COLOR_BGR2GRAY)


def extract_contours(image: np.ndarray) -> list[Contour]:
    level, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    field = np.full((image.shape[0] + 2, image.shape[1] + 2), -1.0)
    np.subtract(level + 0.5, image, out=field[1:-1, 1:-1], dtype=np.float64)

    contours: list[Contour] = []
    for contour in find_contours(
        field,
        level=0.0,
        fully_connected="high",
        positive_orientation="high",
    ):
        points = tuple((x, y) for x, y in (contour[:, ::-1] - 0.5).tolist())
        if len(points) > 1 and points[0] == points[-1]:
            points = points[:-1]
        if len(points) >= 4:
            contours.append(Contour(points, closed=True))
    return contours


__all__ = ["extract_contours", "read_image_in_gray_scale"]
