from pathlib import Path

import cv2
import numpy as np

from common import debug
from common.vectorization import Contour

from .args import get_args
from .graph import Graph


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
    composited = image[:, :, :3] * alpha + 255.0 * (1.0 - alpha)
    return cv2.cvtColor(composited.astype(np.uint8), cv2.COLOR_BGR2GRAY)


def extract_ink_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(blurred, get_args("canny_low"), get_args("canny_high"))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    significant_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= get_args("min_area")
    ]
    debug.count("Canny contours", len(contours))
    debug.count("kept contours", len(significant_contours))

    borders = np.zeros_like(edges)
    cv2.drawContours(borders, significant_contours, -1, 255, thickness=1)
    border_neighbourhood = cv2.dilate(borders, np.ones((3, 3), np.uint8))

    _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(threshold)
    mask = np.zeros_like(threshold)
    for label in range(1, labels_count):
        component = labels == label
        if stats[label, cv2.CC_STAT_AREA] >= get_args("min_area") and np.any(
            border_neighbourhood[component]
        ):
            mask[component] = 255
    debug.count("ink pixels", int(np.count_nonzero(mask)))
    return mask, edges


def graph_to_contours(graph: Graph) -> list[Contour]:
    contours = []
    for edge in graph.edges:
        points = []
        for x, y in edge.pixels:
            point = (float(x), float(y))
            if not points or point != points[-1]:
                points.append(point)
        closed = edge.start.pos == edge.end.pos
        if closed and len(points) > 1 and points[0] == points[-1]:
            points.pop()
        contours.append(Contour(tuple(points), closed))
    return contours


__all__ = ["extract_ink_mask", "graph_to_contours", "read_image_in_gray_scale"]
