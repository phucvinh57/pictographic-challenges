from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import medial_axis


@dataclass(frozen=True)
class MedialAxis:
    axis: np.ndarray
    distance: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        height, width = self.axis.shape
        return int(height), int(width)


def skeletonize(mask: np.ndarray) -> MedialAxis:
    axis, distance = medial_axis(mask > 0, return_distance=True, rng=0)
    return MedialAxis(
        axis=np.asarray(axis, dtype=bool),
        distance=np.asarray(distance, dtype=np.float64),
    )


def rasterize(polylines: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    canvas = np.full(shape, 255, dtype=np.uint8)
    for points in polylines:
        pixels = np.round(points).astype(np.int32)
        if len(pixels) < 2:
            x, y = pixels[0]
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                canvas[y, x] = 0
            continue
        cv2.polylines(canvas, [pixels], False, 0, 1, cv2.LINE_8)
    return canvas


__all__ = ["MedialAxis", "rasterize", "skeletonize"]
