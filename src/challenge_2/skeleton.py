import cv2
import numpy as np


def skeletonize(mask: np.ndarray) -> np.ndarray:
    skeleton = cv2.ximgproc.thinning(mask, None, cv2.ximgproc.THINNING_ZHANGSUEN)
    return cv2.bitwise_not(skeleton)


__all__ = ["skeletonize"]
