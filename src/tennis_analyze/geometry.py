import numpy as np


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map (N, 2) points through a 3x3 homography, returning (N, 2)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    mapped = homogeneous @ np.asarray(h, dtype=float).T
    return mapped[:, :2] / mapped[:, 2:3]


def foot_point(box_xyxy) -> tuple[float, float]:
    """Bottom-center of an xyxy box."""
    x1, _, x2, y2 = box_xyxy
    return ((x1 + x2) / 2, y2)
