"""Synthetic court images for tests: a rear, elevated fixed camera on a 1920x1080 frame."""

import cv2
import numpy as np

from tennis_analyze.court_model import COURT_LENGTH_M, COURT_WIDTH_M, court_lines

DOUBLES_ALLEY_M = 1.37

# Singles corners in pixels: near-left, near-right, far-right, far-left.
# Near baseline is wide and low in the frame, far baseline narrow and high.
_BASE_CORNERS_PX = np.array(
    [[560.0, 960.0], [1360.0, 960.0], [1130.0, 290.0], [790.0, 290.0]], np.float32
)
_CORNERS_M = np.array(
    [
        [0.0, 0.0],
        [COURT_WIDTH_M, 0.0],
        [COURT_WIDTH_M, COURT_LENGTH_M],
        [0.0, COURT_LENGTH_M],
    ],
    np.float32,
)


def make_homography(seed: int) -> np.ndarray:
    """Court meters -> pixels for a plausible rear/elevated view; seed jitters the corners."""
    rng = np.random.default_rng(seed)
    corners_px = _BASE_CORNERS_PX + rng.uniform(-40.0, 40.0, size=(4, 2)).astype(
        np.float32
    )
    return cv2.getPerspectiveTransform(_CORNERS_M, corners_px).astype(float)


def _project(h: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, h).reshape(-1, 2)


def render_court(
    h_court_to_px: np.ndarray, doubles_lines: bool = False, size=(1080, 1920)
) -> np.ndarray:
    """Green BGR frame with white court lines drawn through h_court_to_px."""
    image = np.empty((size[0], size[1], 3), np.uint8)
    image[:] = (60, 140, 60)
    segments = list(court_lines())
    if doubles_lines:
        a, w, length = DOUBLES_ALLEY_M, COURT_WIDTH_M, COURT_LENGTH_M
        segments += [
            ((-a, 0.0), (-a, length)),
            ((w + a, 0.0), (w + a, length)),
            # Baselines run out to the doubles sidelines, as on a real court.
            ((-a, 0.0), (w + a, 0.0)),
            ((-a, length), (w + a, length)),
        ]
    shift = 4  # sub-pixel drawing so the line centers match the homography exactly
    for p, q in segments:
        (x1, y1), (x2, y2) = _project(h_court_to_px, [p, q])
        cv2.line(
            image,
            (round(x1 * 2**shift), round(y1 * 2**shift)),
            (round(x2 * 2**shift), round(y2 * 2**shift)),
            (255, 255, 255),
            3,
            cv2.LINE_AA,
            shift,
        )
    return image
