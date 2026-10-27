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


# --- synthetic match video -------------------------------------------------------------

_MATCH_SEED = 3
_SAMPLE_FPS = 5  # the pipeline's default sampling rate


def _player_foot_m(player: int, t: float, seconds: float) -> tuple[float, float]:
    """Court position (m) of the near (0) or far (1) player at time t."""
    u = t / seconds
    if player == 0:
        return 2.0 + 4.0 * u, 3.0 + 3.0 * u
    return 6.0 - 3.0 * u, 20.0 - 3.0 * u


def _player_box_px(h: np.ndarray, foot_m: tuple[float, float]) -> tuple[float, float, float, float]:
    """Pixel box whose bottom-center is the projected foot; size follows local scale."""
    x, y = foot_m
    (fx, fy), (rx, ry) = _project(h, [(x, y), (x + 1.0, y)])
    meter_px = float(np.hypot(rx - fx, ry - fy))
    half_w, height = 0.3 * meter_px, 1.8 * meter_px
    return (fx - half_w, fy - height, fx + half_w, fy)


def write_synthetic_match(path, seconds: int = 6, fps: int = 30):
    """Write a court video with two white rectangles moving; return ground truth.

    Returns one list[Detection] per frame sampled at 5 fps (what the pipeline reads).
    """
    from tennis_analyze.types import Detection

    h = make_homography(_MATCH_SEED)
    background = render_court(h)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)
    )
    step = round(fps / _SAMPLE_FPS)
    truth = []
    try:
        for i in range(seconds * fps):
            t = i / fps
            frame = background.copy()
            dets = []
            for player in (0, 1):
                box = _player_box_px(h, _player_foot_m(player, t, seconds))
                x1, y1, x2, y2 = (round(v) for v in box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
                dets.append(Detection(box, 0.9))
            writer.write(frame)
            if i % step == 0:
                truth.append(dets)
    finally:
        writer.release()
    return truth


def write_blank_video(path, seconds: int = 2, fps: int = 30) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)
    )
    try:
        for _ in range(seconds * fps):
            writer.write(np.zeros((1080, 1920, 3), np.uint8))
    finally:
        writer.release()


class FakeDetector:
    """PersonDetector returning the given per-sampled-frame detections in order."""

    def __init__(self, per_frame):
        self._frames = iter(per_frame)

    def detect(self, image):
        return list(next(self._frames, []))
