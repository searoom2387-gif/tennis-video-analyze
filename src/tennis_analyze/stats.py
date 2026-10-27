from dataclasses import dataclass

import numpy as np

from tennis_analyze.court_model import COURT_LENGTH_M, COURT_WIDTH_M
from tennis_analyze.types import PlayerSample

MIN_STEP_M = 0.1  # smoothed steps under 0.1 m per sample (~0.5 m/s) are jitter, not movement


@dataclass(frozen=True)
class PlayerStats:
    distance_m: float
    mean_speed_mps: float
    share_left: float
    share_right: float
    share_front: float
    share_back: float


def smooth_path(xy: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving median over x and y independently (edges padded by repeating end samples)."""
    xy = np.asarray(xy, dtype=float)
    half = window // 2
    padded = np.pad(xy, ((half, half), (0, 0)), mode="edge")
    out = np.empty_like(xy)
    for i in range(len(xy)):
        out[i] = np.median(padded[i : i + window], axis=0)
    return out


def _distance(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(smooth_path(xy), axis=0), axis=1)
    return float(steps[steps >= MIN_STEP_M].sum())


def compute_stats(samples: list[PlayerSample]) -> dict[str, PlayerStats]:
    result: dict[str, PlayerStats] = {}
    for player in ("A", "B"):
        own = sorted((s for s in samples if s.player == player), key=lambda s: s.time_s)
        if not own:
            continue
        xy = np.array([(s.x_m, s.y_m) for s in own])
        distance = _distance(xy)
        duration = own[-1].time_s - own[0].time_s
        speed = distance / duration if duration > 0 else 0.0
        left = float(np.mean(xy[:, 0] < COURT_WIDTH_M / 2))
        front = float(np.mean(xy[:, 1] < COURT_LENGTH_M / 2))
        result[player] = PlayerStats(distance, speed, left, 1.0 - left, front, 1.0 - front)
    return result


def heatmap(
    samples: list[PlayerSample], player: str, bins: tuple[int, int] = (24, 48)
) -> np.ndarray:
    pts = np.array([(s.x_m, s.y_m) for s in samples if s.player == player]).reshape(-1, 2)
    hist, _, _ = np.histogram2d(
        pts[:, 0], pts[:, 1], bins=bins, range=[[0, COURT_WIDTH_M], [0, COURT_LENGTH_M]]
    )
    return hist
