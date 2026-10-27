from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    index: int
    time_s: float
    image: np.ndarray


@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class TrackedBox:
    track_id: int
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackSample:
    time_s: float
    track_id: int
    x_m: float
    y_m: float


@dataclass(frozen=True)
class PlayerSample:
    time_s: float
    player: str  # "A" or "B"
    x_m: float
    y_m: float


@dataclass(frozen=True)
class CourtResult:
    homography: np.ndarray  # 3x3, pixels -> court coordinates (meters)
    confidence: float
