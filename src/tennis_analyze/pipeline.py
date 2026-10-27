"""End-to-end pipeline: video -> court -> players -> stats -> report files."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from tennis_analyze.court_detect import (
    CourtDetectionError,
    court_from_corners,
    detect_court,
    render_court_check,
)
from tennis_analyze.players import (
    PersonDetector,
    PlayerTracker,
    select_players,
    stitch_tracks,
    to_track_samples,
)
from tennis_analyze.report import write_outputs
from tennis_analyze.stats import compute_stats
from tennis_analyze.video_io import VideoOpenError, iter_frames

COURT_FRAMES = 30


def _sampled_frame_count(video: Path, target_fps: float) -> int:
    """Number of frames iter_frames will yield (from container metadata, else by decoding)."""
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise VideoOpenError(f"動画を開けません: {video}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if total > 0 and src_fps and src_fps > 0:
        step = max(1, round(src_fps / target_fps))
        return (total + step - 1) // step
    return sum(1 for _ in iter_frames(video, target_fps))


def _court_frames(video: Path, target_fps: float) -> list[np.ndarray]:
    """Up to COURT_FRAMES frames spread evenly over the video (memory stays bounded)."""
    n = _sampled_frame_count(video, target_fps)
    wanted = {int(i) for i in np.linspace(0, max(n - 1, 0), min(COURT_FRAMES, n))}
    images = []
    for ordinal, frame in enumerate(iter_frames(video, target_fps)):
        if ordinal in wanted:
            images.append(frame.image)
        if ordinal >= max(wanted):
            break
    return images


def run(
    video: Path,
    out_dir: Path,
    detector: PersonDetector,
    corners: Sequence[tuple[float, float]] | None = None,
    target_fps: float = 5.0,
    pdf: bool = False,
) -> list[str]:
    """Analyze `video`, write outputs to `out_dir`, return warning messages.

    Raises VideoOpenError, or CourtDetectionError (after writing out_dir/court_check.png
    from the last frame tried).
    """
    video, out_dir = Path(video), Path(out_dir)
    images = _court_frames(video, target_fps)
    if corners is not None:
        court = court_from_corners(corners)
    else:
        try:
            court = detect_court(images)
        except CourtDetectionError:
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / "court_check.png"), images[-1])
            raise
    court_check = render_court_check(images[0], court)
    del images

    tracker = PlayerTracker()
    track_samples = []
    for frame in iter_frames(video, target_fps):
        boxes = tracker.update(detector.detect(frame.image))
        track_samples += to_track_samples(frame.time_s, boxes, court.homography)

    samples, warnings = select_players(stitch_tracks(track_samples))
    write_outputs(out_dir, samples, compute_stats(samples), court_check, warnings, pdf=pdf)
    return warnings
