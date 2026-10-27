from pathlib import Path

import cv2
import numpy as np
import pytest

from tennis_analyze.video_io import VideoOpenError, iter_frames


def _write_video(
    path: Path, n_frames: int, fps: float = 30.0, codec: str = "mp4v"
) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), fps, (64, 48)
    )
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((48, 64, 3), i % 256, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def video_30fps_3s(tmp_path):
    return _write_video(tmp_path / "v.mp4", 90)


@pytest.fixture
def empty_video(tmp_path):
    # 0フレームの mp4 は OpenCV が開けないため、開ける MJPG の .avi を使う
    return _write_video(tmp_path / "empty.avi", 0, codec="MJPG")


def test_iter_frames_downsamples_to_target_fps(video_30fps_3s):
    frames = list(iter_frames(video_30fps_3s, target_fps=5.0))
    assert len(frames) == 15
    assert frames[0].time_s == 0.0
    assert frames[1].time_s == pytest.approx(0.2, abs=0.04)


def test_missing_file_raises_video_open_error(tmp_path):
    with pytest.raises(VideoOpenError, match="開けません"):
        list(iter_frames(tmp_path / "none.mp4"))


def test_empty_video_raises_video_open_error(empty_video):
    with pytest.raises(VideoOpenError, match="フレーム"):
        list(iter_frames(empty_video))


def test_zero_fps_metadata_raises_video_open_error(monkeypatch, video_30fps_3s):
    real_cap = cv2.VideoCapture

    class ZeroFpsCapture:
        def __init__(self, *args):
            self._cap = real_cap(*args)

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 0.0
            return self._cap.get(prop)

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", ZeroFpsCapture)
    with pytest.raises(VideoOpenError, match="fps"):
        list(iter_frames(video_30fps_3s))
