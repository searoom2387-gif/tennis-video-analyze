from collections.abc import Iterator
from pathlib import Path

import cv2

from tennis_analyze.types import Frame


class VideoOpenError(Exception):
    pass


def iter_frames(path: Path, target_fps: float = 5.0) -> Iterator[Frame]:
    """元動画から約 target_fps で間引いたフレームを返す。time_s は元動画の時刻(秒)。"""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoOpenError(f"動画を開けません: {path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 0:
            raise VideoOpenError(f"動画の fps を取得できません: {path}")
        step = max(1, round(src_fps / target_fps))

        ok, image = cap.read()
        if not ok:
            raise VideoOpenError(f"動画にフレームがありません: {path}")

        index = 0
        while ok:
            if index % step == 0:
                yield Frame(index=index, time_s=index / src_fps, image=image)
            ok, image = cap.read()
            index += 1
    finally:
        cap.release()
