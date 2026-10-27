"""Command line: tennis-analyze VIDEO --out DIR [--corners ...] [--pdf] [--fps N]."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tennis_analyze.court_detect import CourtDetectionError
from tennis_analyze.pipeline import run
from tennis_analyze.video_io import VideoOpenError


def _parse_corners(text: str) -> list[tuple[float, float]]:
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        values = []
    if len(values) != 8:
        raise argparse.ArgumentTypeError(
            "x1,y1,x2,y2,x3,y3,x4,y4 の形式で8個の数値を指定してください"
        )
    return list(zip(values[0::2], values[1::2]))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tennis-analyze", description="テニス動画の選手位置を分析します")
    p.add_argument("video", type=Path, help="入力動画")
    p.add_argument("--out", type=Path, required=True, help="出力先ディレクトリ")
    p.add_argument(
        "--corners",
        type=_parse_corners,
        help="コート四隅のピクセル座標(手前左,手前右,奥右,奥左の順): x1,y1,x2,y2,x3,y3,x4,y4",
    )
    p.add_argument("--pdf", action="store_true", help="PDFレポートも出力する")
    p.add_argument("--fps", type=float, default=5.0, help="解析するフレームレート(既定 5)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    from tennis_analyze.players import TorchvisionPersonDetector  # torch itself is imported lazily

    class _Lazy:
        """Builds the real detector on first use, i.e. after court detection succeeded."""

        def __init__(self):
            self._detector = None

        def detect(self, image):
            if self._detector is None:
                self._detector = TorchvisionPersonDetector()
            return self._detector.detect(image)

    try:
        warnings = run(args.video, args.out, _Lazy(), args.corners, args.fps, args.pdf)
    except VideoOpenError as e:
        print(str(e), file=sys.stderr)
        return 1
    except CourtDetectionError as e:
        print(f"コートを検出できませんでした: {e}", file=sys.stderr)
        print(
            f"確認用画像: {args.out / 'court_check.png'}。"
            "--corners でコートの四隅を手動指定してください。",
            file=sys.stderr,
        )
        return 2
    except ImportError as e:
        print(
            f"選手検出に必要な torch / torchvision が見つかりません({e})。"
            "インストールしてから再実行してください。",
            file=sys.stderr,
        )
        return 3
    for w in warnings:
        print(f"警告: {w}", file=sys.stderr)
    print(f"出力先: {args.out}")
    return 0
