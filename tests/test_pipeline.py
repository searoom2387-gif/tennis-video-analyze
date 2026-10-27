import csv
import json

import numpy as np
import pytest

from synthetic import (
    _SAMPLE_FPS,
    FakeDetector,
    _player_foot_m,
    write_blank_video,
    write_synthetic_match,
)
from tennis_analyze.cli import main
from tennis_analyze.court_model import COURT_LENGTH_M, COURT_WIDTH_M, SERVICE_LINE_FROM_NET_M
from tennis_analyze.pipeline import run


def _true_path_length(player: int, times: list[float], seconds: float) -> float:
    """Ground-truth path length of the generator's motion over the sampled times."""
    pts = np.array([_player_foot_m(player, t, seconds) for t in times])
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def test_end_to_end_on_synthetic_match_writes_all_outputs(tmp_path):
    seconds = 6  # write_synthetic_match's default length
    video = tmp_path / "m.mp4"
    boxes = write_synthetic_match(video, seconds=seconds)
    warnings = run(video, tmp_path / "out", FakeDetector(boxes))
    assert (tmp_path / "out" / "report.html").exists()
    assert warnings == []
    stats = json.loads((tmp_path / "out" / "stats.json").read_text())
    assert set(stats) == {"A", "B"}

    with (tmp_path / "out" / "positions.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sample_times = [i / _SAMPLE_FPS for i in range(len(boxes))]
    # Generator: player 0 is the near player (A, y 3->6 m), player 1 the far one (B, y 20->17 m).
    for label, player in (("A", 0), ("B", 1)):
        own = [r for r in rows if r["player"] == label]
        times = [float(r["time_s"]) for r in own]
        assert times == pytest.approx(sample_times, abs=1e-6)

        expected = _true_path_length(player, sample_times, seconds)
        assert stats[label]["distance_m"] == pytest.approx(expected, rel=0.15)
        duration = times[-1] - times[0]
        assert stats[label]["mean_speed_mps"] == pytest.approx(
            stats[label]["distance_m"] / duration, rel=1e-6
        )

        x0, y0 = _player_foot_m(player, 0.0, seconds)
        first = own[0]
        assert np.hypot(float(first["x_m"]) - x0, float(first["y_m"]) - y0) < 0.2

    # Zone shares from the generator's known positions (within 6.40 m of the net = front;
    # left/right as seen by the player, i.e. x mirrored on the far half). Tolerance covers
    # homography-estimation error; an inverted homography would swap/flip these clearly.
    for label, player in (("A", 0), ("B", 1)):
        pts = np.array([_player_foot_m(player, t, seconds) for t in sample_times])
        net_y = COURT_LENGTH_M / 2
        exp_front = float(np.mean(np.abs(pts[:, 1] - net_y) < SERVICE_LINE_FROM_NET_M))
        px = np.where(pts[:, 1] < net_y, pts[:, 0], COURT_WIDTH_M - pts[:, 0])
        exp_left = float(np.mean(px < COURT_WIDTH_M / 2))
        assert 0.0 < exp_front < 1.0 and 0.0 < exp_left < 1.0  # the path crosses both boundaries
        assert stats[label]["share_front"] == pytest.approx(exp_front, abs=0.15)
        assert stats[label]["share_left"] == pytest.approx(exp_left, abs=0.15)


def test_blank_court_exits_with_code_2_and_writes_check_image(tmp_path):
    video = tmp_path / "blank.mp4"
    write_blank_video(video)
    code = main([str(video), "--out", str(tmp_path / "out")])
    assert code == 2
    assert (tmp_path / "out" / "court_check.png").exists()


def test_missing_video_exits_nonzero_with_message(tmp_path, capsys):
    code = main([str(tmp_path / "none.mp4"), "--out", str(tmp_path / "out")])
    assert code != 0 and "開けません" in capsys.readouterr().err


def test_no_players_still_writes_report_with_warning(tmp_path):
    video = tmp_path / "m.mp4"
    write_synthetic_match(video)
    warnings = run(video, tmp_path / "out", FakeDetector([]))
    assert any("見つかりません" in w for w in warnings)
    assert (tmp_path / "out" / "report.html").exists()
