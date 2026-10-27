import json

from synthetic import FakeDetector, write_blank_video, write_synthetic_match
from tennis_analyze.cli import main
from tennis_analyze.pipeline import run


def test_end_to_end_on_synthetic_match_writes_all_outputs(tmp_path):
    video = tmp_path / "m.mp4"
    boxes = write_synthetic_match(video)
    warnings = run(video, tmp_path / "out", FakeDetector(boxes))
    assert (tmp_path / "out" / "report.html").exists()
    stats = json.loads((tmp_path / "out" / "stats.json").read_text())
    assert set(stats) == {"A", "B"} and stats["A"]["distance_m"] > 0
    assert warnings == []


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
