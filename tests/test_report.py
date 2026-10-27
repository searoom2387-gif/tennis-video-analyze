import json

import cv2
import numpy as np

import tennis_analyze.report as report
from tennis_analyze.report import write_outputs
from tennis_analyze.stats import compute_stats
from tennis_analyze.types import PlayerSample

IMG = np.zeros((10, 10, 3), np.uint8)
SAMPLES = [PlayerSample(0.0, "A", 1.0, 2.0), PlayerSample(0.2, "A", 1.5, 2.5),
           PlayerSample(0.0, "B", 5.0, 20.0), PlayerSample(0.2, "B", 5.5, 20.5)]


def test_write_outputs_creates_all_files(tmp_path):
    paths = write_outputs(tmp_path, SAMPLES, compute_stats(SAMPLES), IMG, [])
    assert {p.name for p in paths} == {"positions.csv", "stats.json", "heatmap_A.png",
                                       "heatmap_B.png", "court_check.png", "report.html"}
    assert all(p.exists() for p in paths)


def test_creates_missing_out_dir(tmp_path):
    out = tmp_path / "a" / "b"
    write_outputs(out, SAMPLES, compute_stats(SAMPLES), IMG, [])
    assert (out / "report.html").exists()


def test_positions_csv_has_header_and_one_row_per_sample(tmp_path):
    write_outputs(tmp_path, SAMPLES, compute_stats(SAMPLES), IMG, [])
    lines = (tmp_path / "positions.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "time_s,player,x_m,y_m"
    assert len(lines) == len(SAMPLES) + 1


def test_stats_json_uses_field_names(tmp_path):
    write_outputs(tmp_path, SAMPLES, compute_stats(SAMPLES), IMG, [])
    data = json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
    assert set(data) == {"A", "B"}
    assert set(data["A"]) == {"distance_m", "mean_speed_mps", "share_left", "share_right",
                              "share_front", "share_back"}


def test_report_html_shows_warnings(tmp_path):
    write_outputs(tmp_path, SAMPLES, compute_stats(SAMPLES), IMG, ["選手が1人しか見つかりません"])
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "選手が1人しか見つかりません" in html
    assert "heatmap_A.png" in html and "court_check.png" in html


def test_report_html_escapes_warnings(tmp_path):
    write_outputs(tmp_path, SAMPLES, compute_stats(SAMPLES), IMG, ["<script>x</script>"])
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_missing_player_skips_that_heatmap(tmp_path):
    only_a = [PlayerSample(0.0, "A", 1.0, 2.0)]
    paths = write_outputs(tmp_path, only_a, compute_stats(only_a), IMG, ["w"])
    assert "heatmap_B.png" not in {p.name for p in paths}
    assert "heatmap_B.png" not in (tmp_path / "report.html").read_text(encoding="utf-8")


def test_zero_players_still_writes_files(tmp_path):
    paths = write_outputs(tmp_path, [], {}, IMG, ["選手なし"])
    assert {p.name for p in paths} == {"positions.csv", "stats.json", "court_check.png",
                                       "report.html"}
    assert (tmp_path / "positions.csv").read_text(encoding="utf-8").strip() == "time_s,player,x_m,y_m"
    assert json.loads((tmp_path / "stats.json").read_text(encoding="utf-8")) == {}


def test_heatmap_png_orientation_near_left_is_bottom_left(tmp_path):
    # x=1 (left), y=2 (near side) -> hot pixels in the lower-left of the image
    s = [PlayerSample(0.0, "A", 1.0, 2.0)]
    write_outputs(tmp_path, s, compute_stats(s), IMG, [])
    img = cv2.imread(str(tmp_path / "heatmap_A.png"))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ys, xs = np.nonzero(hsv[:, :, 1] > 100)
    assert len(ys) > 0
    h, w = img.shape[:2]
    assert xs.mean() < w / 2
    assert ys.mean() > h / 2

    s = [PlayerSample(0.0, "A", 7.0, 21.0)]
    write_outputs(tmp_path, s, compute_stats(s), IMG, [])
    img = cv2.imread(str(tmp_path / "heatmap_A.png"))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ys, xs = np.nonzero(hsv[:, :, 1] > 100)
    assert xs.mean() > w / 2
    assert ys.mean() < h / 2


def test_no_figure_leak(tmp_path):
    import matplotlib.pyplot as plt

    write_outputs(tmp_path, SAMPLES, compute_stats(SAMPLES), IMG, [])
    assert plt.get_fignums() == []


def test_pdf_without_weasyprint_adds_warning(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("weasyprint"):
            raise ImportError("no weasyprint")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    paths = write_outputs(tmp_path, SAMPLES, compute_stats(SAMPLES), IMG, [], pdf=True)
    assert "report.pdf" not in {p.name for p in paths}
    assert "weasyprint" in (tmp_path / "report.html").read_text(encoding="utf-8")


def test_pdf_not_attempted_by_default(tmp_path):
    paths = write_outputs(tmp_path, SAMPLES, compute_stats(SAMPLES), IMG, [])
    assert "report.pdf" not in {p.name for p in paths}
    assert hasattr(report, "write_outputs")
